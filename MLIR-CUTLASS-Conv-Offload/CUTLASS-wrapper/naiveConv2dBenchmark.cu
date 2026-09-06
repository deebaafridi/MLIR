#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using namespace std;

__global__ void transposeNCHWtoNHWC(
    const float* input, 
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H * W;
    
    if (idx < total) {
        int64_t w = idx % W;
        int64_t h = (idx / W) % H;
        int64_t c = (idx / (W * H)) % C;
        int64_t n = idx / (W * H * C);
        int64_t nhwc_idx = ((n * H + h) * W + w) * C + c;
        output[nhwc_idx] = input[idx];
    }
}

__global__ void transposeNHWCtoNCHW(
    const float* input, 
    float* output,
    int64_t N, int64_t H, int64_t W, int64_t C) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * H * W * C;
    
    if (idx < total) {
        int64_t c = idx % C;
        int64_t w = (idx / C) % W;
        int64_t h = (idx / (C * W)) % H;
        int64_t n = idx / (C * W * H);
        int64_t nchw_idx = ((n * C + c) * H + h) * W + w;
        output[nchw_idx] = input[idx];
    }
}

__global__ void transposeFCHWtoKHWC(
    const float* input, 
    float* output,
    int64_t K, int64_t C, int64_t R, int64_t S) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = K * C * R * S;
    
    if (idx < total) {
        int64_t s = idx % S;
        int64_t r = (idx / S) % R;
        int64_t c = (idx / (S * R)) % C;
        int64_t k = idx / (S * R * C);
        int64_t khwc_idx = ((k * R + r) * S + s) * C + c;
        output[khwc_idx] = input[idx];
    }
}
using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::TensorNHWC;  
using LayoutB = cutlass::layout::TensorNHWC;  
using LayoutC = cutlass::layout::TensorNHWC;  

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC,
        1,
        ElementAccumulator,
        ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

void cutlass_conv2d_kernel_impl(
    float* output,     
    float* input,       
    float* filter,
    int64_t N,          
    int64_t C,          
    int64_t H,          
    int64_t W,           
    int64_t K,           
    int64_t R,           
    int64_t S,           
    int64_t P,           
    int64_t Q,           
    int64_t stride_h,   
    int64_t stride_w,    
    int64_t pad_h,       
    int64_t pad_w,      
    int64_t dilation_h,  
    int64_t dilation_w,
    float* elapsed_ms = nullptr   
) {
    float *input_nhwc = nullptr;
    float *filter_nhwc = nullptr;
    float *output_nhwc = nullptr;
    
    size_t input_size = N * C * H * W * sizeof(float);
    size_t filter_size = K * C * R * S * sizeof(float);
    size_t output_size = N * K * P * Q * sizeof(float);
    
    cudaMalloc(&input_nhwc, input_size);
    cudaMalloc(&filter_nhwc, filter_size);
    cudaMalloc(&output_nhwc, output_size);
    
    int threads = 256;
    int64_t input_elements = N * C * H * W;
    int64_t filter_elements = K * C * R * S;
    int64_t output_elements = N * K * P * Q;
    
    int blocks_input = (input_elements + threads - 1) / threads;
    int blocks_filter = (filter_elements + threads - 1) / threads;
    int blocks_output = (output_elements + threads - 1) / threads;
    
    cudaEvent_t start, end;
   cudaEventCreate(&start);
    cudaEventCreate(&end);
   cudaEventRecord(start);
    
 
    transposeNCHWtoNHWC<<<blocks_input, threads>>>(
        input, input_nhwc, N, C, H, W);
    
    transposeFCHWtoKHWC<<<blocks_filter, threads>>>(
        filter, filter_nhwc, K, C, R, S);
    
   cudaDeviceSynchronize();
   cudaGetLastError();

  
    cutlass::conv::Conv2dProblemSize problem_size(
        {static_cast<int>(N), static_cast<int>(H), static_cast<int>(W), static_cast<int>(C)},  
        {static_cast<int>(K), static_cast<int>(R), static_cast<int>(S), static_cast<int>(C)}, 
        {static_cast<int>(pad_h), static_cast<int>(pad_h), 
         static_cast<int>(pad_w), static_cast<int>(pad_w)},  
        {static_cast<int>(stride_h), static_cast<int>(stride_w)}, 
        {static_cast<int>(dilation_h), static_cast<int>(dilation_w)}, 
        {static_cast<int>(N), static_cast<int>(P), static_cast<int>(Q), static_cast<int>(K)},  
        cutlass::conv::Mode::kCrossCorrelation,
        1  
    );
    
    typename ImplicitGemm::Arguments arguments{
        problem_size,
        {input_nhwc, LayoutA::packed({N, H, W, C})},
        {filter_nhwc, LayoutB::packed({K, R, S, C})},
        {output_nhwc, LayoutC::packed({N, P, Q, K})},
        {output_nhwc, LayoutC::packed({N, P, Q, K})},
        {1.0f, 0.0f}  
    };
    
    ImplicitGemm conv_op;
    
    cutlass::Status result = conv_op.can_implement(arguments);
    if (result != cutlass::Status::kSuccess) {
        cerr << "CUTLASS cannot implement this convolution configuration" << endl;
        cudaFree(input_nhwc);
        cudaFree(filter_nhwc);
        cudaFree(output_nhwc);
        return;
    }
  
    result = conv_op.initialize(arguments);
    if (result != cutlass::Status::kSuccess) {
        cerr << "CUTLASS initialization failed" << endl;
        cudaFree(input_nhwc);
        cudaFree(filter_nhwc);
        cudaFree(output_nhwc);
        return;
    }
    
    result = conv_op();
    if (result != cutlass::Status::kSuccess) {
        cerr << "CUTLASS convolution failed" << endl;
        cudaFree(input_nhwc);
        cudaFree(filter_nhwc);
        cudaFree(output_nhwc);
        return;
    }
    
cudaDeviceSynchronize();
  
    transposeNHWCtoNCHW<<<blocks_output, threads>>>(
        output_nhwc, output, N, P, Q, K);
    
  cudaDeviceSynchronize();
    

 cudaEventRecord(end);
cudaEventSynchronize(end);

    float ms = 0;
 cudaEventElapsedTime(&ms, start, end);
    
    if (elapsed_ms != nullptr) {
        *elapsed_ms = ms;
    }
    

cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(input_nhwc);
    cudaFree(filter_nhwc);
    cudaFree(output_nhwc);
}

int main() {

    const int N = 1, C = 3, H = 224, W = 224;
    const int K = 64, R = 7, S = 7;
    const int stride_h = 2, stride_w = 2;
    const int pad_h = 3, pad_w = 3;
    const int dilation_h = 1, dilation_w = 1;
    

    const int P = 112, Q = 112;
    
    size_t inSize  = N * C * H * W * sizeof(float);
    size_t wtSize  = K * C * R * S * sizeof(float);
    size_t outSize = N * K * P * Q * sizeof(float);
    
    cout << "Input shape: [" << N << ", " << C << ", " << H << ", " << W << "]" << endl;
    cout << "Filter shape: [" << K << ", " << C << ", " << R << ", " << S << "]" << endl;
    cout << "Output shape: [" << N << ", " << K << ", " << P << ", " << Q << "]" << endl;
    cout << "Stride: [" << stride_h << ", " << stride_w << "]" << endl;
    cout << "Padding: [" << pad_h << ", " << pad_w << "]" << endl;
    cout << endl;
    
  
    vector<float> h_input(N * C * H * W, 1.0f);
    vector<float> h_weight(K * C * R * S, 1.0f);
    vector<float> h_output(N * K * P * Q, 0.0f);
    

    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, inSize);
    cudaMalloc(&d_weight, wtSize);
    cudaMalloc(&d_output, outSize);
    

  cudaMemcpy(d_input, h_input.data(), inSize, cudaMemcpyHostToDevice);
   cudaMemcpy(d_weight, h_weight.data(), wtSize, cudaMemcpyHostToDevice);
   cudaMemcpy(d_output, h_output.data(), outSize, cudaMemcpyHostToDevice);
    

    float warmup_time;
    cutlass_conv2d_kernel_impl(
        d_output, d_input, d_weight,
        N, C, H, W, K, R, S, P, Q,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w,
        &warmup_time
    );
    
    const int num_runs = 10;
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;
    
    for (int i = 0; i < num_runs; i++) {
        float elapsed_time;
        cutlass_conv2d_kernel_impl(
            d_output, d_input, d_weight,
            N, C, H, W, K, R, S, P, Q,
            stride_h, stride_w, pad_h, pad_w,
            dilation_h, dilation_w,
            &elapsed_time
        );
        
        total_time += elapsed_time;
        min_time = min(min_time, elapsed_time);
        max_time = max(max_time, elapsed_time);
        
        cout << "Run " << (i+1) << ": " << elapsed_time << " ms" << endl;
    }
    
    float avg_time = total_time / num_runs;

   cout << "Average time: " << avg_time << " ms" << endl;

cudaMemcpy(h_output.data(), d_output, outSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    
    return 0;
}