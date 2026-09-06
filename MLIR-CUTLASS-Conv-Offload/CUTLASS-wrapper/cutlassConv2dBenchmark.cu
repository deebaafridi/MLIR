#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void convKernel(
     float* input,     
    float*  weight,   
    float* output           
) {
    int n  = blockIdx.x;     
    int oc = threadIdx.x;    

    if (n >= 1 || oc >= 64) return;

    for (int oh = 0; oh < 112; ++oh) {
        for (int ow = 0; ow < 112; ++ow) {
            for (int ic = 0; ic < 3; ++ic) {
                for (int kh = 0; kh < 7; ++kh) {
                    for (int kw = 0; kw < 7; ++kw) {

                        int ih = oh * 2 + kh;
                        int iw = ow * 2 + kw;

                        float inp = input[n*3*230*230 +
                                          ic*230*230 +
                                          ih*230 + iw];

                        float w = weight[oc*3*7*7 +
                                         ic*7*7 +
                                         kh*7 + kw];

                        float &out = output[n*64*112*112 +
                                            oc*112*112 +
                                            oh*112 + ow];

                        out += inp * w;
                    }
                }
            }
        }
    }
}

int main() {

    const int N = 1, C = 3, H = 230, W = 230;
    const int K = 64, R = 7, S = 7;
    const int stride_h = 2, stride_w = 2;
    const int pad_h = 3, pad_w = 3;
    
 
    const int P = 112, Q = 112;
    
    size_t inSize  = N * C * H * W * sizeof(float);
    size_t wtSize  = K * C * R * S * sizeof(float);
    size_t outSize = N * K * P * Q * sizeof(float);
    
    cout << "Input shape: [" << N << ", " << C << ", " << H << ", " << W << "]" << endl;
    cout << "Filter shape: [" << K << ", " << C << ", " << R << ", " << S << "]" << endl;
    cout << "Output shape: [" << N << ", " << K << ", " << P << ", " << Q << "]" <<endl;
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
    
    dim3 blocks(1, 1, 1);
    dim3 threads(64, 1, 1);
    
  
    cudaEvent_t start, end;
  cudaEventCreate(&start);
   cudaEventCreate(&end);


    float warmup_time;
   cudaEventRecord(start);
    convKernel<<<blocks, threads>>>(d_input, d_weight, d_output);
    cudaEventRecord(end);
    cudaEventSynchronize(end);  
   cudaEventElapsedTime(&warmup_time, start, end);
  
    const int num_runs = 10;
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;
    
    for (int i = 0; i < num_runs; i++) {
        float elapsed_time;
        
        cudaEventRecord(start);
        convKernel<<<blocks, threads>>>(d_input, d_weight, d_output);
       cudaEventRecord(end);
        cudaEventSynchronize(end);  
        cudaEventElapsedTime(&elapsed_time, start, end);
        
        total_time += elapsed_time;
        
        cout << "Run " << (i+1) << ": " << elapsed_time << " ms" << endl;
    }
    
    float avg_time = total_time / num_runs;
    
   
    cout << "Average time: " << avg_time << " ms" << endl;
 
    

    cudaMemcpy(h_output.data(), d_output, outSize, cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    
    return 0;
}
