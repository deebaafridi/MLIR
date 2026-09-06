#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct ConvToCutlassPattern : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    Location loc = convOp.getLoc();
    ModuleOp module = convOp->getParentOfType<ModuleOp>();
    
    if (!module) {
      return rewriter.notifyMatchFailure(convOp, "no parent module found");
    }
  
    Value input = convOp.getDpsInputOperand(0)->get();
    Value filter = convOp.getDpsInputOperand(1)->get();
    Value output = convOp.getDpsInitOperand(0)->get();
    
    bool isMemRef = llvm::isa<MemRefType>(input.getType());
    
    auto inputType = llvm::cast<ShapedType>(input.getType());
    auto filterType = llvm::cast<ShapedType>(filter.getType());
    auto outputType = llvm::cast<ShapedType>(output.getType());
    
    if (!inputType.hasStaticShape() || !filterType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, 
          "only static shapes are currently supported");
    }
    
    ArrayRef<int64_t> inputShape = inputType.getShape();   
    ArrayRef<int64_t> filterShape = filterType.getShape(); 
    ArrayRef<int64_t> outputShape = outputType.getShape(); 
  
    SmallVector<Value> dims;
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, inputShape[0]));  
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, inputShape[1]));  
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, inputShape[2]));  
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, inputShape[3]));  
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, filterShape[0])); 
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, filterShape[2])); 
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, filterShape[3])); 
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, outputShape[2])); 
    dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, outputShape[3])); 
    
    auto strides = convOp.getStrides();
    auto dilations = convOp.getDilations();
    
    Value strideH, strideW, dilationH, dilationW;
    
    if (strides && strides.size() == 2) {
      auto strideValues = strides.getValues<int64_t>();
      auto it = strideValues.begin();
      strideH = rewriter.create<arith::ConstantIndexOp>(loc, *it++);
      strideW = rewriter.create<arith::ConstantIndexOp>(loc, *it);
    } else {
      strideH = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      strideW = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    }
    
    if (dilations && dilations.size() == 2) {
      auto dilationValues = dilations.getValues<int64_t>();
      auto it = dilationValues.begin();
      dilationH = rewriter.create<arith::ConstantIndexOp>(loc, *it++);
      dilationW = rewriter.create<arith::ConstantIndexOp>(loc, *it);
    } else {
      dilationH = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      dilationW = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    }
  
    Value padH = rewriter.create<arith::ConstantIndexOp>(loc, filterShape[2] / 2);
    Value padW = rewriter.create<arith::ConstantIndexOp>(loc, filterShape[3] / 2);
   
    func::FuncOp kernelFunc = getOrCreateCutlassConvKernel(module, rewriter, isMemRef);
    
    Value outputCasted, inputCasted, filterCasted;
    if (isMemRef) {
      outputCasted = rewriter.create<memref::CastOp>(
          loc, UnrankedMemRefType::get(rewriter.getF32Type(), 0), output);
      inputCasted = rewriter.create<memref::CastOp>(
          loc, UnrankedMemRefType::get(rewriter.getF32Type(), 0), input);
      filterCasted = rewriter.create<memref::CastOp>(
          loc, UnrankedMemRefType::get(rewriter.getF32Type(), 0), filter);
    } else {
      outputCasted = rewriter.create<tensor::CastOp>(
          loc, UnrankedTensorType::get(rewriter.getF32Type()), output);
      inputCasted = rewriter.create<tensor::CastOp>(
          loc, UnrankedTensorType::get(rewriter.getF32Type()), input);
      filterCasted = rewriter.create<tensor::CastOp>(
          loc, UnrankedTensorType::get(rewriter.getF32Type()), filter);
    }
  
    SmallVector<Value> args = {outputCasted, inputCasted, filterCasted};
    args.append(dims);
    args.append({strideH, strideW, padH, padW, dilationH, dilationW});
  
    rewriter.create<func::CallOp>(loc, kernelFunc, args);
    
    if (isMemRef) {
      rewriter.eraseOp(convOp);
    } else {
      auto resultType = convOp.getResult(0).getType();
  Value result = rewriter.create<tensor::CastOp>(loc, resultType, outputCasted);
  rewriter.replaceOp(convOp, result);
    }
    
    return success();
  }
  
private:
  func::FuncOp getOrCreateCutlassConvKernel(ModuleOp module,
                                           PatternRewriter &rewriter,
                                           bool isMemRef) const {
    StringRef kernelName = "cutlass_conv2d_kernel";
    
    if (auto existingFunc = module.lookupSymbol<func::FuncOp>(kernelName))
      return existingFunc;
    
    Type tensorOrMemRefType;
    if (isMemRef) {
      tensorOrMemRefType = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    } else {
      tensorOrMemRefType = UnrankedTensorType::get(rewriter.getF32Type());
    }
    
    auto indexType = rewriter.getIndexType();
    
  
    SmallVector<Type> inputTypes = {
        tensorOrMemRefType,  
        tensorOrMemRefType,  
        tensorOrMemRefType, 
        indexType, indexType, indexType, indexType,  
        indexType, indexType, indexType,              
        indexType, indexType,                          
        indexType, indexType,                        
        indexType, indexType,                          
        indexType, indexType                         
    };
        // Type resultType = RankedTensorType::get(
        // {ShapedType::kDynamic,
        //  ShapedType::kDynamic,
        //  ShapedType::kDynamic,
        //  ShapedType::kDynamic},
        // rewriter.getF32Type());
    Type resultType = UnrankedTensorType::get(rewriter.getF32Type());

    auto funcType = rewriter.getFunctionType(inputTypes,  ArrayRef<Type>{resultType});
    
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    
    auto funcOp = rewriter.create<func::FuncOp>(module.getLoc(), 
                                                kernelName, funcType);
    funcOp.setPrivate();
    funcOp->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
    
    return funcOp;
  }
};

struct CutlassConvLoweringPass
    : public PassWrapper<CutlassConvLoweringPass, 
                        OperationPass<func::FuncOp>> {
  
  StringRef getArgument() const final { return "cutlass-conv-lowering"; }
  
  StringRef getDescription() const final {
    return "Lower linalg conv2d operations to CUTLASS kernel calls";
  }
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CutlassConvLoweringPass)
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect,
                    func::FuncDialect,
                    arith::ArithDialect,
                    memref::MemRefDialect,
                    tensor::TensorDialect>();
  }
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    
    RewritePatternSet patterns(context);
    patterns.add<ConvToCutlassPattern>(context);
    
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    
    if (failed(applyPatternsAndFoldGreedily(func, frozenPatterns))) {
      signalPassFailure();
      return;
    }
    
  }
};

} 

namespace mlir {

void registerCutlassConvBiasPass() {
  PassRegistration<CutlassConvLoweringPass>();
}

} 