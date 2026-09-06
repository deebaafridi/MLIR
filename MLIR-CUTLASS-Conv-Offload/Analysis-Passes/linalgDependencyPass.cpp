#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace std;


namespace {

struct CollectLinalgPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  CollectLinalgPattern(
      MLIRContext *context,
      llvm::MapVector<mlir::Value,
                      llvm::MapVector<linalg::LinalgOp,
                                      llvm::SmallVector<mlir::Value>>> &map)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context),
        collectedDependency(map) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {

    Value lhs = op->getResult(0);
    llvm::SmallVector<Value> inArguments(op.getDpsInputs());
    collectedDependency[lhs][op] = inArguments;
    return failure();
  }

private:
  llvm::MapVector<mlir::Value, llvm::MapVector<linalg::LinalgOp,
                                               llvm::SmallVector<mlir::Value>>>
      &collectedDependency;
};

struct LinalgOperation
    : public PassWrapper<LinalgOperation, OperationPass<func::FuncOp>> {

  StringRef getArgument() const final { return "linalg-op-matcher"; }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgOperation)

  llvm::SmallVector<linalg::LinalgOp> findDependencies(
      linalg::LinalgOp currentOp,
      const llvm::MapVector<
          mlir::Value,
          llvm::MapVector<linalg::LinalgOp, llvm::SmallVector<mlir::Value>>>
          &dependancyMap) {

    llvm::SmallVector<linalg::LinalgOp> dependencies;
    llvm::SmallVector<Value> args(currentOp.getDpsInputs());

    for (Value arg : args) {
      auto it = dependancyMap.find(arg);
      if (it != dependancyMap.end()) {
        for (auto &opPair : it->second) {
          linalg::LinalgOp dependentOp = opPair.first;
          dependencies.push_back(dependentOp);
        }
      }
    }

    return dependencies;
  }

  void runOnOperation() override {

    func::FuncOp func = getOperation();
    llvm::MapVector<
        mlir::Value,
        llvm::MapVector<linalg::LinalgOp, llvm::SmallVector<mlir::Value>>>
        dependancyMap;

    RewritePatternSet patterns(&getContext());
    patterns.add<CollectLinalgPattern>(&getContext(), dependancyMap);
    (void)applyPatternsGreedily(func, std::move(patterns));

    mlir::AsmState asmState(func);

    for (auto it = dependancyMap.rbegin(); it != dependancyMap.rend(); ++it) {

      Value lhs = it->first;
      llvm::outs() << "LHS Value: ";
      lhs.printAsOperand(llvm::outs(), asmState);
      llvm::outs() << "\n";

      for (auto &inner : it->second) {

        linalg::LinalgOp op = inner.first;
        llvm::outs() << "Associated LinalgOp: " << op->getName() << "\n";
        llvm::outs() << "Inputs: ";

        for (auto val : inner.second) {
          val.printAsOperand(llvm::outs(), asmState);
          llvm::outs() << ", ";
        }
        llvm::outs() << "\n";

        llvm::SmallVector<linalg::LinalgOp> deps =
            findDependencies(op, dependancyMap);
        if (!deps.empty()) {
          llvm::outs() << "Dependent on: ";
          for (linalg::LinalgOp depOp : deps) {
            llvm::outs() << depOp->getName();
            llvm::outs() << "(";
            depOp->getResult(0).printAsOperand(llvm::outs(), asmState);
            llvm::outs() << "), ";
          }
          llvm::outs() << "\n";
        } else {
          llvm::outs() << "Direct Dependencies: None\n";
        }
      }
     
      llvm::outs() << "\n";
    }
  }
};

} // namespace

namespace mlir {

void registerLinalgOp() { PassRegistration<LinalgOperation>(); }

} // namespace mlir
