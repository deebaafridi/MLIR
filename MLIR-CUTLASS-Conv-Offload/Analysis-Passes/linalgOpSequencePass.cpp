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
  CollectLinalgPattern(MLIRContext *context,
                       vector<linalg::LinalgOp> &operations)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context),
        collectedOps(operations) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {

    collectedOps.push_back(dyn_cast<linalg::LinalgOp>(op.getOperation()));

    return failure();
  }

private:
  vector<linalg::LinalgOp> &collectedOps;
};

struct LinalgOperation
    : public PassWrapper<LinalgOperation, OperationPass<func::FuncOp>> {

  StringRef getArgument() const final { return "linalg-op-matcher"; }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgOperation)

  void runOnOperation() override {

    func::FuncOp func = getOperation();
    vector<linalg::LinalgOp> linalgOps;

    RewritePatternSet patterns(&getContext());
    patterns.add<CollectLinalgPattern>(&getContext(), linalgOps);
    (void)applyPatternsGreedily(func, std::move(patterns));

    for (int i = 0; i < (int)linalgOps.size(); ++i) {
      llvm::outs() << "\nCurrent Linalg Operation: " <<
      linalgOps[i]->getName()
                   << "\n";

      llvm::outs() << "Previous two operations:\n";
      if (i - 2 >= 0) {
        llvm::outs() << "  - " << linalgOps[i - 2]->getName() << "\n";
      }
      if (i - 1 >= 0) {
        llvm::outs() << "  - " << linalgOps[i - 1]->getName() << "\n";
      }

      llvm::outs() << "Next two operations:\n";
      if (i + 1 < (int)linalgOps.size()) {
        llvm::outs() << "  - " << linalgOps[i + 1]->getName() << "\n";
      }
      if (i + 2 < (int)linalgOps.size()) {
        llvm::outs() << "  - " << linalgOps[i + 2]->getName() << "\n";
      }
    }
  }
};

} // namespace

namespace mlir {

void registerLinalgOp() { PassRegistration<LinalgOperation>(); }

} // namespace mlir
