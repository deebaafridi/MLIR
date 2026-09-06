# MLIR Passes

This repository contains MLIR compiler engineering passes for lowering and compiling deep learning models to GPU code. Models are exported to MLIR using torch-mlir, then compiled through MLIR passes and offloaded to hand-tuned GPU kernels.
