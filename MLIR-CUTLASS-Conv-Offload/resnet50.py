import torch
import torchvision
import os

from torch_mlir import fx

resnet50 = torchvision.models.resnet50(pretrained=True)
resnet50.eval()

module = fx.export_and_import(resnet50, torch.ones(1, 3, 224, 224), output_type="torch")

script_dir = os.path.dirname(os.path.abspath(__file__))

output_file = os.path.join(script_dir, "resnet50_torch.mlir")
with open(output_file, "w") as f:
    f.write(str(module))
