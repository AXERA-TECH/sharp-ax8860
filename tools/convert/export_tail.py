"""Export the tail (TailNPU768) to ONNX.

    python export_tail.py out.onnx [--ckpt sharp_2572gikvuh.pt]

Inputs: image_u [1,12,768,768], vit_out [35,1024,24,24], lat0 / lat1 [25,1024,24,24], img_vit [1,1024,24,24];
outputs: disparity_u [1,8,768,768], delta [1,28,768,768]. Run onnxsim on the result before split_big.py.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sharp_split import TailNPU768, load, replace_groupnorm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ckpt", default=None, help="SHARP checkpoint (default: download)")
a = ap.parse_args()

m = load(a.ckpt, "cpu")
print("GroupNorm replaced:", replace_groupnorm(m))
tn = TailNPU768(m).eval()
g = torch.Generator().manual_seed(0)
args = (torch.rand(1, 12, 768, 768, generator=g), torch.randn(35, 1024, 24, 24, generator=g),
        torch.randn(25, 1024, 24, 24, generator=g), torch.randn(25, 1024, 24, 24, generator=g),
        torch.randn(1, 1024, 24, 24, generator=g))
with torch.no_grad():
    torch.onnx.export(tn, args, a.out, input_names=["image_u", "vit_out", "lat0", "lat1", "img_vit"],
                      output_names=["disparity_u", "delta"], opset_version=17, dynamo=False)
print("wrote", a.out)
