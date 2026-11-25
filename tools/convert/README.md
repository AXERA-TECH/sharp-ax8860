# Tail conversion / 模型转换 (DPT + Gaussian decoder)

Scripts that rebuild `models/sharp_tail.axmodel` and `models/h4_wb.bin` from Apple's SHARP checkpoint.
The ViT part (`models/fp8vit/`) uses custom FP8 NPU kernels and is only released as prebuilt files.

把 Apple 的 SHARP 权重重新转成 `sharp_tail.axmodel` + `h4_wb.bin` 的脚本。ViT 部分（`fp8vit/`）是定制 FP8 kernel，只提供预编译文件。

```bash
pip install -e /path/to/ml-sharp onnx onnxruntime onnxsim
bash convert_tail.sh /path/to/photos [sharp_2572gikvuh.pt]   # checkpoint is downloaded if omitted
# -> work/build/sharp_tail.axmodel, work/h4_wb.bin
```

`make_calib.py` needs a CUDA GPU, the last step needs `pulsar2` (AX8860 toolchain) on `PATH`; compiling takes about 2 hours.

| Step | Script | What it does |
|---|---|---|
| 1 | `export_tail.py` | PyTorch → ONNX. The model with both ViTs replaced by inputs, rewritten so that every tensor is ≤ 768² (`sharp_split.py`: 2×2 space-to-depth image input, ConvTranspose / 3×3 / k2s2 convs re-expressed in that domain, GroupNorm decomposed, wide 3×3 convs split by input channels) |
| 2 | onnxsim | simplify |
| 3 | `split_big.py` | split 16-bit tensors > 128 MiB along channels (the NPU does not handle larger 16-bit tensors) |
| 4 | `add_h4_outputs.py` | expose the inputs of the depth head's last 1×1 conv as outputs `h4x0/h4x1` and write its fp32 weights to `h4_wb.bin`; `sharp_ax` computes this layer on the CPU |
| 5 | `make_calib.py` | tail inputs for calibration: FP32 ViTs on the GPU for each photo |
| 6 | `pulsar2 build --config tail_u16.json` | U16 activations, MinMax calibration |

Notes:

- Calibrate on many different scenes. With one or a few photos the MinMax ranges clip disparity / delta on other photos and novel views fall apart. The release model used 80 photos (indoor / outdoor, close-ups, landscapes, night, people, animals).
- BF16 or fully U8 tails compile but lose a lot of novel-view quality; U16 is the release setting.
- Each step checks itself where it can (`add_h4_outputs.py` compares the CPU h4 with the graph's own disparity output in onnxruntime).
