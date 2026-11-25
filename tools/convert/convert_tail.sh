#!/bin/bash
# Build sharp_tail.axmodel + h4_wb.bin from the SHARP checkpoint.
#
#   bash convert_tail.sh <photo dir> [checkpoint]      # run from this directory; results in ./work/
#
# Needs: ml-sharp installed (pip install -e ml-sharp), onnx, onnxruntime, onnxsim, a CUDA GPU for make_calib.py,
# and pulsar2 (AX8860 toolchain) on PATH. The pulsar2 build takes about 2 hours.
set -e
cd "$(dirname "$0")"
PHOTOS=${1:?usage: convert_tail.sh <photo dir> [checkpoint]}
CKPT=${2:+--ckpt $2}
mkdir -p work

python export_tail.py work/tail_raw.onnx $CKPT                       # 1. PyTorch -> ONNX (768^2 rewrite)
python -m onnxsim work/tail_raw.onnx work/tail_sim.onnx                # 2. simplify
python split_big.py work/tail_sim.onnx work/tail_split.onnx           # 3. split 16-bit tensors > 128 MiB
python -m onnxsim work/tail_split.onnx work/tail_split_sim.onnx
python add_h4_outputs.py work/tail_split_sim.onnx work/sharp_tail.onnx work/h4_wb.bin   # 4. last depth-head layer -> CPU
python make_calib.py "$PHOTOS" work/calib $CKPT                       # 5. calibration inputs (GPU)
pulsar2 build --config tail_u16.json                                  # 6. compile, U16 activations

echo "done: work/build/sharp_tail.axmodel work/h4_wb.bin -> copy both into models/"
