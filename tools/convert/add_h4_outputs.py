"""Move the last 1x1 conv of the depth head (h4) to the CPU.

    python add_h4_outputs.py in.onnx out.onnx h4_wb.bin

The inputs of the h4 convolutions (ReLU(h2), two 64-channel blocks after split_big.py) become extra graph outputs h4x0 / h4x1;
the NPU graph keeps its own h4 for the delta branch. h4_wb.bin = fp32 W [8,128] followed by b [8]; sharp_ax computes
disparity_u = ReLU(W @ concat(h4x0, h4x1) + b) on the CPU. Novel-view quality depends almost entirely on the disparity,
and INT8 weights in this layer alone cost ~10 dB.
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

src, dst, wb = sys.argv[1:4]
m = onnx.load(src)
I = {t.name: t for t in m.graph.initializer}
convs = [n for n in m.graph.node if n.name.startswith("/h4/") and n.op_type == "Conv"]
assert convs, "no /h4/ Conv nodes found"
xs = [n.input[0] for n in convs]
W = np.concatenate([numpy_helper.to_array(I[n.input[1]]) for n in convs], axis=1)[:, :, 0, 0].astype(np.float32)
b = sum(numpy_helper.to_array(I[n.input[2]]) for n in convs if len(n.input) > 2).astype(np.float32)
print("h4: W", W.shape, "b", b.shape, "inputs", xs)
assert W.shape == (8, 128) and len(xs) == 2
for i, x in enumerate(xs):
    m.graph.node.append(helper.make_node("Identity", [x], [f"h4x{i}"], name=f"h4x{i}_out"))
    m.graph.output.append(helper.make_tensor_value_info(f"h4x{i}", onnx.TensorProto.FLOAT, [1, 64, 768, 768]))
onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=dst.split("/")[-1] + ".data")
np.concatenate([W.ravel(), b]).astype(np.float32).tofile(wb)

# check: CPU h4 on the new outputs == the graph's disparity_u
so = ort.SessionOptions(); so.intra_op_num_threads = 16
s = ort.InferenceSession(dst, so, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(0)
feed = {i.name: rng.standard_normal(i.shape).astype(np.float32) * 0.5 for i in s.get_inputs()}
o = dict(zip([x.name for x in s.get_outputs()], s.run(None, feed)))
x = np.concatenate([o["h4x0"], o["h4x1"]], axis=1)[0].reshape(128, -1)
du = np.maximum(W @ x + b[:, None], 0).reshape(1, 8, 768, 768)
print("CPU h4 vs disparity_u: max|diff| %.3g (max %.3g)" % (np.abs(du - o["disparity_u"]).max(), np.abs(o["disparity_u"]).max()))
