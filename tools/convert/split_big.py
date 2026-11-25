"""ONNX rewrite: 4D tensors larger than LIMIT bytes at 16 bit are split along channels into CG-channel chunks,
and the chunks are passed through the graph without being concatenated again.

A single 16-bit (U16/BF16) tensor larger than 128 MiB (2^27 B) is not handled by the AX8860 NPU, whatever its resolution
(384^2 x 448 ch U16 = 132.1 MB runs, 480 ch = 141.6 MB does not).
Supported: Conv / ConvTranspose (group=1) split by output channels, summed over input chunks; element-wise ops per chunk;
spatial ReduceMean per chunk; channel-aligned Slice takes chunks directly. Anything else falls back to a Concat.

    python split_big.py in.onnx out.onnx        # env SB_LIMIT (default 128 MiB), SB_CG (default 64 channels)
"""
import os, sys, numpy as np, onnx
from onnx import helper, numpy_helper, shape_inference

LIMIT = int(os.environ.get('SB_LIMIT', 128 << 20))
CG = int(os.environ.get('SB_CG', 64))
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
mi = shape_inference.infer_shapes(onnx.load(src, load_external_data=False))
shape = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
inits = {t.name: t for t in m.graph.initializer}
W = lambda n: numpy_helper.to_array(inits[n])
consts = {}
for n in m.graph.node:
    if n.op_type == "Constant":
        consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
def cval(n):
    if n in inits: return W(n)
    return consts.get(n)

big = set()
for k, s in shape.items():
    if len(s) == 4 and all(s) and s[0] * s[1] * s[2] * s[3] * 2 > LIMIT and s[1] % CG == 0:
        big.add(k)
gout = {o.name for o in m.graph.output}
assert not (big & gout), "a graph output itself exceeds the limit"
print("tensors over the limit:", len(big))

new_inits = []
uid = [0]
def add_init(arr, base):
    uid[0] += 1; name = f"{base}__sb{uid[0]}"
    new_inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name)); return name
def nm(base):
    uid[0] += 1; return f"{base}__sb{uid[0]}"

chunks = {}   # tensor -> list of chunk names
EW = {'Relu', 'Add', 'Mul', 'Sub', 'Div', 'Sqrt', 'Pow', 'Sigmoid', 'Exp', 'Neg', 'Reciprocal', 'Clip', 'Abs', 'Tanh'}
out_nodes = []
def attrs(n): return {a.name: helper.get_attribute_value(a) for a in n.attribute}
def full(t):
    """chunked tensor -> full tensor (fallback)"""
    o = nm(t + "_cat"); out_nodes.append(helper.make_node("Concat", chunks[t], [o], axis=1)); print("  fallback Concat:", t); return o
def emit_sum(parts, outname):
    acc = parts[0]
    for i, p in enumerate(parts[1:]):
        o = outname if i == len(parts) - 2 else nm(outname + "_acc")
        out_nodes.append(helper.make_node("Add", [acc, p], [o])); acc = o
    if len(parts) == 1:
        out_nodes.append(helper.make_node("Identity", [acc], [outname]))

for n in m.graph.node:
    ins = list(n.input); outs = list(n.output)
    split_in = [i for i in ins if i in chunks]
    split_out = [o for o in outs if o in big]
    if not split_in and not split_out:
        out_nodes.append(n); continue
    op = n.op_type; a = attrs(n)
    if op in ("Conv", "ConvTranspose"):
        assert a.get("group", 1) == 1, n.name
        w = W(ins[1]); b = W(ins[2]) if len(ins) > 2 and ins[2] else None
        oc_axis = 0 if op == "Conv" else 1; ic_axis = 1 - oc_axis
        Cout = w.shape[oc_axis]
        ogroups = [(j, j + CG) for j in range(0, Cout, CG)] if split_out else [(0, Cout)]
        x = ins[0]
        igroups = [(i * CG, (i + 1) * CG) for i in range(len(chunks[x]))] if x in chunks else [(0, w.shape[ic_axis])]
        xs = chunks[x] if x in chunks else [x]
        res = []
        for gi, (o0, o1) in enumerate(ogroups):
            parts = []
            for ii, (i0, i1) in enumerate(igroups):
                ww = w[o0:o1, i0:i1] if op == "Conv" else w[i0:i1, o0:o1]
                inp = [xs[ii], add_init(ww, ins[1])]
                if b is not None and ii == 0: inp.append(add_init(b[o0:o1], ins[2]))
                po = nm(outs[0] + f"_o{gi}i{ii}")
                out_nodes.append(helper.make_node(op, inp, [po], name=nm(n.name), **a)); parts.append(po)
            go = nm(outs[0] + f"_c{gi}") if split_out else outs[0]
            emit_sum(parts, go); res.append(go)
        if split_out: chunks[outs[0]] = res
    elif op in EW:
        k = len(chunks[split_in[0]]); res = []
        for gi in range(k):
            ii = []
            for i in ins:
                if i in chunks: ii.append(chunks[i][gi]); continue
                s = shape.get(i); v = cval(i)
                if v is not None and v.ndim == 4 and v.shape[1] > 1:
                    ii.append(add_init(v[:, gi * CG:(gi + 1) * CG], i))
                elif (v is not None and (v.size == 1 or (v.ndim == 4 and v.shape[1] == 1))) or (s is not None and (len(s) < 4 or s[1] == 1)):
                    ii.append(i)
                elif s is not None and len(s) == 4:
                    so = nm(i + f"_sl{gi}")
                    out_nodes.append(helper.make_node("Slice", [i, add_init(np.array([gi * CG]), "st"), add_init(np.array([(gi + 1) * CG]), "en"), add_init(np.array([1]), "ax")], [so]))
                    ii.append(so)
                else: raise SystemExit(f"{n.name}: unsupported operand {i} {s}")
            o = nm(outs[0] + f"_c{gi}"); out_nodes.append(helper.make_node(op, ii, [o], name=nm(n.name), **a)); res.append(o)
        chunks[outs[0]] = res
    elif op == "ReduceMean" and split_in and sorted(a.get("axes", cval(ins[1]) if len(ins) > 1 else [])) == [2, 3]:
        res = []
        for gi, c in enumerate(chunks[ins[0]]):
            o = nm(outs[0] + f"_c{gi}"); out_nodes.append(helper.make_node(op, [c] + ins[1:], [o], name=nm(n.name), **a)); res.append(o)
        chunks[outs[0]] = res
    elif op == "Slice" and split_in and split_out:
        st, en = cval(ins[1]), cval(ins[2]); ax = cval(ins[3])
        assert len(st) == 1 and int(ax[0]) == 1 and int(st[0]) % CG == 0 and (int(en[0]) - int(st[0])) % CG == 0, n.name
        chunks[outs[0]] = chunks[ins[0]][int(st[0]) // CG:int(en[0]) // CG]
    elif not split_out:
        ii = [full(i) if i in chunks else i for i in ins]
        out_nodes.append(helper.make_node(op, ii, outs, name=n.name, domain=n.domain, **a))
    else:
        raise SystemExit(f"unsupported: {op} {n.name} -> {split_out}")

used = set(i for n in out_nodes for i in n.input)
keep = [t for t in m.graph.initializer if t.name in used]
del m.graph.node[:]; m.graph.node.extend(out_nodes)
del m.graph.initializer[:]; m.graph.initializer.extend(keep + new_inits)
del m.graph.value_info[:]
onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=dst.split("/")[-1] + ".data")
mc = shape_inference.infer_shapes(onnx.load(dst, load_external_data=False))
left = [v.name for v in mc.graph.value_info if len(v.type.tensor_type.shape.dim) == 4 and
        np.prod([d.dim_value for d in v.type.tensor_type.shape.dim]) * 2 > LIMIT]
print("nodes", len(out_nodes), "still over the limit:", left[:10])
