#!/usr/bin/env python3
"""Replace L2-clip RMSNorm (ReduceL2 -> Clip -> Div -> Mul) with a single LayerNormalization node.

Semantics: y = x * gamma / max(||x||_2, 1e-4). This differs from standard
LayerNorm, but since this model has no real LayerNorm, the JS side in
the patched ORT (patches/onnxruntime-web+*.patch) hijacks 2-input
LayerNormalization to mean this (see layerNorm).

usage: python fuse_rmsnorm.py in.onnx out.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

def main(src, dst, only_d=None):
    model = onnx.load(src)
    sh = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for v in list(sh.graph.value_info) + list(sh.graph.output):
        t = v.type.tensor_type
        shapes[v.name] = [d.dim_value if d.HasField('dim_value') else 0 for d in t.shape.dim]
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    cons = {}
    for n in graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def only_consumer(name):
        cs = cons.get(name, [])
        return cs[0] if len(cs) == 1 else None

    remove = set()
    repl = {}
    fused = 0
    skipped = 0
    for r in [n for n in graph.node if n.op_type == 'ReduceL2']:
        clip = only_consumer(r.output[0])
        if not clip or clip.op_type != 'Clip':
            skipped += 1; continue
        cmin = numpy_helper.to_array(inits[clip.input[1]]) if len(clip.input) > 1 and clip.input[1] in inits else None
        if cmin is None or abs(float(cmin) - 1e-4) > 1e-8:
            skipped += 1; continue
        div = only_consumer(clip.output[0])
        if not div or div.op_type != 'Div' or div.input[0] != r.input[0] or div.input[1] != clip.output[0]:
            skipped += 1; continue
        mul = only_consumer(div.output[0])
        if not mul or mul.op_type != 'Mul':
            skipped += 1; continue
        gamma = mul.input[1] if mul.input[1] in inits else (mul.input[0] if mul.input[0] in inits else None)
        if gamma is None:
            skipped += 1; continue
        axes = None
        for a in r.attribute:
            if a.name == 'axes':
                axes = list(helper.get_attribute_value(a))
        if axes != [-1]:
            skipped += 1; continue
        # Expand gamma to 1-D [d] (JS side reads with flat indices; scalars are broadened to [d] too)
        xshape = shapes.get(r.input[0])
        if not xshape or not xshape[-1]:
            skipped += 1; continue
        d = xshape[-1]
        if only_d and d != only_d:
            skipped += 1; continue
        garr = numpy_helper.to_array(inits[gamma]).reshape(-1)
        if garr.size == 1:
            garr = np.full(d, garr[0], dtype=garr.dtype)
        elif garr.size != d:
            skipped += 1; continue
        fname = f'{gamma}_flat{d}'
        gflat = numpy_helper.from_array(garr, fname)
        if fname not in inits:
            graph.initializer.append(gflat)
            inits[fname] = gflat
        for n in (r, clip, div, mul):
            remove.add(id(n))
        repl[id(mul)] = helper.make_node(
            'LayerNormalization', [r.input[0], gflat.name], [mul.output[0]],
            name=f'l2norm_fused_{fused}', axis=-1, epsilon=9.999999747378752e-05)
        fused += 1

    new_nodes = []
    for n in graph.node:
        if id(n) in repl:
            new_nodes.append(repl[id(n)])
        elif id(n) in remove:
            continue
        else:
            new_nodes.append(n)
    del graph.node[:]
    graph.node.extend(new_nodes)
    print(f'fused {fused} L2Norm, skipped {skipped}')
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst, save_as_external_data=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
