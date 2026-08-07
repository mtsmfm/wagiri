#!/usr/bin/env python3
"""Kernel fusion and redundant-node removal for onnxruntime-web WebGPU (JSEP).

- Gelu fusion: replace the x*0.5*(1+erf(x/sqrt2)) subgraph with com.microsoft::Gelu
  (JSEP has a Gelu kernel; 5 dispatches -> 1 dispatch)
- Redundant Expand removal: when an input to a broadcastable op is explicitly
  Expanded, drop the Expand and let the op's own broadcasting handle it
  (numerically identical)

usage: python fuse_for_jsep.py in.onnx out.onnx
"""
import sys
import onnx
from onnx import helper, numpy_helper
from onnx_utils import remove_unused_initializers

def to_scalar(init):
    a = numpy_helper.to_array(init)
    return float(a.reshape(-1)[0]) if a.size == 1 else None

def main(src, dst):
    model = onnx.load(src)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    by_out = {o: n for n in graph.node for o in n.output}
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    graph_outputs = {o.name for o in graph.output}

    removed = set()
    replacements = []  # (anchor_node, new_node)
    renames = {}

    # ---- Gelu fusion ----
    # Shape from torch export: Div(x, sqrt2) -> Erf -> Add(1) -> Mul(x, .) -> Mul(0.5)
    # or with the Muls in a different order. Replace the final output with Gelu(x).
    gelu_count = 0
    for erf in [n for n in graph.node if n.op_type == 'Erf']:
        div = by_out.get(erf.input[0])
        if div is None or div.op_type != 'Div' or div.input[1] not in inits:
            continue
        s = to_scalar(inits[div.input[1]])
        if s is None or abs(s - 1.4142135) > 1e-3:
            continue
        x = div.input[0]
        adds = consumers.get(erf.output[0], [])
        if len(adds) != 1 or adds[0].op_type != 'Add':
            continue
        add = adds[0]
        other = add.input[0] if add.input[1] == erf.output[0] else add.input[1]
        if other not in inits or to_scalar(inits[other]) != 1.0:
            continue
        mul1s = consumers.get(add.output[0], [])
        if len(mul1s) != 1 or mul1s[0].op_type != 'Mul':
            continue
        mul1 = mul1s[0]
        other1 = mul1.input[0] if mul1.input[1] == add.output[0] else mul1.input[1]
        final = None
        if other1 == x:
            # Mul(x, 1+erf) -> Mul(0.5)
            mul2s = consumers.get(mul1.output[0], [])
            if len(mul2s) == 1 and mul2s[0].op_type == 'Mul':
                mul2 = mul2s[0]
                o2 = mul2.input[0] if mul2.input[1] == mul1.output[0] else mul2.input[1]
                if o2 in inits and to_scalar(inits[o2]) == 0.5:
                    final = mul2
        elif other1 in inits and to_scalar(inits[other1]) == 0.5:
            # Mul(0.5, 1+erf) -> Mul(x)
            mul2s = consumers.get(mul1.output[0], [])
            if len(mul2s) == 1 and mul2s[0].op_type == 'Mul':
                mul2 = mul2s[0]
                o2 = mul2.input[0] if mul2.input[1] == mul1.output[0] else mul2.input[1]
                if o2 == x:
                    final = mul2
        else:
            # Mul(x*0.5, 1+erf) form: other1's producer is Mul(x, 0.5)
            p = by_out.get(other1)
            if p is not None and p.op_type == 'Mul':
                po = p.input[0] if p.input[1] in inits else p.input[1]
                pc = p.input[1] if p.input[1] in inits else p.input[0]
                if po == x and pc in inits and to_scalar(inits[pc]) == 0.5:
                    final = mul1
                    removed.add(id(p))
        if final is None:
            continue
        for n in (div, erf, add, mul1, final):
            removed.add(id(n))
        gelu = helper.make_node('Gelu', [x], [final.output[0]],
                                name=f'gelu_fused_{gelu_count}', domain='com.microsoft')
        replacements.append((final, gelu))
        gelu_count += 1

    # ---- Redundant Expand removal ----
    # If Expand(v)'s output is used only by elementwise binary ops, use v directly
    expand_count = 0
    BCAST_OPS = {'Div', 'Mul', 'Add', 'Sub'}
    for exp in [n for n in graph.node if n.op_type == 'Expand']:
        if exp.output[0] in graph_outputs:
            continue
        cs = consumers.get(exp.output[0], [])
        if cs and all(c.op_type in BCAST_OPS for c in cs):
            removed.add(id(exp))
            renames[exp.output[0]] = exp.input[0]
            expand_count += 1

    new_nodes = []
    for n in graph.node:
        if id(n) in removed:
            for old, new in replacements:
                if old is n:
                    new_nodes.append(new)
            continue
        for i, name in enumerate(n.input):
            if name in renames:
                n.input[i] = renames[name]
        new_nodes.append(n)
    del graph.node[:]
    graph.node.extend(new_nodes)

    print(f'fused {gelu_count} Gelu, removed {expand_count} Expand')
    print(f'removed {len(remove_unused_initializers(graph))} unused initializers')
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst, save_as_external_data=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
