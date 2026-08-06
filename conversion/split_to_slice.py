#!/usr/bin/env python3
"""Replace Split with an equivalent group of Slices.

In onnxruntime-web's WebGPU EP (JSEP), this model's Split falls back to
CPU, incurring a GPU<->CPU transfer per node. Slice runs on the GPU, so
assuming static shapes, each Split is expanded into one Slice per output.
Numerically identical (memory copies only).

usage: python split_to_slice.py in.onnx out.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

def main(src, dst):
    model = onnx.load(src)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    shapes = {}
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        t = v.type.tensor_type
        shapes[v.name] = [d.dim_value if d.HasField('dim_value') else None for d in t.shape.dim]

    new_nodes = []
    new_inits = []
    replaced = 0
    skipped = 0
    for node in graph.node:
        if node.op_type != 'Split':
            new_nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        shape = shapes.get(node.input[0])
        axis = attrs.get('axis', 0)
        # Split widths: use the split attribute if present, otherwise equal parts (requires static shape)
        if 'split' in attrs:
            sizes = list(attrs['split'])
        elif len(node.input) == 1 and shape is not None and shape[axis] is not None:
            dim = shape[axis]
            n = len(node.output)
            if dim % n != 0:
                skipped += 1
                new_nodes.append(node)
                continue
            sizes = [dim // n] * n
        else:
            skipped += 1
            new_nodes.append(node)
            continue

        start = 0
        for i, (out, size) in enumerate(zip(node.output, sizes)):
            mk = lambda suffix, arr: numpy_helper.from_array(
                np.array(arr, dtype=np.int64), f'{node.name}_slice{i}_{suffix}')
            starts, ends, axes = mk('starts', [start]), mk('ends', [start + size]), mk('axes', [axis])
            new_inits.extend([starts, ends, axes])
            new_nodes.append(helper.make_node(
                'Slice', [node.input[0], starts.name, ends.name, axes.name], [out],
                name=f'{node.name}_slice{i}'))
            start += size
        replaced += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    print(f'replaced {replaced} Split nodes, skipped {skipped}')
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst, save_as_external_data=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
