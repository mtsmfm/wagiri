#!/usr/bin/env python3
"""Replace Gemm with MatMul (+Add).

onnxruntime-web's WebGPU EP runs Gemm with a naive shader, while MatMul
has a fast tiled path. Gemm nodes originating from nn.Linear (2D input)
(alpha=1, beta=1, transA=0) are equivalently rewritten as MatMul + Add
with the weight initializer pre-transposed. The weight values themselves
are unchanged (transpose only), so numerical results are identical up to
operation-order differences.

usage: python gemm_to_matmul.py in.onnx out.onnx
"""
import sys
import onnx
from onnx import helper, numpy_helper
from onnx_utils import remove_unused_initializers

def main(src, dst):
    model = onnx.load(src)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}

    new_nodes = []
    replaced = 0
    skipped = 0
    for node in graph.node:
        if node.op_type != 'Gemm':
            new_nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        alpha = attrs.get('alpha', 1.0)
        beta = attrs.get('beta', 1.0)
        transA = attrs.get('transA', 0)
        transB = attrs.get('transB', 0)
        a_in, b_in = node.input[0], node.input[1]
        c_in = node.input[2] if len(node.input) > 2 else None
        if alpha != 1.0 or beta != 1.0 or transA != 0 or b_in not in inits:
            skipped += 1
            new_nodes.append(node)
            continue

        if transB:
            w = numpy_helper.to_array(inits[b_in])
            wt = numpy_helper.from_array(w.T.copy(), b_in + '_T')
            graph.initializer.remove(inits[b_in])
            graph.initializer.append(wt)
            inits[b_in + '_T'] = wt
            del inits[b_in]
            b_name = b_in + '_T'
        else:
            b_name = b_in

        out = node.output[0]
        if c_in is not None:
            mm_out = out + '_mm'
            new_nodes.append(helper.make_node('MatMul', [a_in, b_name], [mm_out], name=node.name + '_mm'))
            new_nodes.append(helper.make_node('Add', [mm_out, c_in], [out], name=node.name + '_bias'))
        else:
            new_nodes.append(helper.make_node('MatMul', [a_in, b_name], [out], name=node.name + '_mm'))
        replaced += 1

    del graph.node[:]
    graph.node.extend(new_nodes)
    print(f'replaced {replaced} Gemm nodes, skipped {skipped}')
    print(f'removed {len(remove_unused_initializers(graph))} unused initializers')
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst, save_as_external_data=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
