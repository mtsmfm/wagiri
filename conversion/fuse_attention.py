#!/usr/bin/env python3
"""Replace block-local + sink attention with com.microsoft::MultiHeadAttention.

Time side (Softmax last dim 56): full attention for the 8 sink queries +
sliding-window attention for the 801 queries, replaced by a single MHA node
(heads folded into the batch, 3D [B*H, S, 64], num_heads=1). The JS side
(patches/onnxruntime-web+*.patch) swaps the MHA implementation for a custom kernel
(wagiriFusedAttention) to give it meaning
(sink count and window width are hardcoded in the kernel).

Frequency side (Softmax last dim 60): plain full attention -> also MHA.

usage: python fuse_attention.py in.onnx out.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx_utils import remove_unused_initializers

def get_attr(n, name, default=None):
    for a in n.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default

def main(src, dst):
    model = onnx.load(src)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    by_out = {o: n for n in graph.node for o in n.output}
    cons = {}
    for n in graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    sh_model = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for v in list(sh_model.graph.value_info) + list(sh_model.graph.output) + list(sh_model.graph.input):
        t = v.type.tensor_type
        shapes[v.name] = tuple(d.dim_value if d.HasField('dim_value') else 0 for d in t.shape.dim)

    def const(name):
        return numpy_helper.to_array(inits[name]) if name in inits else None

    def walk_up_to(name, op_type, limit=12):
        for _ in range(limit):
            n = by_out.get(name)
            if n is None:
                return None
            if n.op_type == op_type:
                return n
            name = n.input[0]
        return None

    new_nodes = []
    remove = set()
    fused_time = 0
    fused_freq = 0

    for sm in [n for n in graph.node if n.op_type == 'Softmax']:
        in_shape = shapes.get(sm.input[0], ())
        if not in_shape:
            continue
        if in_shape[-1] == 56 and len(in_shape) == 5:
            # ---- Time side ----
            B, H, NB, QB, _ = in_shape
            concat = by_out[sm.input[0]]
            sink_mul, local_add = by_out[concat.input[0]], by_out[concat.input[1]]
            local_mul = by_out[local_add.input[0]]
            local_mm = by_out[local_mul.input[0]]
            scale = float(const(local_mul.input[1]))
            # q: local_mm.input[0] <- Reshape <- Pad <- Slice <- rope_q
            rope_q = walk_up_to(local_mm.input[0], 'RotaryEmbedding')
            # k: local_mm.input[1] <- Transpose <- Concat(3blocks) <- ... <- rope_k
            k_windows = local_mm.input[1]
            rope_k = walk_up_to(k_windows, 'RotaryEmbedding', 20)
            # Output: downstream of softmax, Slice -> MatMul(v) -> Add -> Reshape -> Slice -> Concat
            n_it = sm
            final_concat = None
            for _ in range(8):
                cs = cons.get(n_it.output[0], [])
                if not cs:
                    break
                n_it = cs[0]
                if n_it.op_type == 'Concat' and shapes.get(n_it.output[0], ())[-2:] == (809, 64):
                    final_concat = n_it
                    break
            # v: walk up from the AV MatMul's second input ([B,H,809,64] that does not pass through rope)
            sl = cons[sm.output[0]][0]
            av_mm = cons[sl.output[0]][0]
            v_win = av_mm.input[1]
            v4d = v_win
            for _ in range(20):
                n2 = by_out.get(v4d)
                if n2 is None:
                    break
                if shapes.get(v4d, ()) == (B, H, 809, 64) and n2.op_type == 'Transpose':
                    break
                v4d = n2.input[0]
            assert shapes.get(v4d) == (B, H, 809, 64), f'v not found: {v4d} {shapes.get(v4d)}'
            assert final_concat is not None

            out_name = final_concat.output[0]
            pre = f'fusedattn_t{fused_time}'
            shp3 = numpy_helper.from_array(np.array([B * H, 809, 64], np.int64), pre + '_s3')
            shp4 = numpy_helper.from_array(np.array([B, H, 809, 64], np.int64), pre + '_s4')
            graph.initializer.extend([shp3, shp4])
            for nm, t in [('q', rope_q.output[0]), ('k', rope_k.output[0]), ('v', v4d)]:
                new_nodes.append(helper.make_node('Reshape', [t, shp3.name], [f'{pre}_{nm}3'], name=f'{pre}_r{nm}'))
            new_nodes.append(helper.make_node(
                'MultiHeadAttention', [f'{pre}_q3', f'{pre}_k3', f'{pre}_v3'], [f'{pre}_o3'],
                name=pre, domain='com.microsoft', num_heads=1, scale=scale))
            new_nodes.append(helper.make_node('Reshape', [f'{pre}_o3', shp4.name], [out_name], name=f'{pre}_ro'))
            remove.add(id(final_concat))
            fused_time += 1
        elif in_shape[-1] == 60 and len(in_shape) == 4:
            # ---- Frequency side (full attention) ----
            B, H, S, _ = in_shape
            mul = by_out[sm.input[0]]
            mm = by_out[mul.input[0]]
            scale = float(const(mul.input[1]))
            rope_q = walk_up_to(mm.input[0], 'RotaryEmbedding')
            rope_k = walk_up_to(mm.input[1], 'RotaryEmbedding', 20)
            av = cons[sm.output[0]][0]
            assert av.op_type == 'MatMul'
            v4d = av.input[1]
            for _ in range(10):
                n2 = by_out.get(v4d)
                if shapes.get(v4d, ()) == (B, H, S, 64):
                    break
                v4d = n2.input[0]
            assert shapes.get(v4d) == (B, H, S, 64), f'freq v not found {v4d} {shapes.get(v4d)}'
            out_name = av.output[0]
            pre = f'fusedattn_f{fused_freq}'
            shp3 = numpy_helper.from_array(np.array([B * H, S, 64], np.int64), pre + '_s3')
            shp4 = numpy_helper.from_array(np.array([B, H, S, 64], np.int64), pre + '_s4')
            graph.initializer.extend([shp3, shp4])
            for nm, t in [('q', rope_q.output[0]), ('k', rope_k.output[0]), ('v', v4d)]:
                new_nodes.append(helper.make_node('Reshape', [t, shp3.name], [f'{pre}_{nm}3'], name=f'{pre}_r{nm}'))
            new_nodes.append(helper.make_node(
                'MultiHeadAttention', [f'{pre}_q3', f'{pre}_k3', f'{pre}_v3'], [f'{pre}_o3'],
                name=pre, domain='com.microsoft', num_heads=1, scale=scale))
            new_nodes.append(helper.make_node('Reshape', [f'{pre}_o3', shp4.name], [out_name], name=f'{pre}_ro'))
            remove.add(id(av))
            fused_freq += 1

    # Remove old nodes + add new ones
    kept = [n for n in graph.node if id(n) not in remove]
    kept.extend(new_nodes)

    # Prune dead nodes by reachability
    produced = {}
    for n in kept:
        for o in n.output:
            produced[o] = n
    needed = set(o.name for o in graph.output)
    alive = set()
    stack = list(needed)
    while stack:
        t = stack.pop()
        n = produced.get(t)
        if n is None or id(n) in alive:
            continue
        alive.add(id(n))
        stack.extend(n.input)
    final_nodes = [n for n in kept if id(n) in alive]
    # Re-sort into topological order
    ready = set(i.name for i in graph.initializer) | set(i.name for i in graph.input) | {''}
    ordered = []
    pending = final_nodes[:]
    while pending:
        rest = []
        for n in pending:
            if all(i in ready for i in n.input):
                ordered.append(n)
                ready.update(n.output)
            else:
                rest.append(n)
        if len(rest) == len(pending):
            raise RuntimeError('cycle or missing input: ' + rest[0].name)
        pending = rest
    del graph.node[:]
    graph.node.extend(ordered)

    print(f'fused time={fused_time} freq={fused_freq}, nodes {len(kept)} -> {len(ordered)}')
    print(f'removed {len(remove_unused_initializers(graph))} unused initializers')
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst, save_as_external_data=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
