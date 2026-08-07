#!/usr/bin/env python3
"""Fuse legacy exported RoPE subgraphs into com.microsoft::RotaryEmbedding.

New models emit RotaryEmbedding directly in export_onnx.py. This pass remains
available for models produced by older versions of the exporter.

Interleaved RoPE from the torch export:
  x [B, N, S, 64]
  rot = reshape(concat(unsq(-x_odd), unsq(x_even)), x.shape)   # (-x1, x0) pairs
  out = x * cos_full + rot * sin_full
is equivalent to RotaryEmbedding(interleaved=1):
  out[2j]   = x[2j] cos_j - x[2j+1] sin_j
  out[2j+1] = x[2j+1] cos_j + x[2j] sin_j
cos_full/sin_full are [.., S, 64] constants where each value repeats per pair,
so they are reduced to a [S, 32] cache before passing. position_ids is scalar 0
(the JSEP kernel treats a scalar as an implicit arange).

usage: python fuse_rope.py in.onnx out.onnx
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx_utils import remove_unused_initializers

def main(src, dst):
    model = onnx.load(src)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    by_out = {o: n for n in graph.node for o in n.output}

    def producer(name):
        return by_out.get(name)

    def const_or_none(name):
        return numpy_helper.to_array(inits[name]) if name in inits else None

    removed = set()
    new_nodes_at = {}  # id(add_node) -> replacement node
    added_inits = {}
    fused = 0
    skipped = 0

    pos_ids = numpy_helper.from_array(np.array([0], dtype=np.int64), 'rope_position_ids_0')

    for add in [n for n in graph.node if n.op_type == 'Add' and '/rope' in (n.name or '')]:
        mul_cos = producer(add.input[0])
        mul_sin = producer(add.input[1])
        if not mul_cos or not mul_sin or mul_sin.op_type != 'Mul' or mul_cos.op_type not in ('Mul', 'Transpose'):
            skipped += 1
            continue
        # cos side: one of the inputs is an initializer
        def split_mul(m):
            if m.input[1] in inits:
                return m.input[0], m.input[1]
            if m.input[0] in inits:
                return m.input[1], m.input[0]
            return None, None
        # Variant pattern: the cos multiply happens in the pre-transpose layout with a Transpose right before the Add
        #   Add( Transpose(Mul(x_pre, cos)), Mul(rot(Transpose'(x_pre)), sin) )
        # If Transpose and Transpose' share the same perm, this equals RoPE applied to the transposed tensor
        outer_transpose = None
        if mul_cos.op_type == 'Transpose':
            outer_transpose = mul_cos
            mul_cos = producer(outer_transpose.input[0])
            if not mul_cos or mul_cos.op_type != 'Mul':
                skipped += 1
                continue
        x_cos, cos_name = split_mul(mul_cos)
        rot_out, sin_name = split_mul(mul_sin)
        if x_cos is None or rot_out is None:
            skipped += 1
            continue
        # Trace the rotate path: Reshape_1 <- Concat <- [Unsq(Neg(Gather1)), Unsq(Gather0)] <- Reshape <- x
        rsh1 = producer(rot_out)
        if not rsh1 or rsh1.op_type != 'Reshape':
            skipped += 1
            continue
        concat = producer(rsh1.input[0])
        if not concat or concat.op_type != 'Concat' or len(concat.input) != 2:
            skipped += 1
            continue
        u0, u1 = producer(concat.input[0]), producer(concat.input[1])
        if not u0 or not u1 or u0.op_type != 'Unsqueeze' or u1.op_type != 'Unsqueeze':
            skipped += 1
            continue
        neg = producer(u0.input[0])
        g_even = producer(u1.input[0])
        if not neg or neg.op_type != 'Neg' or not g_even or g_even.op_type != 'Gather':
            skipped += 1
            continue
        g_odd = producer(neg.input[0])
        if not g_odd or g_odd.op_type != 'Gather':
            skipped += 1
            continue
        i_even = const_or_none(g_even.input[1])
        i_odd = const_or_none(g_odd.input[1])
        if i_even is None or i_odd is None or int(i_even) != 0 or int(i_odd) != 1:
            skipped += 1
            continue
        rsh0 = producer(g_even.input[0])
        if not rsh0 or rsh0.op_type != 'Reshape' or g_odd.input[0] != g_even.input[0]:
            skipped += 1
            continue
        x_rot = rsh0.input[0]
        if outer_transpose is None:
            if x_rot != x_cos:
                skipped += 1
                continue
        else:
            # x_rot must be x_cos (= x_pre) transposed with the same perm
            def perm(n):
                for a in n.attribute:
                    if a.name == 'perm':
                        return list(helper.get_attribute_value(a))
                return None
            tr = producer(x_rot)
            if not tr or tr.op_type != 'Transpose' or tr.input[0] != x_cos or perm(tr) != perm(outer_transpose):
                skipped += 1
                continue

        # Extract cos/sin caches: squeeze to [S, 64], verify pairwise duplication, reduce to [S, 32]
        cos_full = const_or_none(cos_name)
        sin_full = const_or_none(sin_name)
        if cos_full is None or sin_full is None:
            skipped += 1
            continue
        cos2 = np.squeeze(cos_full)
        sin2 = np.squeeze(sin_full)
        if cos2.ndim != 2 or not np.array_equal(cos2[:, ::2], cos2[:, 1::2]) or not np.array_equal(sin2[:, ::2], sin2[:, 1::2]):
            skipped += 1
            continue
        cos_cache = np.ascontiguousarray(cos2[:, ::2])
        sin_cache = np.ascontiguousarray(sin2[:, ::2])
        ck = ('cos', cos_cache.shape, cos_cache.tobytes())
        sk = ('sin', sin_cache.shape, sin_cache.tobytes())
        if ck not in added_inits:
            nm = f'rope_cos_cache_{len(added_inits)}'
            added_inits[ck] = numpy_helper.from_array(cos_cache, nm)
        if sk not in added_inits:
            nm = f'rope_sin_cache_{len(added_inits)}'
            added_inits[sk] = numpy_helper.from_array(sin_cache, nm)

        for n in (mul_cos, mul_sin, rsh1, concat, u0, u1, neg, g_even, g_odd, rsh0):
            removed.add(id(n))
        if outer_transpose is not None:
            removed.add(id(outer_transpose))
        removed.add(id(add))
        new_nodes_at[id(add)] = helper.make_node(
            'RotaryEmbedding',
            [x_rot, pos_ids.name, added_inits[ck].name, added_inits[sk].name],
            [add.output[0]],
            name=f'rope_fused_{fused}', domain='com.microsoft', interleaved=1)
        fused += 1

    new_nodes = []
    for n in graph.node:
        if id(n) in new_nodes_at:
            new_nodes.append(new_nodes_at[id(n)])
        elif id(n) in removed:
            continue
        else:
            new_nodes.append(n)
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.append(pos_ids)
    graph.initializer.extend(added_inits.values())

    print(f'fused {fused} RotaryEmbedding, skipped {skipped}')
    print(f'removed {len(remove_unused_initializers(graph))} unused initializers')
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, dst, save_as_external_data=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
