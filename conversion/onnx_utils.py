"""Shared helpers for the custom ONNX graph-rewrite passes."""

from onnx import GraphProto


def remove_unused_initializers(graph: GraphProto) -> list[str]:
    """Remove top-level initializers no node or graph output references."""
    referenced = {name for node in graph.node for name in node.input if name}
    referenced.update(output.name for output in graph.output)
    unused = [initializer for initializer in graph.initializer
              if initializer.name not in referenced]
    removed = [initializer.name for initializer in unused]
    for initializer in unused:
        graph.initializer.remove(initializer)
    return removed
