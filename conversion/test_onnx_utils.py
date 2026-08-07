import unittest

from onnx import TensorProto, helper

from conversion.onnx_utils import remove_unused_initializers


class RemoveUnusedInitializersTest(unittest.TestCase):
    def test_removes_only_unreferenced_initializers(self):
        used = helper.make_tensor('used', TensorProto.FLOAT, [1], [1.0])
        output_value = helper.make_tensor('output_value', TensorProto.FLOAT, [1], [2.0])
        unused = helper.make_tensor('unused', TensorProto.FLOAT, [1], [3.0])
        graph = helper.make_graph(
            [helper.make_node('Identity', ['used'], ['node_output'])],
            'main', [],
            [helper.make_tensor_value_info('node_output', TensorProto.FLOAT, [1]),
             helper.make_tensor_value_info('output_value', TensorProto.FLOAT, [1])],
            [used, output_value, unused],
        )

        removed = remove_unused_initializers(graph)

        self.assertEqual(removed, ['unused'])
        self.assertEqual([item.name for item in graph.initializer], ['used', 'output_value'])


if __name__ == '__main__':
    unittest.main()
