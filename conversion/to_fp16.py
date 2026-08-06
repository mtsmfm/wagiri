"""Convert the ONNX model to fp16 to halve distribution size (I/O stays fp32)."""
import argparse
import onnx
from onnxconverter_common import float16

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('dst')
args = ap.parse_args()

model = onnx.load(args.src)
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, args.dst)
print('saved', args.dst)
