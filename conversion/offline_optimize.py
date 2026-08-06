#!/usr/bin/env python3
"""Bake in ORT's graph optimizations (basic = EP-independent) offline.

Preprocessing so the browser can load with graphOptimizationLevel:'disabled'.
The runtime's MatMulAddFusion would fuse MatMul+Add into Gemm, which falls
onto onnxruntime-web WebGPU EP's slow Gemm shader, so constant folding etc.
is done here, and Gemm is converted back to MatMul by gemm_to_matmul.py.

usage: python offline_optimize.py in.onnx out.onnx
"""
import sys
import onnxruntime as ort

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath = sys.argv[2]
ort.InferenceSession(sys.argv[1], so, providers=['CPUExecutionProvider'])
print('saved', sys.argv[2])
