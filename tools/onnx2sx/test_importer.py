#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# Unit tests for the ONNX -> S-Expr v0.1 importer. Each test builds a small
# ONNX model programmatically (onnx.helper) and asserts on the emitted S-Expr
# text / sidecar bytes, or on the structured rejection of an unsupported or
# malformed construct. No model files, no runtime, no C++ involved.
#
# Usage: models/.venv/bin/python -m unittest tools.onnx2sx.test_importer -v

import os
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onnx2sx.importer import ImportError_, convert, write_outputs  # noqa: E402


def model_for(nodes, inputs, outputs, initializers=(), graph_name="g",
              opset=20):
    graph = helper.make_graph(list(nodes), graph_name, list(inputs), list(outputs),
                              list(initializers))
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def vi(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def ftensor(name, shape, data):
    """A float32 initializer from a nested list / numpy array."""
    arr = np.asarray(data, dtype=np.float32).reshape(shape)
    return numpy_helper.from_array(arr, name)


def itensor(name, data):
    """An int64 initializer from a list."""
    arr = np.asarray(data, dtype=np.int64)
    return numpy_helper.from_array(arr, name)


class TestImporter(unittest.TestCase):
    # --- basic structure ----------------------------------------------------

    def test_relu_minimal(self):
        m = model_for(
            [helper.make_node("Relu", ["x"], ["y"])],
            [vi("x", TensorProto.FLOAT, [1, 3])],
            [vi("y", TensorProto.FLOAT, [1, 3])],
        )
        sx, w = convert(m)
        self.assertEqual(w, b"")
        self.assertIn('(input "x" (tensor float32 (shape 1 3)))', sx)
        self.assertIn('(output "y")', sx)
        self.assertIn(
            '(node relu (inputs "x") (outputs ("y" (tensor float32 (shape 1 3)))))', sx)
        self.assertNotIn("(parameter ", sx)

    def test_ssa_chain_preserves_order(self):
        # x -> Add(x, c) -> Relu -> z ; c is a float32 [3] initializer (external).
        c = ftensor("c", [3], [0.5, -1.0, 2.0])
        m = model_for(
            [helper.make_node("Add", ["x", "c"], ["s"]),
             helper.make_node("Relu", ["s"], ["z"])],
            [vi("x", TensorProto.FLOAT, [3])],
            [vi("z", TensorProto.FLOAT, [3])],
            initializers=[c],
        )
        sx, w = convert(m)
        # SSA value 'c' becomes a parameter; both nodes listed in topological order.
        self.assertIn('(parameter "c" (tensor float32 (shape 3)) (data :external "weights.bin" 0 12))', sx)
        self.assertEqual(len(w), 12)
        self.assertLess(sx.index('(node add'), sx.index('(node relu'))
        self.assertIn('(node add (inputs "x" "c")', sx)
        self.assertIn('(node relu (inputs "s")', sx)

    def test_conv_attribute_translation(self):
        W = ftensor("W", [1, 1, 3, 3], np.arange(9))
        B = ftensor("B", [1], [0.25])
        m = model_for(
            [helper.make_node("Conv", ["x", "W", "B"], ["y"],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                              strides=[1, 1], dilations=[1, 1], group=1,
                              auto_pad="NOTSET")],
            [vi("x", TensorProto.FLOAT, [1, 1, 5, 5])],
            [vi("y", TensorProto.FLOAT, [1, 1, 5, 5])],
            initializers=[W, B],
        )
        sx, w = convert(m)
        # (5 + 1 + 1 - 1*2 - 1)/1 + 1 = 5
        self.assertIn('(outputs ("y" (tensor float32 (shape 1 1 5 5))))', sx)
        # typed attribute forms translated faithfully
        self.assertIn('(string "NOTSET")', sx)
        self.assertIn("(ints 1 1)", sx)
        self.assertIn("(int 1)", sx)
        self.assertIn("(ints 1 1 1 1)", sx)
        # two external parameters in initializer order
        self.assertEqual(len(w), 9 * 4 + 4)
        self.assertIn('(data :external "weights.bin" 0 36)', sx)
        self.assertIn('(data :external "weights.bin" 36 4)', sx)

    def test_reduce_mean_and_reshape_constants(self):
        axes = itensor("axes", [-1, -2])
        shape = itensor("shape", [1, 512])
        m = model_for(
            [helper.make_node("ReduceMean", ["x", "axes"], ["m"], keepdims=1),
             helper.make_node("Reshape", ["m", "shape"], ["y"], allowzero=1)],
            [vi("x", TensorProto.FLOAT, [1, 512, 7, 7])],
            [vi("y", TensorProto.FLOAT, [1, 512])],
            initializers=[axes, shape],
        )
        sx, w = convert(m)
        # int64 constants are inlined with :values, not external
        self.assertIn('(parameter "axes" (tensor int64 (shape 2)) (data :values -1 -2))', sx)
        self.assertIn('(parameter "shape" (tensor int64 (shape 2)) (data :values 1 512))', sx)
        self.assertEqual(w, b"")  # no float weights
        self.assertIn('(node reduce_mean (inputs "x" "axes") (outputs ("m" (tensor float32 (shape 1 512 1 1))))', sx)
        self.assertIn('(node reshape (inputs "m" "shape") (outputs ("y" (tensor float32 (shape 1 512))))', sx)

    def test_gemm_translation(self):
        A = vi("A", TensorProto.FLOAT, [1, 512])
        B = ftensor("B", [1000, 512], np.zeros((1000, 512)))
        C = ftensor("C", [1000], np.zeros(1000))
        m = model_for(
            [helper.make_node("Gemm", ["A", "B", "C"], ["Y"], transB=1, alpha=1.0, beta=1.0)],
            [A],
            [vi("Y", TensorProto.FLOAT, [1, 1000])],
            initializers=[B, C],
        )
        sx, w = convert(m)
        self.assertIn('(node gemm (inputs "A" "B" "C") (outputs ("Y" (tensor float32 (shape 1 1000))))', sx)
        self.assertIn("(float 1.0)", sx)
        self.assertIn("transB (int 1)", sx)

    def test_initializer_also_graph_input(self):
        # c is both a graph input and an initializer: v0 has no input defaults,
        # so it must be emitted as a graph input, not a parameter.
        c = ftensor("c", [3], [1.0, 2.0, 3.0])
        m = model_for(
            [helper.make_node("Add", ["x", "c"], ["y"])],
            [vi("x", TensorProto.FLOAT, [3]), vi("c", TensorProto.FLOAT, [3])],
            [vi("y", TensorProto.FLOAT, [3])],
            initializers=[c],
        )
        sx, w = convert(m)
        self.assertIn('(input "c" (tensor float32 (shape 3)))', sx)
        self.assertNotIn('(parameter "c"', sx)
        self.assertEqual(w, b"")

    # --- rejection ----------------------------------------------------------

    def test_unsupported_operator(self):
        m = model_for(
            [helper.make_node("Gelu", ["x"], ["y"])],
            [vi("x", TensorProto.FLOAT, [4])],
            [vi("y", TensorProto.FLOAT, [4])],
        )
        with self.assertRaises(ImportError_) as cm:
            convert(m)
        self.assertIn("unsupported operator 'Gelu'", str(cm.exception))

    def test_unsupported_dtype(self):
        # UINT16 is not in the frozen ten-name S-Expr dtype set.
        m = model_for(
            [helper.make_node("Relu", ["x"], ["y"])],
            [vi("x", TensorProto.UINT16, [4])],
            [vi("y", TensorProto.UINT16, [4])],
        )
        with self.assertRaises(ImportError_) as cm:
            convert(m)
        self.assertIn("unsupported ONNX data type", str(cm.exception))

    def test_dynamic_shape_rejected(self):
        inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3])
        m = model_for(
            [helper.make_node("Relu", ["x"], ["y"])],
            [inp],
            [vi("y", TensorProto.FLOAT, [4, 3])],
        )
        with self.assertRaises(ImportError_) as cm:
            convert(m)
        self.assertIn("dynamic dimension", str(cm.exception))

    def test_missing_attr(self):
        W = ftensor("W", [1, 1, 3, 3], np.zeros((1, 1, 3, 3)))
        m = model_for(
            [helper.make_node("Conv", ["x", "W"], ["y"], pads=[1, 1, 1, 1],
                              strides=[1, 1], dilations=[1, 1])],  # no kernel_shape
            [vi("x", TensorProto.FLOAT, [1, 1, 5, 5])],
            [vi("y", TensorProto.FLOAT, [1, 1, 3, 3])],
            initializers=[W],
        )
        with self.assertRaises(ImportError_) as cm:
            convert(m)
        self.assertIn("kernel_shape", str(cm.exception))

    def test_malformed_attr(self):
        W = ftensor("W", [1, 1, 3, 3], np.zeros((1, 1, 3, 3)))
        m = model_for(
            [helper.make_node("Conv", ["x", "W"], ["y"], kernel_shape=[3, 3],
                              pads=[1, 1], strides=[1, 1], dilations=[1, 1])],
            [vi("x", TensorProto.FLOAT, [1, 1, 5, 5])],
            [vi("y", TensorProto.FLOAT, [1, 1, 3, 3])],
            initializers=[W],
        )
        with self.assertRaises(ImportError_) as cm:
            convert(m)
        self.assertIn("pads", str(cm.exception))

    def test_invalid_tensor_data(self):
        # Declares 2 float32 elements but carries only 4 raw bytes (1 element).
        t = onnx.TensorProto()
        t.name = "c"
        t.data_type = TensorProto.FLOAT
        t.dims.extend([2])
        t.raw_data = b"\x00\x00\x00\x00"
        m = model_for(
            [helper.make_node("Add", ["x", "c"], ["y"])],
            [vi("x", TensorProto.FLOAT, [2])],
            [vi("y", TensorProto.FLOAT, [2])],
            initializers=[t],
        )
        with self.assertRaises(ImportError_) as cm:
            convert(m)
        self.assertIn("invalid tensor data", str(cm.exception))

    def test_duplicate_value_definition(self):
        m = model_for(
            [helper.make_node("Relu", ["x"], ["y"]),
             helper.make_node("Relu", ["y"], ["y"])],  # 'y' defined twice
            [vi("x", TensorProto.FLOAT, [3])],
            [vi("y", TensorProto.FLOAT, [3])],
        )
        with self.assertRaises(ImportError_) as cm:
            convert(m)
        self.assertIn("duplicate definition of value 'y'", str(cm.exception))

    def test_output_references_unknown(self):
        m = model_for(
            [helper.make_node("Relu", ["x"], ["y"])],
            [vi("x", TensorProto.FLOAT, [3])],
            [vi("missing", TensorProto.FLOAT, [3])],
        )
        with self.assertRaises(ImportError_) as cm:
            convert(m)
        self.assertIn("unknown value 'missing'", str(cm.exception))


class TestWriteOutputs(unittest.TestCase):
    def test_writes_both_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "g.sx")
            write_outputs("(x)\n", b"\x01\x02\x03", out, d, "g.weights.bin")
            with open(out) as f:
                self.assertEqual(f.read(), "(x)\n")
            with open(os.path.join(d, "g.weights.bin"), "rb") as f:
                self.assertEqual(f.read(), b"\x01\x02\x03")
            # no stray temp files
            self.assertEqual([n for n in os.listdir(d) if ".tmp." in n], [])

    def test_failure_leaves_no_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            # A file occupies the weights-dir name -> makedirs must fail.
            blocker = os.path.join(d, "not-a-dir")
            with open(blocker, "w") as f:
                f.write("x")
            out = os.path.join(d, "g.sx")
            with self.assertRaises(OSError):
                write_outputs("(x)\n", b"\x01", out, blocker, "g.weights.bin")
            self.assertFalse(os.path.exists(out))
            # nothing but the blocker file should be present
            self.assertEqual(sorted(os.listdir(d)), ["not-a-dir"])


if __name__ == "__main__":
    unittest.main()
