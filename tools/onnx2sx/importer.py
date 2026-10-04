#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""ONNX -> SonicBoom S-Expr v0.1 importer.

This is an **offline development tool**, not a SonicBoom runtime dependency: it
reads an ONNX model, walks its graph, and emits (a) a SonicBoom S-Expr v0.1
document and (b) a flat little-endian external-weight sidecar that the S-Expr
references with `(data :external <file> <offset> <length>)`.

The output is ordinary S-Expr v0.1 text — nothing here bypasses the parser or
the validator, and nothing here is linked into `libsonicboom.so`. The only
runtime the generated artifacts require is the existing `sx::` pipeline
(parse -> compile -> JIT), which is already in the core library.

Supported subset (the 7 ResNet-18 operators, default ONNX domain):

    Conv  Relu  Add  MaxPool  ReduceMean  Reshape  Gemm

Everything else is rejected with an actionable error. See
`design/onnx-to-s-expr-importer.md` for the full contract.

Usage:

    models/.venv/bin/python -m onnx2sx models/resnet18.onnx \
        -o build/import/resnet18.sx \
        --weights-dir build/import
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

# ---------------------------------------------------------------------------
# Dtype mapping (ONNX TensorProto.DataType -> S-Expr name)
# ---------------------------------------------------------------------------

# The frozen ten-name S-Expr dtype set (spec §5). ONNX data types outside this
# set (uint16/uint32/uint64, complex64/128, string, float8, ...) are rejected.
ONNX_TO_SX_DTYPE = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.BFLOAT16: "bfloat16",
    TensorProto.DOUBLE: "float64",
    TensorProto.INT8: "int8",
    TensorProto.UINT8: "uint8",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.BOOL: "bool",
}

# Element types whose initializers are weights -> flat external sidecar.
_FLOAT_DTYPES = frozenset(
    {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}
)

# Integer element types whose initializers are constants -> inlined `:values`.
_INT_DTYPES = frozenset(
    {TensorProto.INT8, TensorProto.UINT8, TensorProto.INT16, TensorProto.INT32,
     TensorProto.INT64}
)

# The seven operators this importer understands, ONNX op type -> S-Expr symbol.
SUPPORTED_OPS = {
    "Conv": "conv",
    "Relu": "relu",
    "Add": "add",
    "MaxPool": "max_pool",
    "ReduceMean": "reduce_mean",
    "Reshape": "reshape",
    "Gemm": "gemm",
}


class ImportError_(Exception):
    """An actionable import failure (message rendered as a CLI diagnostic)."""


# ---------------------------------------------------------------------------
# S-Expr emission helpers
# ---------------------------------------------------------------------------


def sx_quote(name):
    """Quote a value name as an S-Expr string literal (escape \\ and ")."""
    out = ['"']
    for ch in name:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def fmt_float(v):
    """Format a float so the S-Expr lexer reads it as a Float (has '.' or exp)."""
    v = float(v)
    s = repr(v)
    if "e" in s or "E" in s:
        return s
    if "." not in s:
        s += ".0"
    return s


def sx_type(dtype_name, shape):
    """Emit a `(tensor <dtype> (shape ...))` type form."""
    dims = " ".join(str(int(d)) for d in shape)
    return f"(tensor {dtype_name} (shape {dims}))"


# ---------------------------------------------------------------------------
# Shape inference (static only) for the seven supported operators
# ---------------------------------------------------------------------------


def _prod(dims):
    n = 1
    for d in dims:
        n *= d
    return n


def _conv_pool_out(in_shape, out_channels, kernel, pads, strides, dilations):
    """Shared ONNX floor output-size rule for Conv and MaxPool (NCHW).

    output_spatial = floor((in + pad_begin + pad_end
                            - dilation * (kernel - 1) - 1) / stride) + 1

    `out_channels` is F for Conv (weight out-channels) and C for MaxPool (input
    channels); the spatial output does not depend on the channel count.
    """
    n, _c, h, w = in_shape
    kh, kw = kernel
    sy, sx = strides
    dy, dx = dilations
    pt, pl, pb, pr = pads  # ONNX pads order: [top, left, bottom, right]
    out_h = (h + pt + pb - dy * (kh - 1) - 1) // sy + 1
    out_w = (w + pl + pr - dx * (kw - 1) - 1) // sx + 1
    return (n, out_channels, out_h, out_w)


def _broadcast(a, b):
    return tuple(np.broadcast_shapes(a, b))


def _resolve_reshape(shape_const, in_shape, allowzero):
    """Resolve a Reshape target shape (0/-1 rules) into concrete ints."""
    out = [int(x) for x in shape_const]
    numel = _prod(in_shape)
    if not allowzero:
        for i, d in enumerate(out):
            if d == 0:
                if i >= len(in_shape):
                    raise ImportError_(
                        "reshape: zero dimension at index %d has no source "
                        "dimension (input rank %d)" % (i, len(in_shape))
                    )
                out[i] = int(in_shape[i])
    if -1 in out:
        idx = out.index(-1)
        known = 1
        for d in out:
            if d != -1:
                known *= d
        if known == 0 or numel % known != 0:
            raise ImportError_(
                "reshape: cannot infer -1 dimension (input numel %d, "
                "explicit product %d)" % (numel, known)
            )
        out[idx] = numel // known
    if _prod(out) != numel:
        raise ImportError_(
            "reshape: target shape %s has %d elements but input has %d"
            % (tuple(out), _prod(out), numel)
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Attribute translation
# ---------------------------------------------------------------------------


def node_attrs(node):
    """Translate an ONNX node's attributes into ordered S-Expr attr forms.

    Returns a list of "(name (tag value...))" strings. Unknown ONNX attribute
    value kinds (tensor/graph/...) are rejected: none occur in the 7 ops.
    """
    forms = []
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            forms.append((a.name, "(int %d)" % a.i))
        elif a.type == onnx.AttributeProto.FLOAT:
            forms.append((a.name, "(float %s)" % fmt_float(a.f)))
        elif a.type == onnx.AttributeProto.STRING:
            forms.append((a.name, "(string %s)" % sx_quote(a.s.decode("utf-8"))))
        elif a.type == onnx.AttributeProto.INTS:
            forms.append((a.name, "(ints " + " ".join(str(int(v)) for v in a.ints) + ")"))
        elif a.type == onnx.AttributeProto.FLOATS:
            forms.append((a.name, "(floats " + " ".join(fmt_float(v) for v in a.floats) + ")"))
        elif a.type == onnx.AttributeProto.STRINGS:
            forms.append(
                (a.name, "(strings " + " ".join(sx_quote(s.decode("utf-8")) for s in a.strings) + ")")
            )
        else:
            raise ImportError_(
                "node '%s' has attribute '%s' of unsupported ONNX value kind %d"
                % (node.op_type, a.name, a.type)
            )
    return forms


def get_attr(node, name, kind=None, default=None):
    """Fetch an ONNX attribute by name, returning `default` if absent."""
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == onnx.AttributeProto.INT:
            return a.i
        if a.type == onnx.AttributeProto.FLOAT:
            return a.f
        if a.type == onnx.AttributeProto.STRING:
            return a.s.decode("utf-8")
        if a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
        if a.type == onnx.AttributeProto.FLOATS:
            return list(a.floats)
        if a.type == onnx.AttributeProto.STRINGS:
            return [s.decode("utf-8") for s in a.strings]
        raise ImportError_(
            "node '%s' attribute '%s' has unsupported value kind" % (node.op_type, name)
        )
    return default


def _require_ints(node, name, n, op):
    vals = get_attr(node, name, onnx.AttributeProto.INTS)
    if vals is None or len(vals) != n:
        raise ImportError_(
            "%s: attribute '%s' must be %d integers (got %r)" % (op, name, n, vals)
        )
    return list(vals)


# ---------------------------------------------------------------------------
# Initializer decoding
# ---------------------------------------------------------------------------


def read_initializer(tensor, name):
    """Decode an ONNX initializer into a numpy array, with integrity checks."""
    try:
        arr = numpy_helper.to_array(tensor)
    except Exception as exc:
        raise ImportError_(
            "initializer '%s' has invalid tensor data: %s" % (name, exc)
        ) from exc
    if arr.size != _prod(tensor.dims):
        raise ImportError_(
            "initializer '%s' declares %d elements but its data has %d"
            % (name, _prod(tensor.dims), arr.size)
        )
    return arr


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def convert(model, sidecar_name="weights.bin"):
    """Convert an ONNX ModelProto to (sx_text, weights_bytes).

    `sx_text` is the S-Expr v0.1 document; `weights_bytes` is the flat
    little-endian sidecar (empty bytes if the model has no float initializers).
    Raises ImportError_ on any unsupported or malformed construct.
    """
    graph = model.graph

    # --- opset / domain ------------------------------------------------------
    default_version = None
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            default_version = imp.version
        else:
            raise ImportError_(
                "custom opset domain '%s' is not supported (v0 is default-domain only)"
                % imp.domain
            )
    if default_version is None:
        raise ImportError_("model declares no default-domain opset version")

    # --- graph inputs --------------------------------------------------------
    graph_input_names = set()
    inputs = []  # (name, dtype_name, shape)
    for vi in graph.input:
        if vi.name in graph_input_names:
            raise ImportError_("duplicate graph input '%s'" % vi.name)
        graph_input_names.add(vi.name)
        tt = vi.type.tensor_type
        dtype_name = _dtype_name(tt.elem_type, "input '%s'" % vi.name)
        shape = _static_shape(tt, "input '%s'" % vi.name)
        inputs.append((vi.name, dtype_name, shape))

    # --- initializers --------------------------------------------------------
    # name -> (elem_type, dims, numpy array). An initializer that is also a
    # graph input is an overridable input with a default; v0 has no default-
    # value mechanism, so it stays a graph input and its value is dropped.
    inits = {}
    for t in graph.initializer:
        if t.name in inits:
            raise ImportError_("duplicate initializer '%s'" % t.name)
        inits[t.name] = (t.data_type, list(t.dims), read_initializer(t, t.name))

    # --- parameters (weights -> external, integer constants -> :values) ------
    # params: list of (name, dtype_name, shape, ("external", off, len) |
    #                  ("values", [ints])).
    # const_values: name -> [ints] for integer constants (reduce/reshape axes).
    params = []
    const_values = {}
    weights_blob = bytearray()
    for name, (elem_type, dims, arr) in inits.items():
        if name in graph_input_names:
            continue  # overridable input default; not a parameter in v0
        dtype_name = _dtype_name(elem_type, "initializer '%s'" % name)
        shape = tuple(int(d) for d in dims)
        if elem_type in _FLOAT_DTYPES:
            raw = np.ascontiguousarray(arr).tobytes()
            offset = len(weights_blob)
            length = len(raw)
            weights_blob.extend(raw)
            params.append((name, dtype_name, shape, ("external", offset, length)))
        elif elem_type in _INT_DTYPES:
            vals = [int(v) for v in np.ascontiguousarray(arr).reshape(-1).tolist()]
            const_values[name] = vals
            params.append((name, dtype_name, shape, ("values", vals)))
        else:
            raise ImportError_(
                "initializer '%s' has unsupported element type %d in v0"
                % (name, elem_type)
            )

    # --- value environment for inference -------------------------------------
    # name -> (dtype_name, shape). Seeded from inputs and parameters.
    env = {}
    for name, dtype_name, shape in inputs:
        env[name] = (dtype_name, shape)
    for name, dtype_name, shape, _data in params:
        env[name] = (dtype_name, shape)

    # --- nodes ---------------------------------------------------------------
    nodes = []
    defined = set(env)  # SSA names defined so far (inputs + parameters)
    for node in graph.node:
        if node.domain not in ("", "ai.onnx"):
            raise ImportError_(
                "node '%s' uses unsupported domain '%s'" % (node.op_type, node.domain)
            )
        op = SUPPORTED_OPS.get(node.op_type)
        if op is None:
            raise ImportError_(
                "unsupported operator '%s' (supported: %s)"
                % (node.op_type, ", ".join(sorted(SUPPORTED_OPS)))
            )
        if len(node.output) != 1:
            raise ImportError_(
                "operator '%s' must produce exactly one output (got %d)"
                % (node.op_type, len(node.output))
            )

        out_name = node.output[0]
        if out_name in defined:
            raise ImportError_("duplicate definition of value '%s'" % out_name)

        for in_name in node.input:
            if in_name == "":
                raise ImportError_("node '%s' has an empty input name" % node.op_type)
            if in_name not in env:
                raise ImportError_(
                    "node '%s' references unknown value '%s'" % (node.op_type, in_name)
                )

        out_dtype, out_shape = _infer_node(node, op, env, const_values)
        env[out_name] = (out_dtype, out_shape)
        defined.add(out_name)

        attr_forms = node_attrs(node)
        nodes.append((op, list(node.input), out_name, out_dtype, out_shape, attr_forms))

    # --- graph outputs -------------------------------------------------------
    outputs = []
    for out in graph.output:
        if out.name not in env:
            raise ImportError_("graph output references unknown value '%s'" % out.name)
        outputs.append(out.name)

    return (
        _emit_sx(graph.name or "main", default_version, inputs, outputs, params, nodes,
                 sidecar_name),
        bytes(weights_blob),
    )


def _dtype_name(elem_type, where):
    name = ONNX_TO_SX_DTYPE.get(elem_type)
    if name is None:
        raise ImportError_("%s has unsupported ONNX data type %d" % (where, elem_type))
    return name


def _static_shape(tt, where):
    if not tt.HasField("shape"):
        raise ImportError_("%s has no shape (dynamic ranks are unsupported)" % where)
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value") and not d.dim_param:
            if d.dim_value < 0:
                raise ImportError_("%s has a negative dimension" % where)
            dims.append(int(d.dim_value))
        else:
            raise ImportError_(
                "%s has a dynamic dimension (only static shapes in v0)" % where
            )
    return tuple(dims)


def _infer_node(node, op, env, const_values):
    """Compute (out_dtype, out_shape) for one node using the static env."""
    def shp(name):
        return env[name][1]

    def dt(name):
        return env[name][0]

    if op == "relu":
        i = node.input[0]
        return dt(i), shp(i)

    if op == "add":
        a, b = node.input[0], node.input[1]
        da, db = dt(a), dt(b)
        if da != db:
            raise ImportError_("add: mismatched dtypes '%s' and '%s'" % (da, db))
        return da, _broadcast(shp(a), shp(b))

    if op == "conv":
        x, w = node.input[0], node.input[1]
        if get_attr(node, "group", default=1) != 1:
            raise ImportError_("conv: group != 1 is unsupported in v0")
        if get_attr(node, "auto_pad", default="NOTSET") != "NOTSET":
            raise ImportError_("conv: auto_pad != NOTSET is unsupported in v0")
        dilations = _require_ints(node, "dilations", 2, "conv")
        if dilations != [1, 1]:
            raise ImportError_("conv: dilations != [1,1] is unsupported in v0")
        pads = _require_ints(node, "pads", 4, "conv")
        strides = _require_ints(node, "strides", 2, "conv")
        _require_ints(node, "kernel_shape", 2, "conv")
        w_shape = shp(w)
        if len(w_shape) != 4:
            raise ImportError_("conv: weight must be rank 4 [F,C,KH,KW]")
        return dt(x), _conv_pool_out(shp(x), w_shape[0], (w_shape[2], w_shape[3]),
                                     pads, strides, dilations)

    if op == "max_pool":
        x = node.input[0]
        if get_attr(node, "ceil_mode", default=0) != 0:
            raise ImportError_("max_pool: ceil_mode != 0 is unsupported in v0")
        if get_attr(node, "auto_pad", default="NOTSET") != "NOTSET":
            raise ImportError_("max_pool: auto_pad != NOTSET is unsupported in v0")
        dilations = _require_ints(node, "dilations", 2, "max_pool")
        if dilations != [1, 1]:
            raise ImportError_("max_pool: dilations != [1,1] is unsupported in v0")
        kernel = _require_ints(node, "kernel_shape", 2, "max_pool")
        pads = _require_ints(node, "pads", 4, "max_pool")
        strides = _require_ints(node, "strides", 2, "max_pool")
        return dt(x), _conv_pool_out(shp(x), shp(x)[1], kernel, pads, strides,
                                     dilations)

    if op == "reduce_mean":
        x = node.input[0]
        axes_name = node.input[1]
        axes = _constant_int64(const_values, axes_name, "reduce_mean axes")
        keepdims = get_attr(node, "keepdims", default=1)
        in_shape = shp(x)
        rank = len(in_shape)
        reduced = [False] * rank
        for a in axes:
            if a < 0:
                a += rank
            if a < 0 or a >= rank:
                raise ImportError_("reduce_mean: axis %d out of range" % a)
            if reduced[a]:
                raise ImportError_("reduce_mean: duplicate axis %d" % a)
            reduced[a] = True
        if keepdims:
            out = tuple(1 if reduced[i] else in_shape[i] for i in range(rank))
        else:
            out = tuple(in_shape[i] for i in range(rank) if not reduced[i])
        return dt(x), out

    if op == "reshape":
        x = node.input[0]
        shape_name = node.input[1]
        shape_const = _constant_int64(const_values, shape_name, "reshape shape")
        allowzero = get_attr(node, "allowzero", default=0)
        return dt(x), _resolve_reshape(shape_const, shp(x), allowzero)

    if op == "gemm":
        a, b = node.input[0], node.input[1]
        transA = get_attr(node, "transA", default=0)
        transB = get_attr(node, "transB", default=0)
        a_shape, b_shape = shp(a), shp(b)
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ImportError_("gemm: A and B must be rank 2 in v0")
        m = a_shape[1] if transA else a_shape[0]
        n = b_shape[0] if transB else b_shape[1]
        if len(node.input) == 3:
            c_shape = shp(node.input[2])
            if len(c_shape) != 1 or c_shape[0] != n:
                raise ImportError_("gemm: bias must be rank 1 of length %d" % n)
        return dt(a), (m, n)

    raise ImportError_("unsupported operator '%s'" % node.op_type)  # unreachable


def _constant_int64(const_values, name, what):
    vals = const_values.get(name)
    if vals is None:
        raise ImportError_("%s must reference an int64 :values constant ('%s')" % (what, name))
    return vals


def _emit_sx(graph_name, opset_version, inputs, outputs, params, nodes, sidecar_name):
    lines = []
    lines.append(";; SonicBoom S-Expr v0.1 — imported from ONNX")
    lines.append(";; Generated by tools/onnx2sx (offline importer); not hand-authored.")
    lines.append("")
    lines.append("(sonicboom-s-expr")
    lines.append("  (version 0 1)")
    lines.append("  (graph")
    lines.append("    (name %s)" % sx_quote(graph_name))
    lines.append('    (opset "default" %d)' % opset_version)

    lines.append("    (inputs")
    for name, dtype_name, shape in inputs:
        lines.append("      (input %s %s)" % (sx_quote(name), sx_type(dtype_name, shape)))
    lines.append("      )")

    lines.append("    (outputs")
    for name in outputs:
        lines.append("      (output %s)" % sx_quote(name))
    lines.append("      )")

    lines.append("    (parameters")
    for name, dtype_name, shape, data in params:
        if data[0] == "external":
            _, offset, length = data
            data_form = "(data :external %s %d %d)" % (sx_quote(sidecar_name), offset, length)
        else:
            _, vals = data
            data_form = "(data :values " + " ".join(str(int(v)) for v in vals) + ")"
        lines.append(
            "      (parameter %s %s %s)"
            % (sx_quote(name), sx_type(dtype_name, shape), data_form)
        )
    lines.append("      )")

    lines.append("    (nodes")
    for op, input_names, out_name, out_dtype, out_shape, attr_forms in nodes:
        ins = " ".join(sx_quote(n) for n in input_names)
        out = "(%s %s)" % (sx_quote(out_name), sx_type(out_dtype, out_shape))
        # Each attribute is `(name <typed-value>)`; sorted by name for a stable,
        # canonical document (matching the frozen fixture's convention).
        attrs = " ".join(
            "(%s %s)" % (name, form) for name, form in sorted(attr_forms)
        )
        if attrs:
            lines.append(
                "      (node %s (inputs %s) (outputs %s) (attrs %s))"
                % (op, ins, out, attrs)
            )
        else:
            lines.append("      (node %s (inputs %s) (outputs %s))" % (op, ins, out))
    lines.append("      )")

    lines.append("    )")
    lines.append("  )")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Output writing (atomic, failure-safe)
# ---------------------------------------------------------------------------


def write_outputs(sx_text, weights_bytes, output_path, weights_dir, weights_file):
    """Atomically write the S-Expr and sidecar, avoiding partial artifacts.

    Both files are written to temporary names and then renamed into place; on
    any failure the temporary files are removed and no partial output is left.
    """
    tmp_sx = None
    tmp_w = None
    try:
        fd, tmp_sx = tempfile.mkstemp(dir=os.path.dirname(output_path) or ".",
                                      prefix=".sx.tmp.", suffix=".sx")
        with os.fdopen(fd, "w") as f:
            f.write(sx_text)
        os.chmod(tmp_sx, 0o644)
        os.makedirs(weights_dir, exist_ok=True)
        fd, tmp_w = tempfile.mkstemp(dir=weights_dir, prefix=".weights.tmp.")
        with os.fdopen(fd, "wb") as f:
            f.write(weights_bytes)
        os.chmod(tmp_w, 0o644)

        weights_path = os.path.join(weights_dir, weights_file)
        os.replace(tmp_w, weights_path)
        tmp_w = None
        try:
            os.replace(tmp_sx, output_path)
        except OSError:
            # Roll back the already-renamed sidecar so we don't leave a dangling
            # sidecar with no S-Expr.
            try:
                os.remove(weights_path)
            except OSError:
                pass
            raise
        tmp_sx = None
    finally:
        for p in (tmp_sx, tmp_w):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="onnx2sx",
        description="Convert an ONNX model to a SonicBoom S-Expr v0.1 document + weight sidecar.",
    )
    parser.add_argument("input", help="input ONNX model path")
    parser.add_argument("-o", "--output", help="output S-Expr file (default: <stem>.sx in cwd)")
    parser.add_argument("--weights-dir", help="directory for the weight sidecar (default: dirname of --output)")
    parser.add_argument("--weights-file", help="weight sidecar filename (default: <output-stem>.weights.bin)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print("onnx2sx: error: input file not found: %s" % args.input, file=sys.stderr)
        return 1

    output_path = args.output or (os.path.splitext(os.path.basename(args.input))[0] + ".sx")
    weights_dir = args.weights_dir or (os.path.dirname(os.path.abspath(output_path)) or ".")
    weights_file = args.weights_file or (
        os.path.splitext(os.path.basename(output_path))[0] + ".weights.bin"
    )

    try:
        model = onnx.load(args.input, load_external_data=False)
    except Exception as exc:
        print("onnx2sx: error: failed to load ONNX model: %s" % exc, file=sys.stderr)
        return 1

    try:
        sx_text, weights_bytes = convert(model, sidecar_name=weights_file)
        write_outputs(sx_text, weights_bytes, output_path, weights_dir, weights_file)
    except ImportError_ as exc:
        print("onnx2sx: error: %s" % exc, file=sys.stderr)
        return 1
    except OSError as exc:
        print("onnx2sx: error: write failed: %s" % exc, file=sys.stderr)
        return 1

    print("wrote %s (%d bytes)" % (output_path, len(sx_text)))
    print("wrote %s (%d bytes)" % (os.path.join(weights_dir, weights_file), len(weights_bytes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
