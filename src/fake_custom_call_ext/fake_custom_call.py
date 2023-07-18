from functools import partial, reduce
import operator

import jax.numpy as jnp
from jax.lib import xla_client
from jax import core
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

from . import gpu_ops
for _name, _value in gpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="CUDA")

class CustomCallArgsWrapper:
    def __init__(self,
                 output_types,
                 operands,
                 operand_shapes,
                 operand_specific_layouts=None,
                 output_specific_layouts=None):
        self.output_types = output_types
        self.operands = operands
        self.operand_layouts = CustomCallArgsWrapper.generate_layouts(operand_shapes,
                                                                      operand_specific_layouts)
        output_shapes = [x.shape for x in output_types]
        self.output_layouts = CustomCallArgsWrapper.generate_layouts(output_shapes,
                                                                     output_specific_layouts)

    @staticmethod
    def generate_layouts(shapes, specific_layouts):
        def default_layout(shape):
            return range(len(shape) - 1, -1, -1)

        if specific_layouts is None:
            specific_layouts = {}

        layouts = []
        for idx, shape in enumerate(shapes):
            if idx in specific_layouts:
                layouts.append(specific_layouts[idx])
            else:
                layouts.append(default_layout(shape))
        return layouts


def custom_caller(name, args, opaque, has_side_effect, **kwargs):
    out = custom_call(name,
                      args.output_types,
                      args.operands,
                      operand_layouts=args.operand_layouts,
                      result_layouts=args.output_layouts,
                      backend_config=opaque,
                      has_side_effect=has_side_effect,
                      **kwargs)
    return out

def fake_custom_call_fwd_abstract(x, gamma, **kwargs):
    assert len(gamma.shape) == 1
    assert gamma.shape[0] == x.shape[-1]

    x_dtype = jnp.float32
    rsigma_dtype = jnp.float32

    batch_shape = x.shape[:-1]
    return (
        ShapedArray(x.shape, x_dtype, named_shape=x.named_shape),    # output
        ShapedArray((*batch_shape,), rsigma_dtype, named_shape=x.named_shape),    # rsigma
    )

def fake_custom_call_fwd_lowering(ctx, x, gamma, *, epsilon):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(gamma.type)
    w_shape = w_type.shape
    iv_element_type = ir.F32Type.get()

    hidden_size = reduce(operator.mul, w_shape)
    # In Transformer, batch_size = batch x seqlen
    batch_size = reduce(operator.mul, x_shape) // hidden_size
    batch_shape = x_shape[:-1]

    out_types = [
        ir.RankedTensorType.get(x_shape, w_type.element_type),
        ir.RankedTensorType.get((*batch_shape,), iv_element_type),
    ]
    operands = [x, gamma]
    operand_shapes = [x_shape, w_shape]
    args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

    opaque = gpu_ops.build_fake_custom_call_descriptor(
        batch_size,
        hidden_size,
        epsilon,
    )

    out = custom_caller("fake_custom_call_fwd", args, opaque, False)

    return out

_fake_custom_call_fwd_p = core.Primitive("fake_custom_call_fwd")
_fake_custom_call_fwd_p.multiple_results = True
_fake_custom_call_fwd_p.def_impl(partial(xla.apply_primitive, _fake_custom_call_fwd_p))
_fake_custom_call_fwd_p.def_abstract_eval(fake_custom_call_fwd_abstract)
mlir.register_lowering(_fake_custom_call_fwd_p, fake_custom_call_fwd_lowering, platform='cuda')


def fake_custom_call_fwd(x: jnp.ndarray, gamma: jnp.ndarray, epsilon: float):
    return _fake_custom_call_fwd_p.bind(x, gamma, epsilon=epsilon)

# -------------------------------------------------------------------------------------------------------

def fake_custom_call_fwd_single_abstract(x, gamma, **kwargs):
    assert len(gamma.shape) == 1
    assert gamma.shape[0] == x.shape[-1]

    x_dtype = jnp.float32
    return ShapedArray(x.shape, x_dtype, named_shape=x.named_shape)

def fake_custom_call_fwd_single_lowering(ctx, x, gamma, *, epsilon):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(gamma.type)
    w_shape = w_type.shape

    hidden_size = reduce(operator.mul, w_shape)
    # In Transformer, batch_size = batch x seqlen
    batch_size = reduce(operator.mul, x_shape) // hidden_size

    out_types = [
        ir.RankedTensorType.get(x_shape, w_type.element_type),
    ]
    operands = [x, gamma]
    operand_shapes = [x_shape, w_shape]
    args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

    opaque = gpu_ops.build_fake_custom_call_descriptor(
        batch_size,
        hidden_size,
        epsilon,
    )

    out = custom_caller("fake_custom_call_fwd", args, opaque, False)

    return [out]

_fake_custom_call_fwd_single_p = core.Primitive("fake_custom_call_fwd_single")
_fake_custom_call_fwd_single_p.multiple_results = False
_fake_custom_call_fwd_single_p.def_impl(partial(xla.apply_primitive, _fake_custom_call_fwd_single_p))
_fake_custom_call_fwd_single_p.def_abstract_eval(fake_custom_call_fwd_single_abstract)
mlir.register_lowering(_fake_custom_call_fwd_single_p, fake_custom_call_fwd_single_lowering, platform='cuda')

def fake_custom_call_fwd_single(x: jnp.ndarray, gamma: jnp.ndarray, epsilon: float):
    return _fake_custom_call_fwd_single_p.bind(x, gamma, epsilon=epsilon)

# -------------------------------------------------------------------------------------------------------

def fake_custom_call_bwd_abstract(grad_output, rsigma, x, gamma, **kwargs):
    assert len(gamma.shape) == 1
    assert gamma.shape[0] == x.shape[-1]

    x_dtype = jnp.float32
    return (
        ShapedArray(x.shape, x_dtype, named_shape=grad_output.named_shape),    # grad input
        ShapedArray(gamma.shape, x_dtype, named_shape=gamma.named_shape),    # grad gamma
    )

def fake_custom_call_bwd_lowering(ctx, grad_output, rsigma, x, gamma, *, epsilon):
    g_type = ir.RankedTensorType(grad_output.type)
    g_shape = g_type.shape
    r_type = ir.RankedTensorType(rsigma.type)
    r_shape = r_type.shape
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(gamma.type)
    w_shape = w_type.shape

    hidden_size = reduce(operator.mul, w_shape)
    # In Transformer, batch_size = batch x seqlen
    batch_size = reduce(operator.mul, x_shape) // hidden_size

    out_types = [
        ir.RankedTensorType.get(g_shape, g_type.element_type),
        ir.RankedTensorType.get(w_shape, w_type.element_type),
    ]
    operands = [grad_output, rsigma, x, gamma]
    operand_shapes = [g_shape, r_shape, x_shape, w_shape]
    args = CustomCallArgsWrapper(out_types, operands, operand_shapes)

    opaque = gpu_ops.build_fake_custom_call_descriptor(
        batch_size,
        hidden_size,
        epsilon,
    )

    out = custom_caller("fake_custom_call_bwd", args, opaque, False)

    return out

_fake_custom_call_bwd_p = core.Primitive("fake_custom_call_bwd")
_fake_custom_call_bwd_p.multiple_results = True
_fake_custom_call_bwd_p.def_impl(partial(xla.apply_primitive, _fake_custom_call_bwd_p))
_fake_custom_call_bwd_p.def_abstract_eval(fake_custom_call_bwd_abstract)
mlir.register_lowering(_fake_custom_call_bwd_p, fake_custom_call_bwd_lowering, platform='cuda')

def fake_custom_call_bwd(g_out: jnp.ndarray, rsigma: jnp.ndarray, x: jnp.ndarray, gamma: jnp.ndarray, epsilon: float):
    return _fake_custom_call_bwd_p.bind(g_out, rsigma, x, gamma, epsilon=epsilon)