from functools import partial

import os
os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_proto --xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=nested_xmap"

import numpy as np
import jax
from jax import vmap, random
import jax.numpy as jnp
from jax.interpreters import batching
from jax.sharding import Mesh
from jax.experimental.maps import xmap
from jax.experimental.shard_map import shard_map
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec, NamedSharding
from fake_custom_call_ext.fake_custom_call import _fake_custom_call_fwd_p
from fake_custom_call_ext.fake_custom_call import CustomCallArgsWrapper, custom_caller, gpu_ops

from jax.interpreters import pxla
_PXLA_THREAD_RESOURCES = pxla.thread_resources

PIPE=2
BATCH=32
HIDDEN=128

PP_SIZE=2
DP_SIZE=4

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)

from functools import reduce
import operator
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.abstract_arrays import ShapedArray

def fake_custom_call_fwd_abstract(x, gamma, *, mesh=None, **kwargs):
    print(mesh, "abstract")
    # assert len(gamma.shape) == (len(x.shape) - 1)
    assert gamma.shape[-1] == x.shape[-1]

    x_dtype = jnp.float32
    rsigma_dtype = jnp.float32

    batch_shape = x.shape[:-1]
    return (
        ShapedArray(x.shape, x_dtype, named_shape=x.named_shape),    # output
        ShapedArray((*batch_shape,), rsigma_dtype, named_shape=x.named_shape),    # rsigma
    )

def fake_custom_call_fwd_lowering(ctx, x, gamma, *, mesh, epsilon):
    print(mesh, "lowering")

    if mesh is None:
        x_type = ir.RankedTensorType(x.type)
        x_shape = x_type.shape
        w_type = ir.RankedTensorType(gamma.type)
        w_shape = w_type.shape
        iv_element_type = ir.F32Type.get()

        hidden_size = reduce(operator.mul, w_shape)
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
    else:
        def inner(x_i, gamma_i):

            partial_fake_custom_call_fwd = partial(fake_custom_call_fwd, epsilon=epsilon)
            res = xmap(partial_fake_custom_call_fwd,
                    in_axes=({0: "pipeline"}, {0: "pipeline"}),
                    out_axes=[{0: "pipeline"}, {0: "pipeline"}],
                    axis_resources={"pipeline":"pp"})(x_i, gamma_i)
            # res = shard_map(partial_fake_custom_call_fwd, mesh, 
            #                 in_specs=(PartitionSpec('pp', None, None), PartitionSpec('pp', None)),
            #                 out_specs=(PartitionSpec('pp', None, None)), check_rep=False)(x_i, gamma_i)
            return res
        return mlir.lower_fun(inner)(ctx, x, gamma)

_fake_custom_call_fwd_p.def_abstract_eval(fake_custom_call_fwd_abstract)
mlir.register_lowering(_fake_custom_call_fwd_p, fake_custom_call_fwd_lowering, platform='cuda')

def fake_custom_call_fwd(x: jnp.ndarray, gamma: jnp.ndarray, *, mesh=None, epsilon=1e-6):
    return _fake_custom_call_fwd_p.bind(x, gamma, mesh=mesh, epsilon=epsilon)


def fake_custom_call_batch_rule(vector_arg_values, batch_axes, *, mesh, epsilon):
    return fake_custom_call_fwd(*vector_arg_values, mesh=mesh, epsilon=epsilon), [batch_axes[0], batch_axes[0]]

batching.primitive_batchers[_fake_custom_call_fwd_p] = fake_custom_call_batch_rule


def xmap_runner(func, in_axes,
                out_axes, axis_resources,
                inputs):
    """
    xmap_runner
    """
    assert isinstance(inputs, tuple)
    assert isinstance(in_axes, tuple)

    mesh = _PXLA_THREAD_RESOURCES.env.physical_mesh
    fake_in_axes = {}
    fake_axis_resource = {}

    # Fake related setup is a workaround to "NotImplementedError:
    # Collectives in manually partitioned computations are only supported
    # when all mesh axes are partitioned manually (no partial automatic
    # sharding). Make sure that you mention all mesh axes in axis_resources!"
    for i, mesh_axis_names in enumerate(mesh.axis_names):
        if mesh_axis_names not in axis_resources.values():
            fake_axis_name = f"{mesh_axis_names}_fake_{i}"
            fake_in_axes[i] = fake_axis_name
            fake_axis_resource[fake_axis_name] = mesh_axis_names

    fake_input = jnp.zeros(tuple(64 for _ in range(len(fake_in_axes) + 1)))

    xmapped = xmap(lambda func_input, _: func(*func_input),
                   in_axes=(in_axes, fake_in_axes),
                   out_axes=out_axes,
                   axis_resources={
                       **axis_resources,
                       **fake_axis_resource
                   })

    output = xmapped(inputs, fake_input)
    return output


def fake_cc(x, gamma, mesh):

    in_axes = ({0: "data"}, {})
    out_axes = {0: "data"}
    axis_resource = {"data":'dp'}

    x_ = jnp.reshape(x, (DP_SIZE, BATCH // DP_SIZE, HIDDEN))

    run_func = partial(_fake_cc, mesh=mesh, epsilon=1e-5)

    output = xmap_runner(run_func,
                        in_axes,
                        out_axes,
                        axis_resource,
                        (x_, gamma))
    return output

@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _fake_cc(x, gamma, mesh, epsilon):
    output, _ = _fake_cc_fwd(x, gamma, mesh=mesh, epsilon=epsilon)
    return output


def _fake_cc_fwd(x, gamma, mesh, epsilon):
    outs, rsigma = fake_custom_call_fwd(x, gamma, mesh=mesh, epsilon=epsilon)
    return outs, (rsigma, x, gamma)


def _fake_cc_bwd(mesh, epsilon, ctx, g):
    _, x, gamma = ctx

    out = x + g

    return (out, gamma)

_fake_cc.defvjp(_fake_cc_fwd, _fake_cc_bwd)

def func(x, gamma, mesh):
    partial_fake_cc = partial(fake_cc, mesh=mesh)
    out = vmap(partial_fake_cc, in_axes=(0, 0, ))(x, gamma)
    return jnp.mean(out)


devices = np.array(jax.local_devices()).reshape((PP_SIZE, DP_SIZE))
with Mesh(devices, ('pp', 'dp')) as mesh:

    partial_func = partial(func, mesh=mesh)

    pjitter = pjit(partial_func,
                   in_axis_resources=[PartitionSpec('pp', 'dp', None), PartitionSpec('pp', None)],
                   out_axis_resources=None,
              )

    x_ = random.normal(random.PRNGKey(1124), (PIPE, BATCH, HIDDEN))
    x = jax.device_put(x_, NamedSharding(mesh, PartitionSpec('pp', 'dp', None)))
    gamma_ = jnp.ones((PIPE, HIDDEN))
    gamma = jax.device_put(gamma_, NamedSharding(mesh, PartitionSpec('pp', None)))
    outs = pjitter(x, gamma)
    print(outs)
