import threading

import torch

from . import testing  # noqa: F401
from .fused import *  # noqa: F403
from .ops import *  # noqa: F403
from .runtime import vendors
from .runtime.backend import action, device_guard_fn
from .runtime.register import Register  # noqa: F403

__version__ = "2.1"

aten_lib = torch.library.Library("aten", "IMPL")

global register
register = None


class entry:
    def __init__(self) -> None:
        pass


def enable(lib=aten_lib, unused_ops_list=[], vendor=None):
    lock = threading.Lock()
    BACKEND = action.BACKEND
    FORWARD = action.FORWARD
    global register
    lock.acquire()
    if register is not None:
        return register
    register = Register(
        config=(
            ("abs", abs, FORWARD),
            ("add.Tensor", add, FORWARD),
            ("addmm", addmm, FORWARD),
            ("bitwise_and.Tensor", bitwise_and_tensor, FORWARD),
            ("bitwise_and.Scalar", bitwise_and_scalar, FORWARD),
            ("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor, FORWARD),
            ("bitwise_not", bitwise_not, FORWARD),
            ("bitwise_or.Tensor", bitwise_or_tensor, FORWARD),
            ("bitwise_or.Scalar", bitwise_or_scalar, FORWARD),
            ("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor, FORWARD),
            ("bmm", bmm, FORWARD),
            ("clamp", clamp, FORWARD),
            ("clamp.Tensor", clamp_tensor, FORWARD),
            ("cos", cos, FORWARD),
            ("cumsum", cumsum, FORWARD),
            # "div.Tensor",div, FORWARD),
            ("native_dropout", native_dropout, BACKEND),
            ("erf", erf, FORWARD),
            ("embedding", embedding, BACKEND),
            ("eq.Tensor", eq, FORWARD),
            ("eq.Scalar", eq_scalar, FORWARD),
            ("exp", exp, FORWARD),
            ("ge.Tensor", ge, FORWARD),
            ("ge.Scalar", ge_scalar, FORWARD),
            ("gelu", gelu, BACKEND),
            ("native_group_norm", group_norm, BACKEND),
            ("gt.Tensor", gt, FORWARD),
            ("gt.Scalar", gt_scalar, FORWARD),
            ("isinf", isinf, FORWARD),
            ("isnan", isnan, FORWARD),
            ("native_layer_norm", layer_norm, FORWARD),
            ("le.Tensor", le, FORWARD),
            ("le.Scalar", le_scalar, FORWARD),
            ("lt.Tensor", lt, FORWARD),
            ("lt.Scalar", lt_scalar, FORWARD),
            ("rms_norm", rms_norm, FORWARD),
            ("mean", mean, FORWARD),
            ("mean.dim", mean_dim, FORWARD),
            ("mm", mm, FORWARD),
            ("mul.Tensor", mul, FORWARD),
            ("mv", mv, FORWARD),
            ("ne.Tensor", ne, FORWARD),
            ("ne.Scalar", ne_scalar, FORWARD),
            ("neg", neg, FORWARD),
            ("pow.Scalar", pow_scalar, FORWARD),
            ("pow.Tensor_Scalar", pow_tensor_scalar, FORWARD),
            ("pow.Tensor_Tensor", pow_tensor_tensor, FORWARD),
            ("reciprocal", reciprocal, FORWARD),
            ("relu", relu, FORWARD),
            ("rsqrt", rsqrt, FORWARD),
            ("sigmoid", sigmoid, FORWARD),
            ("silu", silu, FORWARD),
            ("sin", sin, FORWARD),
            ("softmax.int", softmax, FORWARD),
            ("sub.Tensor", sub, FORWARD),
            ("tanh", tanh, FORWARD),
            ("triu", triu, FORWARD),
            ("var_mean.correction", var_mean, FORWARD),
            ("linalg_vector_norm", vector_norm, FORWARD),
            ("where.self", where_self, FORWARD),
            ("where.ScalarSelf", where_scalar_self, FORWARD),
            ("where.ScalarOther", where_scalar_other, FORWARD),
            ("max", max, FORWARD),
            ("max.dim", max_dim, FORWARD),
            ("min", min, FORWARD),
            ("min.dim", min_dim, FORWARD),
            ("amax", amax, FORWARD),
            ("argmax", argmax, FORWARD),
            ("prod", prod, FORWARD),
            ("prod.dim_int", prod_dim, FORWARD),
            ("sum", sum, FORWARD),
            ("sum.dim_IntList", sum_dim, FORWARD),
            ("all", all, FORWARD),
            ("all.dim", all_dim, FORWARD),
            ("all.dims", all_dims, FORWARD),
            ("any", any, FORWARD),
            ("any.dim", any_dim, FORWARD),
            ("any.dims", any_dims, FORWARD),
            ("log_softmax.int", log_softmax, FORWARD),
            ("outer", outer, FORWARD),
            ("cross_entropy_loss", cross_entropy_loss, FORWARD),
        ),
        lib=lib,
        debug=True,
        unused_ops_list=unused_ops_list,
        default_vendor=vendor,
    )
    lock.release()


class use_gems:
    def __init__(self, unused_ops_list=[], vendor=None):
        self.lib = torch.library.Library("aten", "IMPL")
        self.unused_ops_list = unused_ops_list
        self.vendor = vendor

    def __enter__(self):
        enable(self.lib, unused_ops_list=self.unused_ops_list, vendor=self.vendor)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lib


def get_forward_ops():
    return register.get_forward_ops()


def get_backend_ops():
    return register.get_backend_ops()


def device():
    return register.get_current_device()


def vendor():
    return register.get_vendor_name()


def support_backend(fn):
    return register.support_backend(fn)


def device_guard(device_info):
    return device_guard_fn(vendor())(device_info)


__all__ = ["enable", "use_gems", "get_forward_ops" "device" "device_guard", "vendors"]
