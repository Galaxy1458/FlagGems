import ast

from .cambricon import *  # noqa: F403
from .iluvatar import *  # noqa: F403
from .kunlunxin import *  # noqa: F403
from .mlu import *  # noqa: F403
from .mthreads import *  # noqa: F403

vendor_list = [
    "nvidia",
    "cambricon",
    "iluvatar",
    "kunlunxin",
    "mlu",
    "mthreads",
    "dcu",
    "ascend",
]


class scheduler:
    def __init__(self):
        pass

    @staticmethod
    def get_device_guard_fn(vendor_name):
        code = f"""
from . import {vendor_name}
fn = {vendor_name}.device.get_torch_device_guard_fn()
"""
        try:
            parsed_ast = ast.parse(code)
            compiled_code = compile(parsed_ast, filename="<ast>", mode="exec")
            exec(compiled_code, globals())
        except Exception as e:
            raise RuntimeError(e)

        return globals()["fn"]

    @staticmethod
    def get_vendor_info(vendor_name):
        code = f"""
from . import {vendor_name}
info = {vendor_name}.device.get_vendor_info()
"""
        parsed_ast = ast.parse(code)
        compiled_code = compile(parsed_ast, filename="<ast>", mode="exec")
        try:
            exec(compiled_code, globals())
        except Exception as e:
            RuntimeError(e)
        return globals()["info"]

    @staticmethod
    def get_vendor_infos() -> list:
        infos = []
        for vendor_name in vendor_list:
            try:
                infos.append(scheduler.get_vendor_info(vendor_name))
            except Exception as e:
                e
        return infos

    @staticmethod
    def get_curent_device_extend_op(vendor_name) -> dict:
        code = f"""
            fn = {vendor_name}.Op.get_current_extend_ops()
        """
        code


def device_guard_fn(vendor_name):
    return scheduler.get_device_guard_fn(vendor_name)


__all__ = ["scheduler" "device_guard_fn"]
