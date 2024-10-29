import ast
from enum import Enum


class action(Enum):
    BACKEND = True
    FORWARD = False


class vendors(Enum):
    NVIDIA = 0
    CAMBRICON = 1
    ASCEND = 2
    ILUVATAR = 3
    MTHREAD = 4
    KUNLUNXIN = 5
    HYGON = 7


vendors_map = {
    "nvidia": vendors.NVIDIA,
    "cambricon": vendors.CAMBRICON,
    "iluvatar": vendors.ILUVATAR,
    "kunlunxin": vendors.KUNLUNXIN,
    "mthreads": vendors.MTHREAD,
    "hygon": vendors.HYGON,
    "ascend": vendors.ASCEND,
}


def PASS(e):
    pass


class scheduler:
    def __init__(self):
        pass

    @staticmethod
    def get_device_guard_fn(vendor_name):
        code = f"""
from . import {vendor_name}
fn = {vendor_name}.device.get_torch_device_guard_fn()
"""
        return scheduler.get_codegen_result(code, "fn")

    @staticmethod
    def get_vendor_info(vendor_name):
        code = f"""
from . import {vendor_name}
info = {vendor_name}.device.get_vendor_info()
"""
        return scheduler.get_codegen_result(code, "info")

    @staticmethod
    def get_codegen_result(code, result_key):
        parsed_ast = ast.parse(code)
        compiled_code = compile(parsed_ast, filename="<ast>", mode="exec")
        try:
            exec(compiled_code, globals())
        except Exception as e:
            RuntimeError(e)
        return globals()[result_key]

    @staticmethod
    def get_vendor_infos() -> list:
        infos = []
        for vendor_name in vendors_map:
            try:
                single_info = scheduler.get_vendor_info(vendor_name)
                infos.append(single_info + (vendors_map[single_info[0]],))
            except Exception as e:
                PASS(e)
        return infos

    @staticmethod
    def get_curent_device_extend_op(vendor_name) -> dict:
        code = f"""
fn = {vendor_name}.Op.get_current_extend_ops()
        """
        return scheduler.get_codegen_result(code, "fn")


def device_guard_fn(vendor_name):
    return scheduler.get_device_guard_fn(vendor_name)


__all__ = ["scheduler" "device_guard_fn"]
