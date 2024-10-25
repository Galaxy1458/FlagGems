import torch

from .op import get_register_config


# from ..common import base_device
class device:
    def __init__(self):
        pass

    @staticmethod
    def get_device_from_sys_cmd():
        return "nvidia-smi"

    @staticmethod
    def get_dispatch_key():
        return "cuda"

    @staticmethod
    def get_vendor_info():
        return ("nvidia", "cuda", "nvidia-smi")

    @staticmethod
    def get_vendor_name():
        return "nvidia"

    @staticmethod
    def get_torch_device_guard_fn():
        return torch.cuda.device


class Op:
    def __init__(self):
        pass

    @staticmethod
    def get_current_extend_ops():
        return get_register_config()
