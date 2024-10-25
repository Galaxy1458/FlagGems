from typing import Optional

from . import device, error

BACKEND = True
FORWARD = False

# import triton


class Register:
    def __init__(
        self,
        config: Optional[dict],
        lib: Optional[any] = None,
        debug: Optional[bool] = False,
        unused_ops_list: Optional[list] = [],
    ):
        self.device = device.Device()
        self.vendor_name, self.device_name, _ = self.device.get_vendor()
        self.DEBUG = debug
        self.backend_list = ["cuda", "MUSA"]
        self._check_backend()
        self.unused_ops_list = unused_ops_list
        self.dispatch_backend_key_prefix = "Autograd"
        self.all(config, lib)
        if debug:
            self._set_info(config)

    def _check_backend(self):
        is_support = self.device_name.lower() in self.backend_list
        if is_support is False:
            error.ErrorHandler().backend_not_support(
                self.device_name, self.backend_list
            )

    def all(self, config, lib):
        try:
            device = self.device_name.upper()
            device_auto = self.dispatch_backend_key_prefix + device
            for key, val in config.items():
                if key in self.unused_ops_list:
                    continue
                func, hasbacken = val
                if hasbacken:
                    lib.impl(key, func, device_auto)
                else:
                    lib.impl(key, func, device)
        except Exception as e:
            e
            error.ErrorHandler().register_error()

    def _set_info(self, config):
        self.config = config
        self.forward_ops = []
        self.backend_ops = []
        for val in config.values():
            fn, hasbackend = val
            fn_name = fn.__name__
            self.backend_ops.append(fn_name) if hasbackend else self.forward_ops.append(
                fn_name
            )

    def get_forward_ops(self) -> list:
        if self.DEBUG is False:
            return
        return self.forward_ops

    def get_backend_ops(self) -> list:
        if self.DEBUG is False:
            return
        return self.backend_ops

    def get_unused_ops(self) -> list:
        return self.unused_ops_list

    def get_vendor_name(self) -> list:
        return self.vendor_name

    def get_current_device(self) -> str:
        return self.device_name

    def support_backend(self, fn) -> bool:
        return fn.__name__ in self.backend_ops
