from typing import Optional

from . import backend, device, error

BACKEND = True
FORWARD = False

# import triton


class Register:
    def __init__(
        self,
        config: Optional[dict],
        lib: Optional[any] = None,
        debug: Optional[bool] = True,
        unused_ops_list: Optional[list] = [],
        default_vendor=None,
    ):
        self.device = device.Device()
        self.vendor_unused_ops = []
        self.vendor_extend_config = {}
        if default_vendor is not None:
            self.__user_init(default_vendor)
        else:
            self.__default_init()
        self.forward_ops = []
        self.backend_ops = []
        self.config = config
        # if self.vendor != default_vendor:
        self.vendor_extend_config = self.get_backend_extend_op()
        # self.vendor_unused_ops = self.get_vendor_unused_op()
        self.DEBUG = debug
        self.vendor_list = backend.vendors_map.keys()
        self._check_backend()
        self.unused_ops = unused_ops_list
        self.dispatch_backend_key_prefix = "Autograd"
        self.for_each(config, lib)
        if debug:
            self._set_info(config)
            self._set_info(self.vendor_extend_config)

    def __user_init(self, default_vendor):
        self.vendor_name = default_vendor
        _, self.device_name, _, self.vendor = self.device.get_vendor()

    def __default_init(self):
        self.vendor_name, self.device_name, _, self.vendor = self.device.get_vendor()
        self.has_extend_op = False

    def _check_backend(self):
        is_support = self.vendor_name in self.vendor_list
        if is_support is False:
            error.ErrorHandler().backend_not_support(
                self.device_name, self.backend_list
            )

    def get_backend_extend_op(self):
        if self.vendor_name != "nvidia":
            return backend.scheduler.get_curent_device_extend_op(self.vendor_name)
        return {}

    def get_vendor_unused_op():
        pass

    def __pass_register_cond(self, key):
        if key in self.unused_ops:
            return False
        elif key in key in self.vendor_unused_ops:
            return False
        else:
            return True

    def registerImpl(self, lib, key, val, device):
        device_auto = self.dispatch_backend_key_prefix + device
        if key in self.vendor_extend_config:
            func, hasbacken = self.vendor_extend_config[key]
        else:
            func, hasbacken = val
        if hasbacken:
            lib.impl(key, func, device_auto)
        else:
            lib.impl(key, func, device)

    def for_each(self, config, lib):
        device = self.device_name.upper()
        try:
            for key, val in config.items():
                if self.__pass_register_cond(key):
                    self.registerImpl(lib, key, val, device)

        except Exception as e:
            error.PASS(e)
            error.ErrorHandler().register_error()

    def _set_info(self, config):
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
        return self.unused_ops

    def get_vendor_name(self) -> str:
        return self.vendor_name

    def get_current_device(self) -> str:
        return self.device_name

    def support_backend(self, fn) -> bool:
        return fn.__name__ in self.backend_ops
