from typing import Optional

from . import backend, device, error


class Register:
    def __init__(
        self,
        config: Optional[tuple[tuple]],
        lib: Optional[any] = None,
        debug: Optional[bool] = True,
        unused_ops_list: Optional[list[str]] = [],
        default_vendor=None,
    ):
        self.device = device.Device()
        self.lib = lib
        self.vendor_unused_ops = []
        self.vendor_extend_config = {}
        if default_vendor is not None:
            self.__user_init(default_vendor)
        else:
            self.__default_init()
        self.device_upper = self.device_name.upper()
        self.DEBUG = debug
        self.forward_ops = []
        self.backend_ops = []
        self.config = config
        # if self.vendor != default_vendor:

        self.vendor_extend_config = self.get_vendor_extend_op()
        self.vendor_unused_ops = self.get_vendor_unused_op()
        self.vendor_list = list(backend.vendors_map.keys())
        self.unused_ops = unused_ops_list
        self._check_backend()
        self.for_each(config)
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

    def get_vendor_extend_op(self):
        # for demo , real is self.vendor_name != "nvidia":
        if self.vendor_name == "nvidia":
            return backend.scheduler.get_curent_device_extend_op(self.vendor_name)
        return ()

    def get_vendor_unused_op(self):
        if self.vendor_name == "nvidia":
            # for demo , real is self.vendor_name != "nvidia":
            return backend.scheduler.get_curent_device_unused_op(self.vendor_name)
        return {}

    def __pass_register_cond(self, key):
        return key not in self.unused_ops and key not in self.vendor_unused_ops

    def registerImpl(self, key, fn, has_backward):
        device_auto = backend.AUTOGRAD + self.device_upper
        if key in self.vendor_extend_config:
            fn, has_backward = (
                self.vendor_extend_config[1],
                self.vendor_extend_config[2],
            )
        self.lib.impl(key, fn, device_auto if has_backward else self.device_upper)

    def for_each(self, config):
        try:
            for key, func, has_backward in config:
                if self.__pass_register_cond(key):
                    self.registerImpl(key, func, has_backward)

        except Exception as e:
            error.PASS(e)
            error.ErrorHandler().register_error()

    def _set_info(self, config):
        for _, fn, hasbackward in config:
            fn_name = fn.__name__
            self.backend_ops.append(
                fn_name
            ) if hasbackward else self.forward_ops.append(fn_name)

    def get_forward_ops(self) -> list[str]:
        return self.forward_ops if self.DEBUG else []

    def get_backend_ops(self) -> list[str]:
        return self.backend_ops if self.DEBUG else []

    def get_unused_ops(self) -> list[str]:
        return self.unused_ops

    def get_vendor_name(self) -> str:
        return self.vendor_name

    def get_current_device(self) -> str:
        return self.device_name

    def support_backend(self, fn) -> bool:
        return fn.__name__ in self.backend_ops
