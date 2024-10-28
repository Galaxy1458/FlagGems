import os
import subprocess

import triton

from . import backend, error


class Device:
    def __init__(self):
        self.device_name = None
        self.vendor_name = None
        self.vendor_list = backend.vendors_map.keys()

    def get_vendor(self, vendor_name=None) -> tuple:
        if vendor_name is not None:
            return backend.scheduler.get_vendor_info(vendor_name)
        vendor_from_evn = self._get_vendor_from_evn()
        if vendor_from_evn is not None:
            return backend.scheduler.get_vendor_info(vendor_from_evn)
        try:
            return self._get_vendor_from_lib()
        except Exception as e:
            error.PASS(e)
            return self._get_vendor_from_sys()

    def _get_vendor_from_evn(self):
        device_from_evn = os.environ.get("GEMS_VENDOR")
        return None if device_from_evn not in self.vendor_list else device_from_evn

    def _get_vendor_from_sys(self):
        vendor_infos = backend.scheduler.get_vendor_infos()
        for info in vendor_infos:
            _, _, cmd, _ = info
            result = subprocess.run([cmd], capture_output=True, text=True)
            if result.returncode == 0:
                return info
        error.ErrorHandler().device_not_found()

    def _get_vendor_from_lib(self):
        return triton.get_vendor_info()

    @staticmethod
    def device_guard():
        return backend.scheduler.get_device_guard_fn()
