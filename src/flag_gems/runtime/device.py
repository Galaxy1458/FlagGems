import os
import subprocess

import triton

from . import backend, error


class Device:
    def __init__(self):
        self.device_name = None
        self.vendor_name = None
        self.vendor_list = backend.vendor_list

    def get_vendor(self) -> tuple:
        device_from_evn = self._get_vendor_from_evn()
        if device_from_evn is not None:
            return backend.scheduler.get_vendor_info(device_from_evn)
        try:
            return self._get_vendor_from_lib()
        except Exception as e:
            e
            return self._get_vendor_from_sys()

    def _get_vendor_from_evn(self):
        device_from_evn = os.environ.get("GEMS_VENDOR")
        device_from_evn = "nvidia"
        return None if device_from_evn not in self.vendor_list else device_from_evn

    def _get_vendor_from_sys(self):
        vendor_infos = backend.scheduler.get_vendor_infos()

        for info in vendor_infos:
            vendor_name, device_name, cmd = info
            result = subprocess.run([cmd], capture_output=True, text=True)
            if result.returncode == 0:
                return info
        error.ErrorHandler().device_not_found()

    def _get_vendor_from_lib(self):
        return triton.get_vendor_info()

    def device_guard(self):
        return backend.scheduler.get_device_guard_fn()
