import ctypes

import roll.platforms.platform as platform_module
from roll.platforms.unknown import UnknownPlatform


def test_device_memory_used_falls_back_to_mem_get_info(monkeypatch):
    def fake_device_memory_used(device=None):
        raise RuntimeError("nvml unavailable")

    def fake_mem_get_info(device=None):
        return (3, 10)

    monkeypatch.setattr("torch.cuda.device_memory_used", fake_device_memory_used)
    monkeypatch.setattr("torch.cuda.mem_get_info", fake_mem_get_info)
    monkeypatch.setattr(
        UnknownPlatform,
        "_nvml_compatible_device_memory_used",
        classmethod(lambda cls, device, primary_exc: None),
    )

    assert UnknownPlatform.device_memory_used() == 7


def test_device_memory_used_falls_back_to_memory_reserved(monkeypatch):
    def fake_device_memory_used(device=None):
        raise RuntimeError("nvml unavailable")

    def fake_mem_get_info(device=None):
        raise RuntimeError("mem_get_info unavailable")

    monkeypatch.setattr("torch.cuda.device_memory_used", fake_device_memory_used)
    monkeypatch.setattr("torch.cuda.mem_get_info", fake_mem_get_info)
    monkeypatch.setattr("torch.cuda.memory_reserved", lambda device=None: 123)
    monkeypatch.setattr(
        UnknownPlatform,
        "_nvml_compatible_device_memory_used",
        classmethod(lambda cls, device, primary_exc: None),
    )

    assert UnknownPlatform.device_memory_used() == 123


def test_map_visible_device_index(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")

    assert platform_module._map_visible_device_index(0, "CUDA_VISIBLE_DEVICES") == 4
    assert platform_module._map_visible_device_index(1, "CUDA_VISIBLE_DEVICES") == 7


def test_device_memory_used_uses_nvml_compatible_library(monkeypatch):
    class FakeFunction:
        def __init__(self, fn):
            self.fn = fn
            self.argtypes = None
            self.restype = None

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    class FakeLibrary:
        def __init__(self):
            self.handle_index = None
            self.nvmlInit_v2 = FakeFunction(lambda: 0)
            self.nvmlErrorString = FakeFunction(lambda retcode: b"ok")
            self.nvmlDeviceGetHandleByIndex_v2 = FakeFunction(self._get_handle)
            self.nvmlDeviceGetMemoryInfo = FakeFunction(self._get_memory_info)

        def _get_handle(self, index, handle_ptr):
            self.handle_index = index
            ctypes.cast(handle_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value = 1234
            return 0

        def _get_memory_info(self, handle, memory_info_ptr):
            memory_info = ctypes.cast(
                memory_info_ptr, ctypes.POINTER(platform_module._NvmlMemoryInfo)
            ).contents
            memory_info.total = 100
            memory_info.free = 25
            memory_info.used = 75
            return 0

    fake_library = FakeLibrary()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    monkeypatch.setattr(
        platform_module,
        "_load_nvml_compatible_library",
        lambda: (fake_library, "libixml.so"),
    )

    used, library_name = platform_module._nvml_compatible_device_memory_used(0, "CUDA_VISIBLE_DEVICES")
    assert used == 75
    assert library_name == "libixml.so"
    assert fake_library.handle_index == 5
