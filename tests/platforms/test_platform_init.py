from types import SimpleNamespace

from roll.platforms import _init_platform


def test_init_platform_falls_back_to_cpu_when_no_visible_cuda_devices(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 0,
        )
    )

    monkeypatch.setattr("roll.platforms.torch", fake_torch)

    platform = _init_platform()

    assert platform.device_name == "CPU"


def test_init_platform_uses_corex_platform_for_iluvatar_device(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            get_device_name=lambda index=0: "Iluvatar BI-V150",
        )
    )

    monkeypatch.setattr("roll.platforms.torch", fake_torch)

    platform = _init_platform()

    assert platform.device_name == "COREX"


def test_init_platform_uses_unknown_platform_for_unrecognized_cuda_device(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            get_device_name=lambda index=0: "Some Vendor Accelerator",
        )
    )

    monkeypatch.setattr("roll.platforms.torch", fake_torch)

    platform = _init_platform()

    assert platform.device_name == "UNKNOWN"
