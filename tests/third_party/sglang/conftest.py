import os


if os.environ.get("ROLL_NPU_CI") == "1":
    # Full engine-backed SGLang abort tests can exceed the NPU CI container
    # memory limit; keep them opt-in while retaining the lightweight abort smoke.
    collect_ignore = [
        "test_abort.py",
        "test_abort_grpc.py",
        "test_abort_http.py",
        "test_fp8.py",
    ]
    if os.environ.get("ROLL_NPU_SGLANG_ABORT_ENGINE") == "1":
        collect_ignore.remove("test_abort.py")
