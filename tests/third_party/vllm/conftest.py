import os
import sys


_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


if os.environ.get("ROLL_NPU_CI") == "1":
    collect_ignore = [
        "test_add_requests.py",
        "test_collective_rpc.py",
        "test_fp8.py",
        "test_fp8_perf.py",
        "test_sleep_level.py",
        "test_vllm_local_actor.py",
        "test_vllm_local_async.py",
        "test_vllm_mem_oom.py",
        "vllm_generate_test.py", # npu server has no MATH_train_reformat_241225.json
    ]
