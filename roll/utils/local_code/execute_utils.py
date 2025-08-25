# This code is adapted from OpenAI's release
# https://github.com/openai/human-eval/blob/master/human_eval/execution.py

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
import sys

BASE_IMPORTS = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nfrom typing import List, Dict, Any, Optional, Tuple\nfrom math import inf\nsys.setrecursionlimit(6*10**5)\n"


def codeexecute_check_correctness(check_program, timeout=3):
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(check_program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")
    return result[0] == "passed"


def unsafe_execute(check_program, result, timeout):
    import os
    import shutil
    import tempfile
    import sys

    original_rmtree = shutil.rmtree
    original_rmdir = os.rmdir
    original_chdir = os.chdir
    original_unlink = os.unlink
    def execute_with_restrictions():
        temp_unlink = os.unlink
        reliability_guard()
        exec_globals = {}
        with swallow_io():
            with time_limit(timeout):
                exec(check_program, exec_globals)
        
        return "passed"
    
    try:
        with create_tempdir():
            try:
                result_value = execute_with_restrictions()
                result.append(result_value)
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")
            finally:
                shutil.rmtree = original_rmtree
                os.rmdir = original_rmdir
                os.chdir = original_chdir
                os.unlink = original_unlink
    except Exception as e:
        result.append(f"failed: Error in tempdir handling: {e}")
        shutil.rmtree = original_rmtree
        os.rmdir = original_rmdir
        os.chdir = original_chdir
        os.unlink = original_unlink


@contextlib.contextmanager
def time_limit(seconds):

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes=None):
    """This disables various destructive functions and prevents the generated
    code from interfering with the test (e.g. fork bomb, killing other
    processes, removing filesystem files, etc.)

    WARNING This function is NOT a security sandbox. Untrusted code, including,
    model- generated code, should not be blindly executed outside of one. See
    the Codex paper for more information about OpenAI's code sandbox, and
    proceed with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
