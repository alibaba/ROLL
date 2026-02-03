import asyncio
import functools
from typing import Callable, Coroutine, Any, Type

def run_sync(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Any]:
    """
    A decorator to run an async method synchronously.
    It gets or creates an event loop and runs the async method until it completes.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert asyncio.iscoroutinefunction(func)
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        assert not loop.is_closed() and not loop.is_running()
        coro = func(*args, **kwargs)
        return loop.run_until_complete(coro)
    return wrapper

def create_sync_class(OriginalCls: Type) -> Type:
    """
    A factory function that creates a synchronous subclass of a given class.
    
    It generates and returns a new class that inherits from the original
    but overrides all of its `async def` methods with synchronous wrappers.

    The name of the new class will be 'Sync' + original name.
    """
    new_class_attrs = {
        '__doc__': OriginalCls.__doc__
    }

    for name, method in OriginalCls.__dict__.items():
        if not name.startswith('_') and asyncio.iscoroutinefunction(method):
            new_class_attrs[name] = run_sync(method)

    SyncVersion = type(f"Sync{OriginalCls.__name__}", (OriginalCls,), new_class_attrs)
    return SyncVersion