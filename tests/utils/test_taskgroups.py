import asyncio
import pytest

from roll.utils.taskgroups import TaskGroup

async def test_base():
    async def foo(result, index):
        result[index] = 2333

    result = [None] * 4
    async with TaskGroup() as tg:
        for i in range(4):
            tg.create_task(foo(result, i))
    assert result == [2333, 2333, 2333, 2333]

async def test_cancel_parent():
    async def foo(result, index):
        result[index] = 2333

    async def tg_task(expected):
        result = [None] * 4
        try:
            async with TaskGroup() as tg:
                for i in range(4):
                    await asyncio.sleep(2)
                    tg.create_task(foo(result, i))
        except asyncio.CancelledError:
            assert result == expected
            raise

    task = asyncio.create_task(tg_task(expected=[None, None, None, None]))
    await asyncio.sleep(1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    task = asyncio.create_task(tg_task([2333, None, None, None]))
    await asyncio.sleep(3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

async def test_parent_exception():
    async def foo(result, index, sleep_time=0):
        await asyncio.sleep(sleep_time)
        result[index] = 2333

    async def tg_task():
        result = [None] * 4
        try:
            async with TaskGroup() as tg:
                tg.create_task(foo(result, 0, sleep_time=0))
                tg.create_task(foo(result, 1, sleep_time=0))
                tg.create_task(foo(result, 2, sleep_time=2))
                tg.create_task(foo(result, 3, sleep_time=2))
                await asyncio.sleep(1)
                raise RuntimeError
        except RuntimeError:
            assert result == [2333, 2333, None, None]
            raise

    with pytest.raises(RuntimeError):
        await asyncio.create_task(tg_task())

async def test_tg_exception():
    async def foo(result, index, sleep_time=0, raise_exception=False):
        await asyncio.sleep(sleep_time)
        if raise_exception:
            raise RuntimeError
        result[index] = 2333

    async def tg_task():
        result = [None] * 4
        try:
            async with TaskGroup() as tg:
                tg.create_task(foo(result, 0, sleep_time=0, raise_exception=False))
                tg.create_task(foo(result, 1, sleep_time=0, raise_exception=False))
                tg.create_task(foo(result, 2, sleep_time=0, raise_exception=True))
                tg.create_task(foo(result, 3, sleep_time=2, raise_exception=False))
                # dead loop to test whether TaskGroup can propragate exception
                while True:
                    await asyncio.sleep(1)
        except RuntimeError:
            assert result == [2333, 2333, None, None]
            raise

    with pytest.raises(RuntimeError):
        await asyncio.create_task(tg_task())

async def test_cancel_tg():
    async def foo(result, index, cancel=False):
        await asyncio.sleep(1)
        result[index] = 2333

    async def tg_task():
        result = [None] * 4
        async with TaskGroup() as tg:
            tg.create_task(foo(result, 0))
            tg.create_task(foo(result, 1))
            task = tg.create_task(foo(result, 2))
            task.cancel()
            tg.create_task(foo(result, 3))
        assert result == [2333, 2333, None, 2333]

    await asyncio.create_task(tg_task())

async def main():
    await asyncio.create_task(test_base())
    await asyncio.create_task(test_cancel_parent())
    await asyncio.create_task(test_parent_exception())
    await asyncio.create_task(test_tg_exception())
    await asyncio.create_task(test_cancel_tg())

if __name__ == "__main__":
    asyncio.run(main())
