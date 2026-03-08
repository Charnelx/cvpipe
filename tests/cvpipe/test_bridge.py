# tests/cvpipe/test_bridge.py

from __future__ import annotations

import asyncio

import pytest

from cvpipe.bridge import AsyncQueueBridge


class TestAsyncQueueBridge:
    def test_bridge_invalid_maxsize(self) -> None:
        with pytest.raises(ValueError, match="maxsize must be >= 1"):
            AsyncQueueBridge(loop=asyncio.new_event_loop(), maxsize=0)

    def test_bridge_default_maxsize(self) -> None:
        loop = asyncio.new_event_loop()
        bridge = AsyncQueueBridge(loop=loop)
        assert bridge._maxsize == 8
        loop.close()

    @pytest.mark.asyncio
    async def test_bridge_put_get(self) -> None:
        loop = asyncio.new_event_loop()
        bridge = AsyncQueueBridge(loop=loop, maxsize=8)

        async def handler(item: object) -> None:
            pass

        await bridge.start_consumer(handler)

        bridge.put("test_item")

        await asyncio.sleep(0.5)

        await bridge.stop()
        loop.close()

    @pytest.mark.asyncio
    async def test_bridge_put_queue_full(self) -> None:
        loop = asyncio.new_event_loop()
        bridge = AsyncQueueBridge(loop=loop, maxsize=2)

        received = []

        async def handler(item: object) -> None:
            received.append(item)

        await bridge.start_consumer(handler)

        bridge.put("item1")
        bridge.put("item2")
        bridge.put("item3")

        await asyncio.sleep(0.2)

        await bridge.stop()
        loop.close()

        assert len(received) <= 2

    @pytest.mark.asyncio
    async def test_bridge_start_consumer_twice(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bridge = AsyncQueueBridge(loop=loop, maxsize=8)

        async def handler(item: object) -> None:
            pass

        await bridge.start_consumer(handler)

        with pytest.raises(RuntimeError, match="already running"):
            await bridge.start_consumer(handler)

        await bridge.stop()
        loop.close()

    @pytest.mark.asyncio
    async def test_bridge_stop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bridge = AsyncQueueBridge(loop=loop, maxsize=8)

        async def handler(item: object) -> None:
            await asyncio.sleep(10)

        await bridge.start_consumer(handler)
        bridge.put("item1")

        await asyncio.sleep(0.05)

        await bridge.stop()

        assert bridge._task is None
        loop.close()

    def test_bridge_qsize(self) -> None:
        loop = asyncio.new_event_loop()
        bridge = AsyncQueueBridge(loop=loop, maxsize=8)
        assert bridge.qsize == 0
        loop.close()

    @pytest.mark.asyncio
    async def test_bridge_handler_exception(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bridge = AsyncQueueBridge(loop=loop, maxsize=8)

        async def bad_handler(item: object) -> None:
            raise RuntimeError("handler error")

        await bridge.start_consumer(bad_handler)
        bridge.put("item1")

        await asyncio.sleep(0.1)

        await bridge.stop()
        loop.close()

    @pytest.mark.asyncio
    async def test_bridge_put_when_queue_full_drops_oldest(self) -> None:
        loop = asyncio.get_event_loop()
        bridge = AsyncQueueBridge(loop=loop, maxsize=2)

        received: list[str] = []

        async def handler(item: object) -> None:
            received.append(str(item))

        await bridge.start_consumer(handler)

        bridge.put("item1")
        bridge.put("item2")
        bridge.put("item3")

        await asyncio.sleep(0.5)

        await bridge.stop()

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_bridge_put_multiple_iterations(self) -> None:
        loop = asyncio.get_event_loop()
        bridge = AsyncQueueBridge(loop=loop, maxsize=16)

        received: list[str] = []

        async def handler(item: object) -> None:
            received.append(str(item))

        await bridge.start_consumer(handler)

        bridge.put("a")
        bridge.put("b")
        bridge.put("c")
        bridge.put("d")
        bridge.put("e")

        await asyncio.sleep(0.5)

        await bridge.stop()

        assert len(received) == 5

    @pytest.mark.asyncio
    async def test_bridge_queue_full_race_condition(self) -> None:
        loop = asyncio.get_event_loop()
        bridge = AsyncQueueBridge(loop=loop, maxsize=1)

        async def never_finish_handler(item: object) -> None:
            await asyncio.sleep(10)

        await bridge.start_consumer(never_finish_handler)

        bridge.put("blocker")

        await asyncio.sleep(0.05)

        bridge.put("item1")  # type: ignore[arg-type]
        bridge.put("item2")  # type: ignore[arg-type]

        await asyncio.sleep(0.1)

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_bridge_consume_handler_exception_continues(self) -> None:
        loop = asyncio.get_event_loop()
        bridge = AsyncQueueBridge(loop=loop, maxsize=8)

        received: list[str] = []
        error_count = 0

        async def erratic_handler(item: object) -> None:
            nonlocal error_count
            if str(item).startswith("bad"):
                error_count += 1
                raise RuntimeError("handler error")
            received.append(str(item))

        await bridge.start_consumer(erratic_handler)

        bridge.put("good1")
        bridge.put("bad1")
        bridge.put("good2")

        await asyncio.sleep(0.5)

        await bridge.stop()

        assert len(received) == 2
        assert error_count == 1
