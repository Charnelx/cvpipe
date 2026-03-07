# tests/cvpipe/test_bridge.py
import pytest
import asyncio
from cvpipe.bridge import AsyncQueueBridge


def test_constructor_valid_maxsize():
    """AsyncQueueBridge should construct with valid maxsize."""
    loop = asyncio.new_event_loop()
    bridge = AsyncQueueBridge(loop=loop, maxsize=8)
    assert bridge._maxsize == 8
    loop.close()


def test_constructor_maxsize_zero_raises():
    """AsyncQueueBridge should raise on maxsize < 1."""
    loop = asyncio.new_event_loop()
    with pytest.raises(ValueError, match="maxsize must be >= 1"):
        AsyncQueueBridge(loop=loop, maxsize=0)
    loop.close()


def test_constructor_negative_raises():
    """AsyncQueueBridge should raise on negative maxsize."""
    loop = asyncio.new_event_loop()
    with pytest.raises(ValueError, match="maxsize must be >= 1"):
        AsyncQueueBridge(loop=loop, maxsize=-1)
    loop.close()


def test_default_maxsize():
    """Default maxsize should be 8."""
    loop = asyncio.new_event_loop()
    bridge = AsyncQueueBridge(loop=loop)
    assert bridge._maxsize == 8
    loop.close()


@pytest.mark.asyncio
async def test_put_calls_handler():
    """put() should call handler in event loop."""
    loop = asyncio.get_event_loop()
    bridge = AsyncQueueBridge(loop=loop, maxsize=8)
    received = []

    async def handler(item):
        received.append(item)

    await bridge.start_consumer(handler)
    bridge.put("test_item")

    await asyncio.sleep(0.1)

    assert received == ["test_item"]
    await bridge.stop()


@pytest.mark.asyncio
async def test_drop_oldest_when_full():
    """put() should drop oldest when queue is full."""
    loop = asyncio.get_event_loop()
    bridge = AsyncQueueBridge(loop=loop, maxsize=2)
    received = []

    async def handler(item):
        received.append(item)

    await bridge.start_consumer(handler)

    bridge.put("item1")
    bridge.put("item2")
    bridge.put("item3")

    await asyncio.sleep(0.1)

    assert "item3" in received
    assert "item1" not in received
    await bridge.stop()


@pytest.mark.asyncio
async def test_handler_exception_continues():
    """Handler exception should not stop bridge."""
    loop = asyncio.get_event_loop()
    bridge = AsyncQueueBridge(loop=loop, maxsize=8)
    received = []

    async def handler(item):
        if item == "bad":
            raise ValueError("test error")
        received.append(item)

    await bridge.start_consumer(handler)

    bridge.put("good1")
    bridge.put("bad")
    bridge.put("good2")

    await asyncio.sleep(0.1)

    assert "good1" in received
    assert "good2" in received
    await bridge.stop()


@pytest.mark.asyncio
async def test_stop_before_start():
    """stop() before start_consumer should be no-op."""
    loop = asyncio.new_event_loop()
    bridge = AsyncQueueBridge(loop=loop, maxsize=8)

    await bridge.stop()
    loop.close()


@pytest.mark.asyncio
async def test_start_consumer_twice_raises():
    """start_consumer() twice should raise RuntimeError."""
    loop = asyncio.new_event_loop()
    bridge = AsyncQueueBridge(loop=loop, maxsize=8)

    async def handler(item):
        pass

    await bridge.start_consumer(handler)

    with pytest.raises(RuntimeError, match="already running"):
        await bridge.start_consumer(handler)

    await bridge.stop()
    loop.close()
