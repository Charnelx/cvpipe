# cvpipe/bridge.py

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class AsyncQueueBridge:
    """
    Thread-safe bridge from a non-async producer thread to an asyncio event loop.

    Designed for use with ResultBus: subscribe bridge.put to a ResultBus, then
    start an async consumer to process results in the event loop.

    The bridge is lossy: if the internal queue is full when put() is called, the
    oldest item is discarded. This matches ResultBus's own drop-oldest semantics
    and prevents unbounded memory growth if the consumer is slow.

    Usage::

        # At application startup (in async context):
        loop = asyncio.get_event_loop()
        bridge = AsyncQueueBridge(loop=loop, maxsize=8)
        result_bus.subscribe(bridge.put)
        result_bus.start()
        await bridge.start_consumer(my_async_handler)

        # my_async_handler receives one item at a time:
        async def my_async_handler(result: FrameResult) -> None:
            await websocket.send_bytes(result.jpeg_bytes)

        # At shutdown:
        await bridge.stop()
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, maxsize: int = 8) -> None:
        """
        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The event loop in which the consumer coroutine will run.
            Must be the same loop in which start_consumer() is called.
        maxsize : int
            Maximum queue depth before oldest items are dropped.
            Default 8 provides ~267ms of buffering at 30fps.
        """
        if maxsize < 1:
            raise ValueError(f"AsyncQueueBridge maxsize must be >= 1, got {maxsize}")
        self._loop = loop
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._task: asyncio.Task | None = None
        self._maxsize = maxsize

    def put(self, item: Any) -> None:
        """
        Enqueue an item from any thread. Non-blocking.

        If the queue is full, the oldest item is discarded to make room.
        This method is safe to call from a non-async thread (e.g., a
        ResultBus subscriber thread).

        Parameters
        ----------
        item : Any
            The item to enqueue (typically a FrameResult).
        """

        def _put_in_loop() -> None:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    logger.debug("[AsyncQueueBridge] Queue full — dropped oldest item")
                except asyncio.QueueEmpty:
                    pass
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                logger.debug("[AsyncQueueBridge] Queue full race — item discarded")

        self._loop.call_soon_threadsafe(_put_in_loop)

    async def start_consumer(
        self,
        handler: Callable[[Any], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Start the async consumer coroutine.

        Must be called from within the event loop (i.e., from an async function
        or FastAPI startup handler). Call only once.

        Parameters
        ----------
        handler : async callable
            Called with each dequeued item. Must be an async function.
            Exceptions are caught and logged; the consumer continues.
        """
        if self._task is not None:
            raise RuntimeError("AsyncQueueBridge consumer is already running")
        self._task = asyncio.create_task(
            self._consume(handler),
            name="AsyncQueueBridge-consumer",
        )

    async def stop(self) -> None:
        """
        Cancel the consumer task and wait for it to finish.

        Safe to call even if start_consumer() was never called.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    @property
    def qsize(self) -> int:
        """Current number of items in the queue. For diagnostics."""
        return self._queue.qsize()

    async def _consume(self, handler: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        while True:
            item = await self._queue.get()
            try:
                await handler(item)
            except Exception:
                logger.exception("[AsyncQueueBridge] Handler %r raised — continuing", handler)
            finally:
                self._queue.task_done()
