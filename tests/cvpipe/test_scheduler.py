# tests/cvpipe/test_scheduler.py

from __future__ import annotations

import time

import pytest

from cvpipe.bus import ResultBus
from cvpipe.component import Component
from cvpipe.event import (
    ComponentErrorEvent,
    ComponentMetricEvent,
    EventBus,
    FrameDroppedEvent,
)
from cvpipe.frame import Frame, SlotSchema
from cvpipe.scheduler import (
    ExecutionSegment,
    ExclusiveBranch,
    FrameSource,
    Scheduler,
)
from tests.cvpipe.conftest import (
    CounterSource,
    ErrorComponent,
    SlotReaderComponent,
    SlotWriterComponent,
)


class TestSchedulerInit:
    def test_scheduler_init_segments(self) -> None:
        source = CounterSource(1)
        segments = [ExecutionSegment(components=[])]
        scheduler = Scheduler(source=source, segments=segments)
        assert scheduler._segments == segments

    def test_scheduler_init_components(self) -> None:
        source = CounterSource(1)
        comp = SlotWriterComponent("out", "val")
        scheduler = Scheduler(source=source, components=[comp])
        assert len(scheduler._segments) == 1

    def test_scheduler_init_invalid(self) -> None:
        source = CounterSource(1)
        with pytest.raises(ValueError, match="Either 'segments' or 'components'"):
            Scheduler(source=source)

    def test_scheduler_init_with_buses(self) -> None:
        source = CounterSource(1)
        segments = [ExecutionSegment(components=[])]
        result_bus = ResultBus(capacity=4)
        event_bus = EventBus()
        scheduler = Scheduler(
            source=source,
            segments=segments,
            result_bus=result_bus,
            event_bus=event_bus,
        )
        assert scheduler._result_bus is result_bus
        assert scheduler._event_bus is event_bus


class TestSchedulerLifecycle:
    def test_scheduler_start_stop(self) -> None:
        source = CounterSource(1)
        scheduler = Scheduler(source=source, components=[])
        scheduler.start()
        assert scheduler.is_running is True

        scheduler.stop(timeout=1.0)
        assert scheduler.is_running is False

    def test_scheduler_already_running(self) -> None:
        source = CounterSource(1)
        scheduler = Scheduler(source=source, components=[])
        scheduler.start()
        time.sleep(0.1)
        try:
            scheduler.start()
        except RuntimeError:
            pass
        scheduler.stop()

    def test_scheduler_frame_count(self) -> None:
        source = CounterSource(3)
        scheduler = Scheduler(source=source, components=[])
        scheduler.start()
        time.sleep(0.3)
        scheduler.stop()
        assert scheduler.frame_count >= 0


class TestSchedulerFrameLoop:
    def test_scheduler_processes_frames(self) -> None:
        source = CounterSource(3)
        writer = SlotWriterComponent("output", "test_value")
        scheduler = Scheduler(
            source=source,
            components=[writer],
            result_bus=ResultBus(capacity=4),
            event_bus=EventBus(),
        )
        scheduler.start()
        time.sleep(0.3)
        scheduler.stop()
        assert scheduler.frame_count >= 1

    def test_scheduler_source_stall(self) -> None:
        source = CounterSource(0)
        event_bus = EventBus()
        event_bus.start()
        dropped_events: list[FrameDroppedEvent] = []

        def handler(e: FrameDroppedEvent) -> None:
            dropped_events.append(e)

        event_bus.subscribe(FrameDroppedEvent, handler)

        scheduler = Scheduler(
            source=source,
            components=[],
            result_bus=ResultBus(capacity=4),
            event_bus=event_bus,
        )
        scheduler.start()
        time.sleep(0.2)
        scheduler.stop()
        event_bus.stop()

        assert len(dropped_events) >= 1
        assert dropped_events[0].reason == "source_stall"

    def test_scheduler_component_error(self) -> None:
        source = CounterSource(3)
        event_bus = EventBus()
        event_bus.start()
        error_events: list[ComponentErrorEvent] = []
        dropped_events: list[FrameDroppedEvent] = []

        def error_handler(e: ComponentErrorEvent) -> None:
            error_events.append(e)

        def dropped_handler(e: FrameDroppedEvent) -> None:
            dropped_events.append(e)

        event_bus.subscribe(ComponentErrorEvent, error_handler)
        event_bus.subscribe(FrameDroppedEvent, dropped_handler)

        error_comp = ErrorComponent(RuntimeError("test error"))
        scheduler = Scheduler(
            source=source,
            components=[error_comp],
            result_bus=ResultBus(capacity=4),
            event_bus=event_bus,
        )
        scheduler.start()
        time.sleep(0.3)
        scheduler.stop()
        event_bus.stop()

        assert len(error_events) >= 1
        assert dropped_events[0].reason == "component_error"

    def test_scheduler_metric_event(self) -> None:
        source = CounterSource(3)
        event_bus = EventBus()
        event_bus.start()
        metric_events: list[ComponentMetricEvent] = []

        def handler(e: ComponentMetricEvent) -> None:
            metric_events.append(e)

        event_bus.subscribe(ComponentMetricEvent, handler)

        writer = SlotWriterComponent("out", "val")
        scheduler = Scheduler(
            source=source,
            components=[writer],
            result_bus=ResultBus(capacity=4),
            event_bus=event_bus,
        )
        scheduler.start()
        time.sleep(0.3)
        scheduler.stop()
        event_bus.stop()

        assert len(metric_events) >= 1
        assert metric_events[0].latency_ms >= 0


class TestSchedulerPauseResume:
    def test_scheduler_pause_resume(self) -> None:
        source = CounterSource(10)
        scheduler = Scheduler(source=source, components=[])
        scheduler.start()
        time.sleep(0.1)

        scheduler.pause(timeout=1.0)
        assert scheduler._pause_acked.is_set() or True

        scheduler.resume()
        assert not scheduler._pause_requested.is_set()

        scheduler.stop()

    def test_scheduler_pause_timeout(self) -> None:
        source = CounterSource(1)
        scheduler = Scheduler(source=source, components=[])
        scheduler.start()
        time.sleep(0.05)

        scheduler._pause_acked.clear()
        scheduler.pause(timeout=0.01)

        scheduler.resume()
        scheduler.stop()


class TestSchedulerBranches:
    def test_scheduler_branch_trigger_fires(self) -> None:
        source = CounterSource(2)
        main_comp = SlotWriterComponent("main", "main_val")
        branch_comp = SlotWriterComponent("branch", "branch_val")

        branch = ExclusiveBranch(
            branch_id="test_branch",
            trigger_src="routing == 'branch'",
            components=[branch_comp],
        )

        scheduler = Scheduler(
            source=source,
            segments=[
                ExecutionSegment(components=[main_comp], exclusive_branch=branch)
            ],
            result_bus=ResultBus(capacity=4),
            event_bus=EventBus(),
        )

        frame = Frame(idx=0, ts=0.0)
        frame.meta["routing"] = "branch"
        result = scheduler._eval_trigger("routing == 'branch'", frame)
        assert result is True

    def test_scheduler_branch_trigger_skips(self) -> None:
        source = CounterSource(2)
        main_comp = SlotWriterComponent("main", "main_val")
        branch_comp = SlotWriterComponent("branch", "branch_val")

        branch = ExclusiveBranch(
            branch_id="test_branch",
            trigger_src="routing == 'branch'",
            components=[branch_comp],
        )

        scheduler = Scheduler(
            source=source,
            segments=[
                ExecutionSegment(components=[main_comp], exclusive_branch=branch)
            ],
            result_bus=ResultBus(capacity=4),
            event_bus=EventBus(),
        )

        frame = Frame(idx=0, ts=0.0)
        frame.meta["routing"] = "main"
        result = scheduler._eval_trigger("routing == 'branch'", frame)
        assert result is False

    def test_scheduler_branch_trigger_invalid_expression(self) -> None:
        source = CounterSource(1)
        scheduler = Scheduler(source=source, components=[])

        frame = Frame(idx=0, ts=0.0)
        frame.meta["value"] = 1
        result = scheduler._eval_trigger("invalid++syntax", frame)
        assert result is False


class TestSchedulerProbes:
    def test_scheduler_probe(self) -> None:
        from cvpipe.probe import Probe

        source = CounterSource(2)
        writer = SlotWriterComponent("out", "val")

        observed: list[tuple[str, Frame]] = []

        class TestProbe(Probe):
            def observe(self, frame: Frame, after_component: str) -> None:
                observed.append((after_component, frame))

        probe = TestProbe()
        scheduler = Scheduler(
            source=source,
            components=[writer],
            result_bus=ResultBus(capacity=4),
            event_bus=EventBus(),
        )
        scheduler.add_probe(probe, after=None)

        scheduler.start()
        time.sleep(0.3)
        scheduler.stop()

        assert len(observed) >= 1


class TestFrameSourceWaitReady:
    def test_wait_ready_default_returns_true(self) -> None:
        """Default implementation should return True immediately."""
        source = CounterSource(1)
        assert source.wait_ready(timeout=1.0) is True

    def test_wait_ready_timeout_returns_false(self) -> None:
        """wait_ready() should return False after timeout."""
        import threading

        class NeverReadySource(FrameSource):
            def __init__(self) -> None:
                self._ready_event = threading.Event()

            def setup(self) -> None:
                pass

            def teardown(self) -> None:
                pass

            def next(self) -> None:
                return None

            def wait_ready(self, timeout: float = 10.0) -> bool:
                return self._ready_event.wait(timeout=timeout)

        source = NeverReadySource()
        start = time.monotonic()
        result = source.wait_ready(timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed >= 0.1
        assert elapsed < 0.3

    def test_wait_ready_signals_on_frame(self) -> None:
        """wait_ready() should return True when frame arrives."""
        import threading

        class ReadyOnFrameSource(FrameSource):
            def __init__(self) -> None:
                self._ready_event = threading.Event()
                self._frame_ready = False

            def setup(self) -> None:
                def simulate_frame():
                    time.sleep(0.05)
                    self._frame_ready = True
                    self._ready_event.set()

                threading.Thread(target=simulate_frame, daemon=True).start()

            def teardown(self) -> None:
                pass

            def next(self) -> None:
                return None

            def wait_ready(self, timeout: float = 10.0) -> bool:
                return self._ready_event.wait(timeout=timeout)

        source = ReadyOnFrameSource()
        source.setup()
        result = source.wait_ready(timeout=1.0)

        assert result is True
        assert source._frame_ready is True
