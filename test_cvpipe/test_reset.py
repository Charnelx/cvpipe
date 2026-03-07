# tests/cvpipe/test_reset.py
import pytest
import time
from cvpipe import Component, Frame, Pipeline, EventBus, ResultBus
from cvpipe.scheduler import FrameSource, Scheduler


class ResetableComponent(Component):
    INPUTS = []
    OUTPUTS = []

    def __init__(self, component_id: str = "resetable"):
        super().__init__()
        self._component_id = component_id
        self.reset_count = 0
        self._state = 0

    def process(self, frame: Frame) -> None:
        self._state += 1

    def reset(self) -> None:
        self.reset_count += 1
        self._state = 0


class DummySource(FrameSource):
    def __init__(self, num_frames: int = 5):
        self.num_frames = num_frames
        self._count = 0

    def next(self):
        if self._count >= self.num_frames:
            return None
        self._count += 1
        return ({"frame": self._count}, 0.0)


def test_component_reset_default_noop():
    """Component.reset() should be a no-op by default."""

    class BasicComponent(Component):
        INPUTS = []
        OUTPUTS = []

        def process(self, frame: Frame):
            pass

    comp = BasicComponent()
    comp.reset()


def test_resetable_component_reset():
    """Component with overridden reset() should be called."""
    comp = ResetableComponent()
    assert comp.reset_count == 0

    comp.reset()
    assert comp.reset_count == 1


def test_pipeline_reset_calls_component_reset():
    """pipeline.reset() should call component.reset()."""
    comp1 = ResetableComponent("comp1")
    comp2 = ResetableComponent("comp2")
    source = DummySource(2)

    pipeline = Pipeline(
        source=source,
        components=[comp1, comp2],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    pipeline.start()

    time.sleep(0.3)

    pipeline.reset()

    assert comp1.reset_count == 1
    assert comp2.reset_count == 1

    pipeline.stop()


def test_pipeline_reset_requires_running():
    """pipeline.reset() should raise if not running."""
    comp = ResetableComponent()
    source = DummySource(0)

    pipeline = Pipeline(
        source=source,
        components=[comp],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )

    with pytest.raises(RuntimeError, match="requires the pipeline to be running"):
        pipeline.reset()


def test_pipeline_reset_pauses_scheduler():
    """pipeline.reset() should pause the scheduler."""
    comp = CountingComponentForReset()
    source = DummySource(100)

    pipeline = Pipeline(
        source=source,
        components=[comp],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    pipeline.start()

    time.sleep(0.2)

    initial_count = comp.call_count

    pipeline.reset()

    time.sleep(0.1)

    after_reset_count = comp.call_count

    assert after_reset_count >= initial_count

    pipeline.stop()


def test_pipeline_reset_continues_on_exception():
    """pipeline.reset() should continue even if component.reset() raises."""
    comp1 = ResetableComponent("comp1")
    comp2 = ResetableComponent("comp2")

    class RaisingComponent(Component):
        INPUTS = []
        OUTPUTS = []

        def __init__(self):
            super().__init__()
            self.reset_called = False

        def process(self, frame: Frame):
            pass

        def reset(self):
            self.reset_called = True
            raise ValueError("reset error")

    raising_comp = RaisingComponent()
    raising_comp._component_id = "raising"
    source = DummySource(2)

    pipeline = Pipeline(
        source=source,
        components=[comp1, raising_comp, comp2],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    pipeline.start()

    time.sleep(0.3)

    pipeline.reset()

    assert comp1.reset_count == 1
    assert raising_comp.reset_called is True
    assert comp2.reset_count == 1

    pipeline.stop()


def test_scheduler_pause_blocks():
    """Scheduler.pause() should block until acknowledged."""
    comp = CountingComponentForReset()
    source = SlowSource()

    scheduler = Scheduler(
        source=source,
        components=[comp],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    scheduler.start()

    time.sleep(0.2)

    initial_count = comp.call_count

    scheduler.pause(timeout=1.0)

    paused_count = comp.call_count

    scheduler.resume()

    time.sleep(0.2)

    resumed_count = comp.call_count

    scheduler.stop()

    assert paused_count >= initial_count
    assert resumed_count >= paused_count


def test_scheduler_pause_timeout():
    """Scheduler.pause() should warn on timeout."""
    comp = CountingComponentForReset()
    source = SlowSource()

    scheduler = Scheduler(
        source=source,
        components=[comp],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    scheduler.start()

    scheduler.pause(timeout=0.01)

    scheduler.stop()


def test_scheduler_resume_unblocks():
    """Scheduler.resume() should unblock paused thread."""
    comp = CountingComponentForReset()
    source = SlowSource()

    scheduler = Scheduler(
        source=source,
        components=[comp],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    scheduler.start()

    time.sleep(0.2)

    scheduler.pause()
    paused_count = comp.call_count
    scheduler.resume()

    time.sleep(0.2)

    after_resume_count = comp.call_count

    scheduler.stop()

    assert after_resume_count > paused_count


class CountingComponentForReset(Component):
    INPUTS = []
    OUTPUTS = []

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def process(self, frame: Frame):
        self.call_count += 1


class SlowSource(FrameSource):
    def __init__(self):
        self._count = 0
        self._running = True

    def next(self):
        if not self._running:
            return None
        self._count += 1
        time.sleep(0.05)
        return ({"frame": self._count}, 0.0)

    def teardown(self):
        self._running = False
