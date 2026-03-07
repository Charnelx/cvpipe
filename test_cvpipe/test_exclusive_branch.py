# tests/cvpipe/test_exclusive_branch.py
from cvpipe import Component, Frame, EventBus, ResultBus
from cvpipe.scheduler import FrameSource, Scheduler, ExecutionSegment, ExclusiveBranch
from cvpipe.config import BranchSpec


class PassthroughComponent(Component):
    INPUTS = []
    OUTPUTS = []

    def __init__(self, component_id: str = "passthrough"):
        super().__init__()
        self._component_id = component_id

    def process(self, frame: Frame) -> None:
        pass


class CountingComponent(Component):
    INPUTS = []
    OUTPUTS = []

    def __init__(self, component_id: str = "counting"):
        super().__init__()
        self._component_id = component_id
        self.call_count = 0

    def process(self, frame: Frame) -> None:
        self.call_count += 1


class DummySource(FrameSource):
    def __init__(self, num_frames: int = 3):
        self.num_frames = num_frames
        self._count = 0

    def next(self):
        if self._count >= self.num_frames:
            return None
        self._count += 1
        return ({"frame": self._count}, 0.0)


class DummySourceWithTrigger(FrameSource):
    def __init__(self, num_frames: int = 3, skip_main: bool = False):
        self.num_frames = num_frames
        self._count = 0
        self.skip_main = skip_main

    def next(self):
        if self._count >= self.num_frames:
            return None
        self._count += 1
        return ({"frame": self._count, "skip_main": self.skip_main}, 0.0)


def test_scheduler_accepts_segments():
    """Scheduler should accept segments parameter."""
    comp = PassthroughComponent()
    seg = ExecutionSegment(components=[comp])
    source = DummySource(1)

    scheduler = Scheduler(
        source=source,
        segments=[seg],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    assert scheduler.is_running is False


def test_scheduler_backward_compat_components():
    """Scheduler should accept components parameter for backward compatibility."""
    comp = PassthroughComponent()
    source = DummySource(1)

    scheduler = Scheduler(
        source=source,
        components=[comp],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    assert scheduler.is_running is False


def test_scheduler_segments_run():
    """Components in segments should run."""
    comp = CountingComponent()
    seg = ExecutionSegment(components=[comp])
    source = DummySource(2)

    scheduler = Scheduler(
        source=source,
        segments=[seg],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    scheduler.start()
    import time

    time.sleep(0.2)
    scheduler.stop()

    assert comp.call_count == 2


def test_exclusive_branch_trigger_true():
    """Exclusive branch should run when trigger is true."""
    main_comp = CountingComponent("main")
    branch_comp = CountingComponent("branch")
    seg = ExecutionSegment(
        components=[main_comp],
        exclusive_branch=ExclusiveBranch(
            branch_id="test_branch",
            trigger_src="skip_main == True",
            components=[branch_comp],
        ),
    )
    source = DummySourceWithTrigger(2, skip_main=True)

    scheduler = Scheduler(
        source=source,
        segments=[seg],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )
    scheduler.start()
    import time

    time.sleep(0.2)
    scheduler.stop()

    assert main_comp.call_count == 0
    assert branch_comp.call_count == 2


def test_exclusive_branch_trigger_false():
    """Main path should run when trigger is false."""
    main_comp = CountingComponent("main")
    branch_comp = CountingComponent("branch")
    seg = ExecutionSegment(
        components=[main_comp],
        exclusive_branch=ExclusiveBranch(
            branch_id="test_branch",
            trigger_src="skip_main == True",
            components=[branch_comp],
        ),
    )
    source = DummySourceWithTrigger(2, skip_main=False)

    scheduler = Scheduler(
        source=source,
        segments=[seg],
        result_bus=ResultBus(),
        event_bus=EventBus(),
    )

    scheduler.start()
    import time

    time.sleep(0.2)
    scheduler.stop()

    assert main_comp.call_count == 2
    assert branch_comp.call_count == 0


def test_eval_trigger_with_meta():
    """Trigger should evaluate against frame.meta."""
    from cvpipe.scheduler import Scheduler

    class TestScheduler(Scheduler):
        def __init__(self):
            pass

    scheduler = TestScheduler()
    frame = Frame(idx=0, ts=0.0)
    frame.meta["skip_main"] = True

    result = scheduler._eval_trigger("skip_main == True", frame)
    assert result is True


def test_eval_trigger_missing_key():
    """Trigger with missing key should return False."""
    from cvpipe.scheduler import Scheduler

    class TestScheduler(Scheduler):
        def __init__(self):
            pass

    scheduler = TestScheduler()
    frame = Frame(idx=0, ts=0.0)

    result = scheduler._eval_trigger("skip_main == True", frame)
    assert result is False


def test_eval_trigger_exception():
    """Trigger with exception should return False and log debug."""
    from cvpipe.scheduler import Scheduler

    class TestScheduler(Scheduler):
        def __init__(self):
            pass

    scheduler = TestScheduler()
    frame = Frame(idx=0, ts=0.0)

    result = scheduler._eval_trigger("invalid syntax {{", frame)

    assert result is False


def test_branchspec_exclusive_default_false():
    """BranchSpec should default exclusive to False."""
    spec = BranchSpec(
        id="test",
        components=[],
        trigger="x == True",
        inject_after="a",
        merge_before="b",
    )
    assert spec.exclusive is False


def test_branchspec_exclusive_true():
    """BranchSpec should accept exclusive=True."""
    spec = BranchSpec(
        id="test",
        components=[],
        trigger="x == True",
        inject_after="a",
        merge_before="b",
        exclusive=True,
    )
    assert spec.exclusive is True


def test_config_parse_exclusive():
    """PipelineConfig should parse exclusive field."""
    from cvpipe.config import PipelineConfig
    import tempfile
    import os

    yaml_content = """
pipeline:
  source: test_source
  components:
    - module: test
      id: a
    - module: test
      id: b
    - module: test
      id: c
  branches:
    - id: fast_path
      trigger: "use_fast == True"
      inject_after: a
      merge_before: c
      exclusive: true
      components:
        - module: test
          id: fast
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = PipelineConfig.from_yaml(temp_path)
        assert len(config.branches) == 1
        assert config.branches[0].exclusive is True
    finally:
        os.unlink(temp_path)


def test_config_parse_exclusive_default():
    """PipelineConfig should default exclusive to False."""
    from cvpipe.config import PipelineConfig
    import tempfile
    import os

    yaml_content = """
pipeline:
  source: test_source
  components:
    - module: test
      id: a
  branches:
    - id: normal_branch
      trigger: "x == True"
      inject_after: a
      merge_before: a
      components: []
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = PipelineConfig.from_yaml(temp_path)
        assert len(config.branches) == 1
        assert config.branches[0].exclusive is False
    finally:
        os.unlink(temp_path)
