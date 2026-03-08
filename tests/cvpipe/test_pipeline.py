# tests/cvpipe/test_pipeline.py

from __future__ import annotations

import time

import pytest

from cvpipe.bus import ResultBus
from cvpipe.component import Component
from cvpipe.config import BranchSpec
from cvpipe.errors import PipelineConfigError
from cvpipe.event import EventBus
from cvpipe.frame import Frame, SlotSchema
from cvpipe.pipeline import Pipeline
from tests.cvpipe.conftest import CounterSource


class _SimpleComponentA(Component):
    INPUTS: list[SlotSchema] = []
    OUTPUTS: list[SlotSchema] = [
        SlotSchema("slot_a", int),
    ]

    def process(self, frame: Frame) -> None:
        frame.slots["slot_a"] = 42


class _SimpleComponentB(Component):
    INPUTS: list[SlotSchema] = [
        SlotSchema("slot_a", int),
    ]
    OUTPUTS: list[SlotSchema] = [
        SlotSchema("slot_b", int),
    ]

    def process(self, frame: Frame) -> None:
        frame.slots["slot_b"] = frame.slots["slot_a"] + 1


class _SimpleComponentC(Component):
    INPUTS: list[SlotSchema] = [
        SlotSchema("slot_b", int),
    ]
    OUTPUTS: list[SlotSchema] = []

    def process(self, frame: Frame) -> None:
        pass


class _ComponentWithDuplicateOutput(Component):
    OUTPUTS: list[SlotSchema] = [
        SlotSchema("shared_slot", int),
    ]

    def process(self, frame: Frame) -> None:
        pass


class TestPipelineInit:
    def test_pipeline_init(self) -> None:
        source = CounterSource(1)
        comp = _SimpleComponentA()
        pipeline = Pipeline(
            source=source,
            components=[comp],
            result_bus=ResultBus(),
            event_bus=EventBus(),
        )
        assert pipeline._source is source
        assert pipeline._components == [comp]
        assert pipeline.is_running is False

    def test_pipeline_init_default_buses(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(source=source, components=[])
        assert pipeline._result_bus is not None
        assert pipeline._event_bus is not None


class TestPipelineValidation:
    def test_pipeline_validate_empty(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(source=source, components=[])
        pipeline.validate()

    def test_pipeline_validate_linear(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA(), _SimpleComponentB()],
        )
        pipeline.validate()

    def test_pipeline_validate_duplicate_slot(self) -> None:
        source = CounterSource(1)

        class DuplicateOutputComp(Component):
            OUTPUTS: list[SlotSchema] = [
                SlotSchema("slot_x", int),
            ]

            def process(self, frame: Frame) -> None:
                pass

        pipeline = Pipeline(
            source=source,
            components=[DuplicateOutputComp(), DuplicateOutputComp()],
        )
        with pytest.raises(PipelineConfigError) as exc:
            pipeline.validate()
        assert "multiple components" in str(exc.value).lower()

    def test_pipeline_validate_missing_input(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentB()],
        )
        with pytest.raises(PipelineConfigError) as exc:
            pipeline.validate()
        assert "requires input slot" in str(exc.value).lower()

    def test_pipeline_validate_coord_mismatch(self) -> None:
        source = CounterSource(1)

        class ProducerWithCoord(Component):
            OUTPUTS: list[SlotSchema] = [
                SlotSchema("box", int, coord_system="xyxy"),
            ]

            def process(self, frame: Frame) -> None:
                pass

        class ConsumerWithDifferentCoord(Component):
            INPUTS: list[SlotSchema] = [
                SlotSchema("box", int, coord_system="xywh"),
            ]
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

        pipeline = Pipeline(
            source=source,
            components=[ProducerWithCoord(), ConsumerWithDifferentCoord()],
        )
        with pytest.raises(PipelineConfigError) as exc:
            pipeline.validate()
        assert "coordinate system mismatch" in str(exc.value).lower()

    def test_pipeline_validate_shape_mismatch(self) -> None:
        source = CounterSource(1)

        class ProducerWithShape(Component):
            OUTPUTS: list[SlotSchema] = [
                SlotSchema("vec", int, shape=(3,)),
            ]

            def process(self, frame: Frame) -> None:
                pass

        class ConsumerWithShape(Component):
            INPUTS: list[SlotSchema] = [
                SlotSchema("vec", int, shape=(4,)),
            ]
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

        pipeline = Pipeline(
            source=source,
            components=[ProducerWithShape(), ConsumerWithShape()],
        )
        with pytest.raises(PipelineConfigError) as exc:
            pipeline.validate()
        assert "shape" in str(exc.value).lower()

    def test_pipeline_validate_exclusive_branch_invalid_inject_after(self) -> None:
        source = CounterSource(1)
        comp = _SimpleComponentA()
        pipeline = Pipeline(
            source=source,
            components=[comp],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="nonexistent",
                    merge_before="nonexistent2",
                    exclusive=True,
                )
            ],
        )
        with pytest.raises(PipelineConfigError):
            pipeline.validate()

    def test_pipeline_validate_all_errors_collected(self) -> None:
        source = CounterSource(1)

        class BadComp(Component):
            INPUTS: list[SlotSchema] = [
                SlotSchema("missing", int),
            ]
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentB(), BadComp()],
        )
        with pytest.raises(PipelineConfigError) as exc:
            pipeline.validate()
        assert len(exc.value.errors) >= 2


class TestPipelineLifecycle:
    def test_pipeline_start_stop(self) -> None:
        source = CounterSource(3)
        comp = _SimpleComponentA()
        result_bus = ResultBus(capacity=4)
        event_bus = EventBus()
        pipeline = Pipeline(
            source=source,
            components=[comp],
            result_bus=result_bus,
            event_bus=event_bus,
        )
        pipeline.validate()
        pipeline.start()

        assert pipeline.is_running is True
        time.sleep(0.2)

        pipeline.stop()
        assert pipeline.is_running is False

    def test_pipeline_start_not_validated(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(source=source, components=[_SimpleComponentA()])
        pipeline.start()
        time.sleep(0.1)
        pipeline.stop()
        assert pipeline._validated is True

    def test_pipeline_start_twice(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(source=source, components=[_SimpleComponentA()])
        pipeline.start()
        with pytest.raises(RuntimeError, match="already running"):
            pipeline.start()
        pipeline.stop()


class TestPipelineComponentLookup:
    def test_pipeline_component(self) -> None:
        source = CounterSource(1)
        comp = _SimpleComponentA()
        comp._component_id = "comp_a"
        pipeline = Pipeline(source=source, components=[comp])

        found = pipeline.component("comp_a")
        assert found is comp

    def test_pipeline_component_not_found(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(source=source, components=[])

        with pytest.raises(KeyError) as exc:
            pipeline.component("nonexistent")
        assert "nonexistent" in str(exc.value)


class TestPipelineReset:
    def test_pipeline_reset(self) -> None:
        source = CounterSource(10)
        reset_called: list[bool] = []

        class ResettableComponent(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

            def reset(self) -> None:
                reset_called.append(True)

        pipeline = Pipeline(source=source, components=[ResettableComponent()])
        pipeline.start()
        time.sleep(0.1)
        pipeline.reset()
        time.sleep(0.05)
        pipeline.stop()

        assert len(reset_called) >= 1

    def test_pipeline_reset_not_running(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(source=source, components=[_SimpleComponentA()])

        with pytest.raises(RuntimeError, match="requires the pipeline to be running"):
            pipeline.reset()


class TestPipelineExecutionPlan:
    def test_build_execution_plan_no_branches(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA(), _SimpleComponentB()],
        )
        segments = pipeline._build_execution_plan()
        assert len(segments) == 2

    def test_build_execution_plan_with_branches(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA(), _SimpleComponentB()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="SimpleComponentA",
                    merge_before="SimpleComponentB",
                    exclusive=True,
                )
            ],
        )
        segments = pipeline._build_execution_plan()
        assert len(segments) >= 1


class TestPipelineProperties:
    def test_pipeline_result_bus_property(self) -> None:
        source = CounterSource(1)
        result_bus = ResultBus()
        pipeline = Pipeline(source=source, components=[], result_bus=result_bus)
        assert pipeline.result_bus is result_bus

    def test_pipeline_event_bus_property(self) -> None:
        source = CounterSource(1)
        event_bus = EventBus()
        pipeline = Pipeline(source=source, components=[], event_bus=event_bus)
        assert pipeline.event_bus is event_bus


class TestPipelineBranchWiring:
    def test_pipeline_start_with_branch_components(self) -> None:
        source = CounterSource(1)
        branch_comp = _SimpleComponentA()
        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="SimpleComponentA",
                    merge_before="SimpleComponentA",
                    exclusive=False,
                )
            ],
            branch_components={"branch1": [branch_comp]},
        )
        pipeline.validate()
        pipeline.start()
        time.sleep(0.1)
        pipeline.stop()

        assert branch_comp._component_id != ""


class TestPipelineNestedBranches:
    def test_partition_range_nested_branches(self) -> None:
        source = CounterSource(1)

        class CompA(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = [SlotSchema("a", int)]

            def process(self, frame: Frame) -> None:
                frame.slots["a"] = 1

        class CompB(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("a", int)]
            OUTPUTS: list[SlotSchema] = [SlotSchema("b", int)]

            def process(self, frame: Frame) -> None:
                frame.slots["b"] = 2

        class CompC(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("b", int)]
            OUTPUTS: list[SlotSchema] = [SlotSchema("c", int)]

            def process(self, frame: Frame) -> None:
                frame.slots["c"] = 3

        class CompD(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("c", int)]
            OUTPUTS: list[SlotSchema] = [SlotSchema("d", int)]

            def process(self, frame: Frame) -> None:
                frame.slots["d"] = 4

        class BranchCompX(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("a", int)]
            OUTPUTS: list[SlotSchema] = [SlotSchema("x", int)]

            def process(self, frame: Frame) -> None:
                frame.slots["x"] = 100

        class BranchCompY(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("b", int)]
            OUTPUTS: list[SlotSchema] = [SlotSchema("y", int)]

            def process(self, frame: Frame) -> None:
                frame.slots["y"] = 200

        pipeline = Pipeline(
            source=source,
            components=[CompA(), CompB(), CompC(), CompD()],
            branches=[
                BranchSpec(
                    id="outer",
                    components=[],
                    trigger="x",
                    inject_after="CompA",
                    merge_before="CompD",
                    exclusive=True,
                ),
                BranchSpec(
                    id="inner",
                    components=[],
                    trigger="y",
                    inject_after="CompB",
                    merge_before="CompC",
                    exclusive=True,
                ),
            ],
            branch_components={
                "outer": [BranchCompX()],
                "inner": [BranchCompY()],
            },
        )
        segments = pipeline._build_execution_plan()
        assert len(segments) >= 1

    def test_partition_range_components_before_and_after_branches(self) -> None:
        source = CounterSource(1)

        class CompA(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = [SlotSchema("a", int)]

            def process(self, frame: Frame) -> None:
                frame.slots["a"] = 1

        class CompB(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("a", int)]
            OUTPUTS: list[SlotSchema] = [SlotSchema("b", int)]

            def process(self, frame: Frame) -> None:
                frame.slots["b"] = 2

        class CompC(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("b", int)]
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

        pipeline = Pipeline(
            source=source,
            components=[CompA(), CompB(), CompC()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="CompA",
                    merge_before="CompC",
                    exclusive=True,
                )
            ],
            branch_components={"branch1": []},
        )
        segments = pipeline._build_execution_plan()
        assert len(segments) >= 1


class TestPipelineValidationErrors:
    def test_validate_exclusive_branch_inject_after_not_found(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="NonExistent",
                    merge_before="SimpleComponentA",
                    exclusive=True,
                )
            ],
        )
        with pytest.raises(PipelineConfigError) as exc:
            pipeline.validate()
        assert "inject_after" in str(exc.value).lower()

    def test_validate_exclusive_branch_merge_before_not_found(self) -> None:
        source = CounterSource(1)
        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="SimpleComponentA",
                    merge_before="NonExistent",
                    exclusive=True,
                )
            ],
        )
        with pytest.raises(PipelineConfigError) as exc:
            pipeline.validate()
        assert "merge_before" in str(exc.value).lower()

    def test_validate_exclusive_branch_inject_after_after_merge_before(self) -> None:
        source = CounterSource(1)

        class CompA(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = [SlotSchema("a", int)]

            def process(self, frame: Frame) -> None:
                pass

        class CompB(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = [SlotSchema("b", int)]

            def process(self, frame: Frame) -> None:
                pass

        pipeline = Pipeline(
            source=source,
            components=[CompA(), CompB()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="CompB",
                    merge_before="CompA",
                    exclusive=True,
                )
            ],
        )
        with pytest.raises(PipelineConfigError) as exc:
            pipeline.validate()
        assert "inject_after must precede merge_before" in str(exc.value).lower()

    def test_validate_branch_requires_skipped_main_path_output(self) -> None:
        source = CounterSource(1)

        class CompA(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = [SlotSchema("a", int)]

            def process(self, frame: Frame) -> None:
                pass

        class CompB(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("a", int)]
            OUTPUTS: list[SlotSchema] = [SlotSchema("b", int)]

            def process(self, frame: Frame) -> None:
                pass

        class CompC(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("b", int)]
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

        class BranchComp(Component):
            INPUTS: list[SlotSchema] = [SlotSchema("b", int)]
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

        pipeline = Pipeline(
            source=source,
            components=[CompA(), CompB(), CompC()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="CompA",
                    merge_before="CompC",
                    exclusive=True,
                )
            ],
            branch_components={"branch1": [BranchComp()]},
        )
        with pytest.raises(PipelineConfigError) as exc:
            pipeline.validate()
        assert (
            "skipped main-path" in str(exc.value).lower()
            or "only produced by skipped" in str(exc.value).lower()
        )


class TestPipelineComponentLookupInBranch:
    def test_pipeline_component_in_branch(self) -> None:
        source = CounterSource(1)
        branch_comp = _SimpleComponentA()
        branch_comp._component_id = "branch_comp"
        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="SimpleComponentA",
                    merge_before="SimpleComponentA",
                    exclusive=False,
                )
            ],
            branch_components={"branch1": [branch_comp]},
        )

        found = pipeline.component("branch_comp")
        assert found is branch_comp


class TestPipelineSetupTeardownErrors:
    def test_pipeline_component_setup_exception(self) -> None:
        source = CounterSource(1)

        class FailingComponent(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

            def setup(self) -> None:
                raise RuntimeError("setup failed")

        pipeline = Pipeline(source=source, components=[FailingComponent()])

        with pytest.raises(RuntimeError, match="setup failed"):
            pipeline.start()

    def test_pipeline_branch_component_setup_exception(self) -> None:
        source = CounterSource(1)

        class BranchFailingComponent(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

            def setup(self) -> None:
                raise RuntimeError("branch setup failed")

        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="SimpleComponentA",
                    merge_before="SimpleComponentA",
                    exclusive=False,
                )
            ],
            branch_components={"branch1": [BranchFailingComponent()]},
        )

        with pytest.raises(RuntimeError, match="branch setup failed"):
            pipeline.start()

    def test_pipeline_component_teardown_exception(self) -> None:
        source = CounterSource(3)

        class TeardownFailingComponent(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

            def teardown(self) -> None:
                raise RuntimeError("teardown failed")

        pipeline = Pipeline(source=source, components=[TeardownFailingComponent()])
        pipeline.start()
        time.sleep(0.1)
        pipeline.stop()

    def test_pipeline_branch_component_teardown_exception(self) -> None:
        source = CounterSource(3)

        class BranchTeardownFailingComponent(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

            def teardown(self) -> None:
                raise RuntimeError("branch teardown failed")

        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="SimpleComponentA",
                    merge_before="SimpleComponentA",
                    exclusive=False,
                )
            ],
            branch_components={"branch1": [BranchTeardownFailingComponent()]},
        )
        pipeline.start()
        time.sleep(0.1)
        pipeline.stop()


class TestPipelineResetErrors:
    def test_pipeline_reset_component_exception(self) -> None:
        source = CounterSource(10)

        class ResetFailingComponent(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

            def reset(self) -> None:
                raise RuntimeError("reset failed")

        pipeline = Pipeline(source=source, components=[ResetFailingComponent()])
        pipeline.start()
        time.sleep(0.1)
        pipeline.reset()
        time.sleep(0.05)
        pipeline.stop()

    def test_pipeline_reset_branch_component_exception(self) -> None:
        source = CounterSource(10)

        class BranchResetFailingComponent(Component):
            INPUTS: list[SlotSchema] = []
            OUTPUTS: list[SlotSchema] = []

            def process(self, frame: Frame) -> None:
                pass

            def reset(self) -> None:
                raise RuntimeError("branch reset failed")

        pipeline = Pipeline(
            source=source,
            components=[_SimpleComponentA()],
            branches=[
                BranchSpec(
                    id="branch1",
                    components=[],
                    trigger="x",
                    inject_after="SimpleComponentA",
                    merge_before="SimpleComponentA",
                    exclusive=False,
                )
            ],
            branch_components={"branch1": [BranchResetFailingComponent()]},
        )
        pipeline.start()
        time.sleep(0.1)
        pipeline.reset()
        time.sleep(0.05)
        pipeline.stop()


class TestPipelineValidationEdgeCases:
    pass
