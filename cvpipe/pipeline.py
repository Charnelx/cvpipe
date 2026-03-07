# cvpipe/pipeline.py

from __future__ import annotations

import logging

from .bus import ResultBus
from .component import Component
from .config import BranchSpec
from .errors import PipelineConfigError
from .event import EventBus, PipelineStateEvent
from .frame import SlotSchema
from .scheduler import Scheduler, FrameSource, ExecutionSegment, ExclusiveBranch

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Assembled, validated, runnable pipeline.

    Do not construct directly — use pipeline_factory.build() in the
    application layer, which handles component instantiation from YAML.

    For tests and advanced use, construct with explicit components:

        pipeline = Pipeline(
            source=my_source,
            components=[comp_a, comp_b],
            result_bus=ResultBus(),
            event_bus=EventBus(),
        )
        pipeline.validate()
        pipeline.start()

    Public API (used by server.py):
        pipeline.result_bus  → ResultBus
        pipeline.event_bus   → EventBus
        pipeline.validate()  → None or raises PipelineConfigError
        pipeline.start()     → None
        pipeline.stop()      → None
        pipeline.is_running  → bool
    """

    def __init__(
        self,
        source: FrameSource,
        components: list[Component],
        result_bus: ResultBus | None = None,
        event_bus: EventBus | None = None,
        connections: dict[str, list[str]] | None = None,
        branches: list[BranchSpec] | None = None,
        branch_components: dict[str, list[Component]] | None = None,
    ) -> None:
        self._source = source
        self._components = list(components)
        self._result_bus = result_bus or ResultBus(capacity=4)
        self._event_bus = event_bus or EventBus()
        self._connections = connections
        self._branches = branches or []
        self._branch_components = branch_components or {}
        self._scheduler: Scheduler | None = None
        self._validated = False

    @property
    def result_bus(self) -> ResultBus:
        return self._result_bus

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.is_running

    def validate(self) -> None:
        """
        Validate all component contracts in the pipeline graph.

        Collects ALL errors before raising. Raises PipelineConfigError
        with the full list of errors if any are found.

        Must be called before start(). start() calls validate() automatically
        if it has not been called, but explicit validation is recommended
        to surface errors before server startup.

        Raises
        ------
        PipelineConfigError
            If one or more contract violations are found.
        """
        errors = self._collect_errors()
        if errors:
            raise PipelineConfigError(errors)
        self._validated = True
        logger.info("[Pipeline] Validation passed (%d components)", len(self._components))

    def start(self) -> None:
        """
        Start the pipeline.

        1. Validates if not already validated
        2. Injects _component_id and _event_bus into each component
        3. Calls setup() on each component in order
        4. Wires EventBus subscriptions from SUBSCRIBES declarations
        5. Starts EventBus dispatch thread
        6. Starts ResultBus subscriber threads
        7. Calls source.setup()
        8. Creates and starts Scheduler
        9. Emits PipelineStateEvent("running")

        Raises
        ------
        PipelineConfigError
            If validation fails.
        RuntimeError
            If pipeline is already running.
        """
        if self.is_running:
            raise RuntimeError("Pipeline is already running")

        if not self._validated:
            self.validate()

        for comp in self._components:
            comp._component_id = self._get_component_id(comp)
            comp._event_bus = self._event_bus

        for comp in self._components:
            if comp.SUBSCRIBES:
                for event_type in comp.SUBSCRIBES:
                    self._event_bus.subscribe(event_type, comp.on_event)

        all_components = list(self._components)
        for branch_id, branch_comps in self._branch_components.items():
            for comp in branch_comps:
                comp._component_id = self._get_component_id(comp)
                comp._event_bus = self._event_bus

            for comp in branch_comps:
                if comp.SUBSCRIBES:
                    for event_type in comp.SUBSCRIBES:
                        self._event_bus.subscribe(event_type, comp.on_event)

            all_components.extend(branch_comps)

        self._event_bus.start()
        self._event_bus.publish(PipelineStateEvent(state="starting"))
        self._result_bus.start()

        self._source.setup()
        for comp in self._components:
            try:
                comp.setup()
            except Exception as e:
                logger.exception("[Pipeline] Component '%s' setup() failed", comp.component_id)
                self._event_bus.publish(PipelineStateEvent(state="error", detail=str(e)))
                raise

        for branch_id, branch_comps in self._branch_components.items():
            for comp in branch_comps:
                try:
                    comp.setup()
                except Exception as e:
                    logger.exception(
                        "[Pipeline] Branch component '%s' setup() failed",
                        comp.component_id,
                    )
                    self._event_bus.publish(PipelineStateEvent(state="error", detail=str(e)))
                    raise

        segments = self._build_execution_plan()

        self._scheduler = Scheduler(
            source=self._source,
            segments=segments,
            result_bus=self._result_bus,
            event_bus=self._event_bus,
        )
        self._scheduler.start()
        self._event_bus.publish(PipelineStateEvent(state="running"))
        logger.info("[Pipeline] Started — %d components", len(self._components))

    def stop(self) -> None:
        """
        Stop the pipeline gracefully.

        1. Emits PipelineStateEvent("stopping")
        2. Stops Scheduler (waits for current frame to complete)
        3. Calls teardown() on each component in reverse order
        4. Calls source.teardown()
        5. Stops ResultBus subscriber threads
        6. Stops EventBus dispatch thread
        7. Emits PipelineStateEvent("stopped")
        """
        self._event_bus.publish(PipelineStateEvent(state="stopping"))

        if self._scheduler is not None:
            self._scheduler.stop()

        for comp in reversed(self._components):
            try:
                comp.teardown()
            except Exception:
                logger.exception("[Pipeline] Component '%s' teardown() failed", comp.component_id)

        for branch_id, branch_comps in self._branch_components.items():
            for comp in reversed(branch_comps):
                try:
                    comp.teardown()
                except Exception:
                    logger.exception(
                        "[Pipeline] Branch component '%s' teardown() failed",
                        comp.component_id,
                    )

        self._source.teardown()
        self._result_bus.stop()
        self._event_bus.publish(PipelineStateEvent(state="stopped"))
        self._event_bus.stop()
        logger.info("[Pipeline] Stopped")

    def reset(self) -> None:
        """
        Reset all component application state without stopping the pipeline.

        Sequence:
          1. Pause the Scheduler (waits for current frame to complete)
          2. Call comp.reset() on each component in topological order
          3. Resume the Scheduler
          4. Emit PipelineStateEvent(state="reset") on the EventBus

        Thread safety:
            reset() is called from the main/API thread.
            The Scheduler is paused before any comp.reset() is called.
            process() is guaranteed not to run concurrently with any reset() call.
            on_event() may still be called during reset (EventBus continues running).
            If a component's reset() accesses state also accessed by on_event(),
            it must acquire self._lock.

        Error handling:
            If any comp.reset() raises, the exception is logged and the next
            component's reset() is still called. The Scheduler is always resumed,
            even if all reset() calls raise. This prevents the pipeline from
            becoming permanently stuck in a paused state.

        Raises
        ------
        RuntimeError
            If the pipeline is not currently running (not started, or already stopped).
            reset() requires the Scheduler to be active in order to pause/resume it.
        """
        if not self.is_running:
            raise RuntimeError(
                "Pipeline.reset() requires the pipeline to be running. Call start() before reset()."
            )
        assert self._scheduler is not None

        self._scheduler.pause()
        try:
            for comp in self._components:
                try:
                    comp.reset()
                except Exception:
                    logger.exception(
                        "[Pipeline] Component '%s' reset() raised — continuing",
                        comp.component_id,
                    )

            for branch_id, branch_comps in self._branch_components.items():
                for comp in branch_comps:
                    try:
                        comp.reset()
                    except Exception:
                        logger.exception(
                            "[Pipeline] Branch component '%s' reset() raised — continuing",
                            comp.component_id,
                        )
        finally:
            self._scheduler.resume()

        self._event_bus.publish(PipelineStateEvent(state="reset"))
        total_components = len(self._components) + sum(
            len(c) for c in self._branch_components.values()
        )
        logger.info("[Pipeline] reset() complete — %d components reset", total_components)

    def _build_execution_plan(self) -> list[ExecutionSegment]:
        """
        Partition the component list into ExecutionSegments according to
        exclusive branch boundaries.

        Algorithm:
        1. Identify all exclusive branches and their covered ranges
           (inject_after index, merge_before index) in self._components.
        2. Recursively partition the component list at exclusive branch boundaries.
        3. For each branch, find nested branches within its range and recursively
           partition the sub-range.
        4. Attach each exclusive branch to the segment covering its range.
        5. Components outside all exclusive branch ranges form their own segments.
        6. Non-exclusive branches are wired via the existing additive mechanism (unchanged).

        Returns
        -------
        list[ExecutionSegment]
            Ordered segments suitable for Scheduler construction.
        """
        exclusive_branches = [b for b in self._branches if b.exclusive]

        if not exclusive_branches:
            return [ExecutionSegment(components=[c]) for c in self._components]

        component_ids = [self._get_component_id(c) for c in self._components]
        id_to_index = {cid: i for i, cid in enumerate(component_ids)}

        branch_info_list: list[tuple[int, int, ExclusiveBranch]] = []
        for branch_spec in exclusive_branches:
            inject_idx = id_to_index.get(branch_spec.inject_after)
            merge_idx = id_to_index.get(branch_spec.merge_before)
            if inject_idx is None or merge_idx is None:
                continue
            if inject_idx < merge_idx:
                branch_components = self._branch_components.get(branch_spec.id, [])
                exclusive = ExclusiveBranch(
                    branch_id=branch_spec.id,
                    trigger_src=branch_spec.trigger,
                    components=branch_components,
                )
                branch_info_list.append((inject_idx, merge_idx, exclusive))

        if not branch_info_list:
            return [ExecutionSegment(components=[c]) for c in self._components]

        return self._partition_range(0, len(self._components), branch_info_list)

    def _partition_range(
        self,
        start: int,
        end: int,
        branch_info_list: list[tuple[int, int, ExclusiveBranch]],
    ) -> list[ExecutionSegment]:
        """
        Recursively partition a component range by exclusive branch boundaries.

        Parameters
        ----------
        start : int
            Start index (inclusive) of the range to partition.
        end : int
            End index (exclusive) of the range to partition.
        branch_info_list : list[tuple[int, int, ExclusiveBranch]]
            List of (inject_idx, merge_idx, exclusive_branch) for all branches.

        Returns
        -------
        list[ExecutionSegment]
            Partitioned segments for this range.
        """
        segments: list[ExecutionSegment] = []
        current_idx = start

        branches_in_range = [
            (inj, mrg, exc) for inj, mrg, exc in branch_info_list if inj >= start and mrg <= end
        ]

        branches_in_range.sort(key=lambda x: x[0])

        for inject_idx, merge_idx, exclusive in branches_in_range:
            if current_idx <= inject_idx:
                segments.append(
                    ExecutionSegment(components=self._components[current_idx: inject_idx + 1])
                )

            inner_branches = [
                (inj, mrg, exc)
                for inj, mrg, exc in branch_info_list
                if inj > inject_idx and mrg < merge_idx
            ]

            if inner_branches:
                inner_segments = self._partition_range(inject_idx + 1, merge_idx, branch_info_list)
                segments.append(
                    ExecutionSegment(
                        components=[],
                        exclusive_branch=exclusive,
                        nested_segments=inner_segments,
                    )
                )
            else:
                segments.append(
                    ExecutionSegment(
                        components=self._components[inject_idx:merge_idx],
                        exclusive_branch=exclusive,
                    )
                )

            current_idx = merge_idx

        if current_idx < end:
            segments.append(ExecutionSegment(components=self._components[current_idx:end]))

        return segments

    def _collect_errors(self) -> list[str]:
        errors: list[str] = []

        writers: dict[str, list[str]] = {}
        for comp in self._components:
            cid = self._get_component_id(comp)
            for schema in comp.OUTPUTS:
                writers.setdefault(schema.name, []).append(cid)
        for slot_name, cids in writers.items():
            if len(cids) > 1:
                errors.append(
                    f"Slot '{slot_name}' declared as OUTPUT by multiple components: "
                    f"{cids}. Each slot must have exactly one writer."
                )

        output_map: dict[str, tuple[str, SlotSchema]] = {}
        for comp in self._components:
            cid = self._get_component_id(comp)
            for schema in comp.OUTPUTS:
                output_map[schema.name] = (cid, schema)

        for i, comp in enumerate(self._components):
            cid = self._get_component_id(comp)
            upstream_outputs: dict[str, tuple[str, SlotSchema]] = {}
            for j, upstream in enumerate(self._components):
                if j >= i:
                    break
                uid = self._get_component_id(upstream)
                for schema in upstream.OUTPUTS:
                    upstream_outputs[schema.name] = (uid, schema)

            for input_schema in comp.INPUTS:
                if input_schema.name not in upstream_outputs:
                    errors.append(
                        f"Component '{cid}' requires input slot '{input_schema.name}' "
                        f"but no upstream component produces it."
                    )
                    continue

                upstream_id, upstream_schema = upstream_outputs[input_schema.name]
                compat_errors = upstream_schema.compatible_with(input_schema)
                for ce in compat_errors:
                    errors.append(f"Contract error between '{upstream_id}' → '{cid}': {ce}")

        exclusive_branches = [b for b in self._branches if b.exclusive]
        component_ids = {self._get_component_id(c) for c in self._components}

        for branch in exclusive_branches:
            if branch.inject_after not in component_ids:
                errors.append(
                    f"Exclusive branch '{branch.id}': inject_after '{branch.inject_after}' "
                    f"not found in pipeline components"
                )
            if branch.merge_before not in component_ids:
                errors.append(
                    f"Exclusive branch '{branch.id}': merge_before '{branch.merge_before}' "
                    f"not found in pipeline components"
                )

        for i, branch1 in enumerate(exclusive_branches):
            idx1_inject = None
            idx1_merge = None
            for j, c in enumerate(self._components):
                if self._get_component_id(c) == branch1.inject_after:
                    idx1_inject = j
                if self._get_component_id(c) == branch1.merge_before:
                    idx1_merge = j

            if idx1_inject is None or idx1_merge is None:
                continue
            if idx1_inject >= idx1_merge:
                errors.append(
                    f"Exclusive branch '{branch1.id}': inject_after must precede merge_before"
                )

            for branch2 in exclusive_branches[i + 1 :]:
                idx2_inject = None
                idx2_merge = None
                for j, c in enumerate(self._components):
                    if self._get_component_id(c) == branch2.inject_after:
                        idx2_inject = j
                    if self._get_component_id(c) == branch2.merge_before:
                        idx2_merge = j

                if idx2_inject is None or idx2_merge is None:
                    continue

                range1 = (idx1_inject + 1, idx1_merge - 1)
                range2 = (idx2_inject + 1, idx2_merge - 1)

                if range1[0] < range2[1] and range2[0] < range1[1]:
                    if not (range2[0] >= range1[0] and range2[1] <= range1[1]):
                        errors.append(
                            f"Exclusive branches '{branch1.id}' and '{branch2.id}' have "
                            f"overlapping ranges"
                        )

        for branch in exclusive_branches:
            branch_comps = self._branch_components.get(branch.id, [])
            if not branch_comps:
                continue

            inject_idx = None
            merge_idx = None
            for j, c in enumerate(self._components):
                if self._get_component_id(c) == branch.inject_after:
                    inject_idx = j
                if self._get_component_id(c) == branch.merge_before:
                    merge_idx = j

            if inject_idx is None or merge_idx is None:
                continue

            skipped_main_outputs: set[str] = set()
            for j in range(inject_idx + 1, merge_idx):
                comp = self._components[j]
                for schema in comp.OUTPUTS:
                    skipped_main_outputs.add(schema.name)

            available_outputs: dict[str, str] = {}
            for j in range(0, inject_idx + 1):
                comp = self._components[j]
                cid = self._get_component_id(comp)
                for schema in comp.OUTPUTS:
                    available_outputs[schema.name] = cid

            for branch_comp in branch_comps:
                cid = self._get_component_id(branch_comp)
                for input_schema in branch_comp.INPUTS:
                    if input_schema.name in skipped_main_outputs:
                        if input_schema.name not in available_outputs:
                            errors.append(
                                f"Exclusive branch '{branch.id}': component '{cid}' "
                                f"requires input slot '{input_schema.name}' which is only "
                                f"produced by skipped main-path components. Either add an "
                                f"equivalent producer before '{branch.inject_after}' or move "
                                f"the component into the main path."
                            )

        return errors

    def _get_component_id(self, comp: Component) -> str:
        """Return component's id, falling back to class name if not yet set."""
        return comp._component_id or type(comp).__name__

    def component(self, component_id: str) -> Component:
        """
        Retrieve a component instance by its pipeline ID.

        Parameters
        ----------
        component_id : str
            The ID assigned to the component in the pipeline configuration
            (the ``id`` field in the YAML spec, or the value assigned to
            ``comp._component_id`` before ``Pipeline.start()``).

        Returns
        -------
        Component
            The component instance with the given ID.

        Raises
        ------
        KeyError
            If no component with the given ID exists in this pipeline.

        Notes
        -----
        This method searches all main-path components and branch components.
        The component is accessible before and after ``start()``.
        The returned instance is the live component — calling methods on it
        directly (outside of ``process()`` or ``on_event()``) is the caller's
        responsibility to make thread-safe.

        Example::

            embedder = pipeline.component("embedder")
            result = embedder.embed_images(images)
        """
        for comp in self._components:
            if self._get_component_id(comp) == component_id:
                return comp
        for branch_id, branch_comps in self._branch_components.items():
            for comp in branch_comps:
                if self._get_component_id(comp) == component_id:
                    return comp
        available_ids = [self._get_component_id(c) for c in self._components]
        for branch_id, branch_comps in self._branch_components.items():
            available_ids.extend(self._get_component_id(c) for c in branch_comps)
        raise KeyError(
            f"No component with id {component_id!r} in pipeline. Available: {available_ids}"
        )
