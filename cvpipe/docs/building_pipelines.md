# Building Pipelines

A pipeline has three parts: a **source** that produces frames, **components** that process
them in order, and optional **branches** for conditional routing.

---

## YAML structure

```yaml
pipeline:
  source: webcam_source           # FrameSource module name (dir under components/)
  source_config:                  # optional — passed as kwargs to source constructor
    device_index: 0

  components:                     # main execution path, in order
    - module: preprocessor        # component directory name
      id: prep                    # unique ID — used in branches, pipeline.component(), logs
      config:                     # optional — passed as kwargs to component constructor
        device: cuda

    - module: faster_rcnn_detector
      id: detector
      config:
        weights: checkpoints/frcnn.pth
        confidence: 0.5

    - module: tracker
      id: tracker

    - module: result_assembler
      id: assembler

  branches:                       # optional
    - id: fast_mode
      trigger: "mode == 'fast'"
      inject_after: prep
      merge_before: tracker
      exclusive: true
      components:
        - module: yolo_detector
          id: fast_det
          config: {weights: yolov8n.pt, confidence: 0.3}
```

---

## Building with cvpipe.build()

The standard way to assemble a pipeline. `build()` handles discovery, construction, and
wiring from a single call:

```python
from cvpipe import build
from pathlib import Path

pipeline = build(
    config_path=Path("pipeline.yaml"),
    components_dir=Path("myapp/"),
)
pipeline.validate()
pipeline.start()
```

`build()` discovers all `Component` and `FrameSource` subclasses from every subdirectory
under `components_dir` that contains an `__init__.py`. Directory name becomes registry
key. Instantiates each with the `config` dict from YAML. Returns a ready-to-validate
`Pipeline`.

The `components_dir` argument is the root of your project's component and source trees.
Discovery walks all subdirectories, so the conventional layout works without any
additional configuration:

```
myapp/
└── components/
   ├── preprocessor/__init__.py
   ├── faster_rcnn_detector/__init__.py
   └── tracker/__init__.py
   └── webcam_source/__init__.py
```

---

## Manual construction (advanced)

When you need post-construction dependency injection — sharing live Python objects
between components that cannot be expressed in YAML — you build the pipeline manually:

```python
from cvpipe import Pipeline, ComponentRegistry
from cvpipe.config import PipelineConfig
from pathlib import Path

registry = ComponentRegistry()
registry.discover(Path("myapp/components/"))
registry.discover(Path("myapp/sources/"))

config = PipelineConfig.from_yaml(Path("pipeline.yaml"))

# Instantiate main-path components
components = []
for spec in config.components:
    cls  = registry.get(spec.module)
    comp = cls(**spec.config)
    comp._component_id = spec.id
    components.append(comp)

# Instantiate branch components
branch_components = {}
for branch_spec in config.branches:
    b_comps = []
    for spec in branch_spec.components:
        cls  = registry.get(spec.module)
        comp = cls(**spec.config)
        comp._component_id = spec.id
        b_comps.append(comp)
    branch_components[branch_spec.id] = b_comps

# Instantiate source from registry
source = registry.get_source(config.source)(**config.source_config)

pipeline = Pipeline(
    source=source,
    components=components,
    connections=config.connections,
    branches=config.branches,
    branch_components=branch_components,
)

# Post-construction dependency injection — shared live objects
scorer  = pipeline.component("scorer")
tracker = pipeline.component("tracker")
scorer._shared_registry = my_registry_object   # cannot be expressed in YAML
scorer._db = db_connection

pipeline.validate()
pipeline.start()
```

This pattern (manual construction + post-construction injection) lives in a
`pipeline_factory.py` function in your project.

---

## Validation

Always call `validate()` before `start()`:

```python
pipeline.validate()   # raises PipelineConfigError listing every issue found
pipeline.start()
```

`start()` validates automatically if you haven't, but explicit validation surfaces errors
at assembly time — before seconds of model loading have begun.

What validation checks:
- A component requires a slot that no upstream component produces
- Two components declare the same slot name in `OUTPUTS`
- Writer and reader declare different coordinate systems for the same slot
- Branch `inject_after` or `merge_before` IDs that don't exist in the component list
- Exclusive branch ranges that partially overlap

---

## Pipeline lifecycle

```python
pipeline = Pipeline(source=source, components=[prep, detector, tracker])

pipeline.validate()
pipeline.start()    # setup() on source then all components, then starts frame loop

# ... pipeline is running ...

pipeline.stop()     # waits for current frame, then teardown() in reverse order
```

**`start()` sequence:**
1. Validates (skips if already validated)
2. Injects `_component_id` and `_event_bus` into every component
3. Wires `EventBus` subscriptions from each component's `SUBSCRIBES` list
4. Starts `EventBus` and `ResultBus` threads
5. Calls `source.setup()`, then `setup()` on each component in pipeline order
6. Creates `Scheduler` and starts the streaming thread
7. Emits `PipelineStateEvent(state="running")`

**`stop()` sequence:**
1. Emits `PipelineStateEvent(state="stopping")`
2. Stops the Scheduler (waits for the current frame to finish)
3. Calls `teardown()` on each component in reverse order, then `source.teardown()`
4. Stops `ResultBus` and `EventBus` threads
5. Emits `PipelineStateEvent(state="stopped")`

---

## Resetting the pipeline

`pipeline.reset()` resets all component state without stopping the pipeline:

```python
pipeline.reset()    # pauses Scheduler, calls reset() on all components, resumes
```

**`reset()` sequence:**
1. Pauses the Scheduler (waits for the current frame to complete)
2. Calls `reset()` on each component in topological order
3. Resumes the Scheduler
4. Emits `PipelineStateEvent(state="reset")`

The EventBus keeps running during reset, so `on_event()` can still fire. If a
component's `reset()` accesses state also touched by `on_event()`, acquire `self._lock`.

---

## Subscribing to results and events

Subscribe to events **before calling `start()`** — the buses start when the pipeline
does, and subscriptions added after have a small window where events can be missed.

```python
pipeline = Pipeline(source=source, components=[...])

# High-frequency per-frame results
pipeline.result_bus.subscribe(lambda result: queue.put(result))

# Pipeline lifecycle events
pipeline.event_bus.subscribe(PipelineStateEvent,
                              lambda e: logger.info("state: %s", e.state))
pipeline.event_bus.subscribe(ComponentErrorEvent,
                              lambda e: logger.error("%s: %s", e.component_id, e.message))

pipeline.start()
```

For async applications, use `AsyncQueueBridge` — see [Observability](./observability.md).

---

## Accessing a component at runtime

`pipeline.component(id)` retrieves any live component by its YAML ID. Searches main-path
and branch components:

```python
# Post-construction dependency injection
scorer = pipeline.component("scorer")
scorer._shared_registry = my_registry   # inject a live object after construction

# Call a service method from the API layer (outside process())
classifier = pipeline.component("classifier")
embedding  = classifier.encode_query_image(query_image)  # call from a thread pool
```

---

## Branches

A branch is a conditional sub-pipeline. The trigger is a Python expression evaluated
against `frame.meta`:

```yaml
branches:
  - id: nighttime
    trigger: "illumination < 50"
    inject_after: preprocessor
    merge_before: tracker
    components:
      - module: low_light_enhancer
        id: enhancer
```

When `frame.meta["illumination"] < 50`, `low_light_enhancer` runs between `preprocessor`
and `tracker`. When 50 or above, the main-path components in that range run instead.

If the key doesn't exist in `frame.meta` when the trigger evaluates, the trigger returns
`False` and the main path runs — no exception.

### Exclusive branches

A regular branch is **additive** — branch components run in addition to the main path.
An **exclusive** branch is an **if/else** — when the trigger fires, main-path components
in the covered range are skipped entirely:

```yaml
branches:
  - id: inference_paused
    trigger: "inference_enabled == False"
    inject_after: preprocessor
    merge_before: assembler
    exclusive: true
    components:
      - module: passthrough_marker
        id: passthrough
```

When `inference_enabled` is `False`: `preprocessor` runs, `passthrough` runs, `assembler`
runs — everything between `preprocessor` and `assembler` is skipped.

An exclusive branch with an empty `components` list simply skips the covered segment:

```yaml
- id: skip_segment
  trigger: "skip == True"
  inject_after: prep
  merge_before: assembler
  exclusive: true
  components: []
```

### Nested branches

Branches can nest as long as one range is fully inside the other. Partial overlap is a
validation error. The typical pattern: outer branch disables all inference, inner branch
selects between detectors:

```yaml
branches:
  - id: inference_off
    trigger: "not inference_enabled"
    inject_after: preprocessor
    merge_before: assembler
    exclusive: true
    components:
      - module: passthrough_marker
        id: passthrough

  - id: fast_mode
    trigger: "mode == 'fast'"
    inject_after: preprocessor
    merge_before: tracker
    exclusive: true
    components:
      - module: yolo_detector
        id: fast_det
```

### Writing trigger keys

The trigger key must be in `frame.meta` before the branch point (`inject_after`).
Write it in a component that runs earlier in the pipeline:

```python
# In Preprocessor.process():
frame.meta["inference_enabled"] = self._inference_enabled
frame.meta["mode"] = "fast" if self._fast_mode else "full"
```

Keep trigger expressions simple — one key, one condition. If the logic is complex, put it
in a dedicated routing component and write a single decision string to `frame.meta`.

---

## Recommendations

**Validate explicitly.** Call `pipeline.validate()` before `start()` so errors surface
before model loading begins.

**Use `build()` for standard pipelines.** Manual construction is only needed for
post-construction dependency injection of live Python objects.

**Swapping models means swapping modules.** To replace Faster R-CNN with YOLOv8, change
`module: faster_rcnn_detector` to `module: yolo_detector` in `pipeline.yaml`. The
tracker, assembler, and server are unchanged. This is the payoff of typed slot contracts.

**Stable component IDs matter.** IDs appear in `inject_after`, `merge_before`,
`pipeline.component()`, logs, and metrics. Once assigned, changing an ID means updating
everything that references it.

---

## Next Steps

- [Observability](./observability.md) — Monitor pipeline health
- [API Reference](./api_reference.md) — Complete reference for Pipeline, Registry, and config
