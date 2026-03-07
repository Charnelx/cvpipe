# Building Pipelines

This guide explains how to configure and run cvpipe pipelines.

## YAML Structure

```yaml
pipeline:
  source: camera_source       # FrameSource module name
  source_config:              # optional: params passed to source
    device_index: 0

  components:                 # list of components in execution order
    - module: frcnn_proposer # module name (directory name)
      id: proposer           # unique component ID
      config:                # optional: constructor kwargs
        confidence: 0.5
        nms_iou: 0.45

    - module: dino_embedder
      id: embedder

  connections:                # optional: explicit connections
    embedder: [proposer]    # embedder reads from proposer outputs

  branches:                   # optional: conditional branches
    - id: scan_branch
      trigger: "routing_decision == 'scan'"
      inject_after: router
      merge_before: tracker
      components:
        - module: scan_proposer
          id: scan_proposer
```

## Auto-Inference of Connections

If you omit `connections`, cvpipe infers them automatically:

1. For each component input slot, find the most recent upstream component that produces it
2. Use that connection automatically

Example: If `Proposer` outputs `proposals_xyxy` and `Embedder` inputs `proposals_xyxy`, they're automatically connected.

## When to Use Explicit Connections

Use explicit connections when:
- Multiple components produce the same slot name
- You want non-linear data flow
- The auto-inference doesn't match your intent

## Component Discovery

cvpipe discovers components from a directory:

```python
from cvpipe import ComponentRegistry

registry = ComponentRegistry()
registry.discover(Path("detector/components/"))
```

Each subdirectory with an `__init__.py` that exports exactly one `Component` subclass becomes available by its directory name.

## Validation

Call `pipeline.validate()` to check for errors:

```python
pipeline = Pipeline(source=source, components=[...])
pipeline.validate()  # raises PipelineConfigError if invalid
```

Validation checks:
- Duplicate slot writers (two components output same slot)
- Missing input slots (component requires slot no one produces)
- Coordinate system mismatches
- Invalid component IDs

## Pipeline Lifecycle

```python
pipeline = Pipeline(
    source=my_source,
    components=[comp_a, comp_b],
)

pipeline.validate()   # check for errors
pipeline.start()      # starts all threads
# ... pipeline runs ...
pipeline.stop()       # stops all threads
```

### What happens on start()

1. Validates if not already validated
2. Injects `_component_id` and `_event_bus` into each component
3. Calls `setup()` on each component
4. Wires EventBus subscriptions
5. Starts EventBus and ResultBus threads
6. Creates and starts Scheduler

### What happens on stop()

1. Emits `PipelineStateEvent("stopping")`
2. Stops Scheduler (waits for current frame)
3. Calls `teardown()` on each component in reverse order
4. Stops ResultBus and EventBus threads
5. Emits `PipelineStateEvent("stopped")`

## Accessing result_bus and event_bus

```python
pipeline = Pipeline(...)

# Subscribe to events
pipeline.event_bus.subscribe(ComponentErrorEvent, my_handler)

# Get results
pipeline.result_bus.subscribe(my_result_handler)

pipeline.start()
```

## Resetting the Pipeline

The `pipeline.reset()` method resets all component state without stopping the pipeline. This is useful for:

- **Session switching**: When switching between different detection classes or calibration data
- **Calibration updates**: After updating calibration parameters, reset to apply them
- **State recovery**: When a component's internal state becomes inconsistent

### How reset() Works

1. Pauses the Scheduler (waits for current frame to complete)
2. Calls `reset()` on each component in topological order
3. Resumes the Scheduler
4. Emits `PipelineStateEvent(state="reset")`

```python
pipeline = Pipeline(...)

# At runtime, trigger a reset (e.g., when calibration changes)
pipeline.reset()  # All components reset their internal state

# Pipeline continues running without interruption
```

### Thread Safety

- `reset()` is called from the main/API thread
- The Scheduler is paused before any component's `reset()` is called
- `process()` is guaranteed not to run concurrently with any `reset()` call
- `on_event()` may still be called during reset (EventBus continues running)
- If a component's `reset()` accesses state also accessed by `on_event()`, it must acquire `self._lock`

## Accessing Components

Use `pipeline.component(id)` to retrieve a component instance by its ID:

```python
pipeline = Pipeline(...)

# Access a specific component
embedder = pipeline.component("embedder")

# Call methods directly (outside of process() or on_event())
result = embedder.embed_images(images)
```

### When to Use component()

- **Dependency injection**: Pass component instances to other components during construction
- **Service methods**: Call non-process methods on components (e.g., `embed_images()`, `reload_model()`)
- **Testing**: Access component state for assertions

### Accessing Branch Components

The `component()` method also searches branch components:

```python
# Get a component from an exclusive branch
scan_proposer = pipeline.component("scan_proposer")
scan_proposer.reload_model("/new/model.onnx")
```

## Branches

Branches are conditional sub-pipelines that run only when a trigger condition is met. They allow the same pipeline to handle multiple modes (e.g., "scan" vs "track").

### Branch Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique branch identifier |
| `trigger` | string | Python expression evaluated against `frame.meta` |
| `inject_after` | string | Component ID after which branch starts |
| `merge_before` | string | Component ID before which branch merges |
| `components` | list | Components that run when trigger is true |

### How Branches Work

1. **inject_after**: The branch receives the Frame state after this component runs
2. **trigger**: A Python expression evaluated with `frame.meta` as local variables
3. **merge_before**: The branch's outputs are available to this component onward

### Example

```yaml
pipeline:
  source: camera_source
  components:
    - module: router
      id: router
    - module: tracker
      id: tracker
      
  branches:
    - id: scan_branch
      trigger: "routing_decision == 'scan'"
      inject_after: router
      merge_before: tracker
      components:
        - module: scan_proposer
          id: scan_proposer
```

When `frame.meta["routing_decision"] == "scan"`, the `scan_proposer` runs between `router` and `tracker`.

## Exclusive Branches

Exclusive branches are a structural if/else — when the trigger is true, the main-path components between `inject_after` and `merge_before` are **skipped entirely** and the branch components run instead.

### When to Use Exclusive Branches

- When you want to skip a segment of the pipeline based on a condition
- When adding guard clauses to every component becomes tedious
- When you want the routing logic visible in the YAML topology

### Exclusive Branch Fields

| Field | Type | Description |
|-------|------|-------------|
| `exclusive` | bool | If `true`, branch is exclusive (default: `false`) |

### Example

```yaml
pipeline:
  source: camera_source
  components:
    - module: preprocessor
      id: prep
    - module: heavy_detector
      id: heavy
    - module: tracker
      id: tracker
      
  branches:
    - id: fast_path
      trigger: "use_fast == True"
      inject_after: prep
      merge_before: tracker
      exclusive: true
      components:
        - module: lightweight_detector
          id: fast
```

When `frame.meta["use_fast"]` is `True`:
- `prep` runs
- `heavy` is **skipped**
- `fast` runs
- `tracker` runs

When `frame.meta["use_fast"]` is `False`:
- `prep` runs
- `heavy` runs
- `fast` is **skipped**
- `tracker` runs

### Empty Exclusive Branch

An exclusive branch with an empty `components` list skips the covered segment entirely:

```yaml
branches:
  - id: skip_heavy
    trigger: "skip_heavy_processing == True"
    inject_after: prep
    merge_before: tracker
    exclusive: true
    components: []  # nothing runs, just skip the segment
```

## → Next Steps

- [Observability](./observability.md) — Monitor pipeline health
- [API Reference](./api_reference.md) — Complete API docs
