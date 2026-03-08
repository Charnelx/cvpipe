# cvpipe Documentation

cvpipe is a framework for building computer vision inference pipelines as directed
acyclic graphs (DAGs) of swappable components.

---

## Where to start

| | |
|---|---|
| **New here?** | [Concepts](./concepts.md) — the mental model behind everything |
| **Building your first component?** | [Building Components](./building_components.md) |
| **Configuring a pipeline in YAML?** | [Building Pipelines](./building_pipelines.md) |
| **Want a complete working example?** | [End-to-End Example](./example_webcam_api.md) |
| **Adding monitoring?** | [Observability](./observability.md) |
| **Looking up a specific class?** | [API Reference](./api_reference.md) |

---

## Two-minute overview

A pipeline reads frames from a **source**, passes each through a sequence of
**components**, and emits results on a **ResultBus**:

```
WebcamSource --> [Preprocessor] --> [Detector] --> [Tracker] --> [ResultAssembler]
                                                                        |
                                                            ResultBus -> WebSocket
```

Each component declares what tensor slots it reads (`INPUTS`) and writes (`OUTPUTS`).
cvpipe validates the whole graph before any model loads.

**Assemble from YAML with one call:**

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

**Branches** let you conditionally skip or swap segments of the pipeline:

```yaml
branches:
  - id: fast_mode
    trigger: "mode == 'fast'"
    inject_after: preprocessor
    merge_before: tracker
    exclusive: true
    components:
      - module: lightweight_detector
        id: fast_det
```

When `frame.meta["mode"] == "fast"`, the lightweight detector runs and the main-path
components between `preprocessor` and `tracker` are skipped. No guard clauses anywhere.

---

## Install

```bash
pip install git+https://github.com/Charnelx/cvpipe.git
```

Requirements: Python 3.11+. PyTorch optional (required for GPU tensor slots).
