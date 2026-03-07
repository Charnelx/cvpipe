# cvpipe Documentation

cvpipe is a framework for building computer vision inference pipelines as directed acyclic graphs (DAGs) of swappable components.

## What cvpipe Is

cvpipe provides:
- A typed slot system for inter-component data contracts, validated at assembly time
- A mutable per-frame workspace (`Frame`) that components read and write in place
- A pub/sub `EventBus` for management signals (class registration, calibration updates, errors)
- A lossy ring-buffer `ResultBus` for high-frequency inference output
- A `Probe` system for non-breaking observability hooks
- YAML-driven pipeline topology with automatic connection inference
- Component auto-discovery from a directory of module packages

## What cvpipe Is NOT

- cvpipe has **zero knowledge** of any detector, model, camera, or application
- It is a generic framework — all domain logic lives in the application layer (`detector/`)
- cvpipe never imports from `detector/` or any application package

## Quick Start

```python
from cvpipe import Pipeline, FrameSource, Component, SlotSchema, ResultBus, EventBus

class SimpleSource(FrameSource):
    def __init__(self, n): 
        self.n = n
        self.count = 0
    
    def next(self):
        if self.count >= self.n:
            return None
        self.count += 1
        return (f"frame_{self.count}", 0.0)

class Producer(Component):
    OUTPUTS = [SlotSchema("data", int)]  # dtype=int goes to frame.meta, not slots
    def process(self, frame):
        frame.meta["data"] = 42

class Consumer(Component):
    INPUTS = [SlotSchema("data", int)]
    def process(self, frame):
        print(f"Received: {frame.meta['data']}")

# Run the pipeline
pipeline = Pipeline(
    source=SimpleSource(3),
    components=[Producer(), Consumer()],
)
pipeline.validate()
pipeline.start()
```

## Requirements

- Python 3.11+
- Optional: `torch` for GPU tensor slots

## Links

- [Concepts](./concepts.md) — Core concepts and architecture
- [Building Components](./building_components.md) — Component author guide
- [Building Pipelines](./building_pipelines.md) — Pipeline configuration
- [Observability](./observability.md) — Diagnostics and monitoring
- [API Reference](./api_reference.md) — Complete API documentation
