# benchmarks/slot_validation_overhead.py
"""
Measure slot validation overhead.

Run with: python benchmarks/slot_validation_overhead.py

Expected output shows overhead per slot write in microseconds.
Target: <0.5 µs per write.
"""

from __future__ import annotations

import timeit
import sys

ITERATIONS = 100000

print("Slot Validation Overhead Benchmark")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Iterations: {ITERATIONS:,}")
print()

# Check if torch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: torch not available, skipping tensor benchmarks")
    print()

# Baseline dict operations
print("1. Baseline dict operations")
print("-" * 40)

setup_dict = "d = {}"
time_dict_set = timeit.timeit('d["x"] = 1', setup=setup_dict, number=ITERATIONS)
print(f"   dict set:        {time_dict_set * 1e6 / ITERATIONS:.3f} µs")

setup_dict_get = 'd = {"x": 1}'
time_dict_get = timeit.timeit('d["x"]', setup=setup_dict_get, number=ITERATIONS)
print(f"   dict get:        {time_dict_get * 1e6 / ITERATIONS:.3f} µs")

setup_dict_check = 'd = {"x": 1}'
time_dict_check = timeit.timeit('"x" in d', setup=setup_dict_check, number=ITERATIONS)
print(f"   dict contains:   {time_dict_check * 1e6 / ITERATIONS:.3f} µs")

print()

if TORCH_AVAILABLE:
    # Individual checks
    print("2. Individual tensor checks")
    print("-" * 40)

    setup_tensor = "import torch; t = torch.randn(100, 4, dtype=torch.float32)"

    time_isinstance = timeit.timeit(
        "isinstance(t, torch.Tensor)", setup=setup_tensor, number=ITERATIONS
    )
    print(f"   isinstance:      {time_isinstance * 1e6 / ITERATIONS:.3f} µs")

    time_dtype = timeit.timeit(
        "t.dtype == torch.float32", setup=setup_tensor, number=ITERATIONS
    )
    print(f"   dtype check:     {time_dtype * 1e6 / ITERATIONS:.3f} µs")

    time_shape_rank = timeit.timeit(
        "len(t.shape) == 2", setup=setup_tensor, number=ITERATIONS
    )
    print(f"   shape rank:      {time_shape_rank * 1e6 / ITERATIONS:.3f} µs")

    time_shape_dim = timeit.timeit(
        "t.shape[1] == 4", setup=setup_tensor, number=ITERATIONS
    )
    print(f"   shape dim:       {time_shape_dim * 1e6 / ITERATIONS:.3f} µs")

    time_device = timeit.timeit("not t.is_cuda", setup=setup_tensor, number=ITERATIONS)
    print(f"   device check:    {time_device * 1e6 / ITERATIONS:.3f} µs")

    print()

    # Full validation
    print("3. Full validation chain")
    print("-" * 40)

    setup_full = """
import torch
t = torch.randn(100, 4, dtype=torch.float32)
"""
    stmt_full = """
isinstance(t, torch.Tensor) and \
t.dtype == torch.float32 and \
len(t.shape) == 2 and \
t.shape[1] == 4 and \
not t.is_cuda
"""
    time_full = timeit.timeit(stmt_full, setup=setup_full, number=ITERATIONS)
    print(f"   full validation: {time_full * 1e6 / ITERATIONS:.3f} µs")

    print()

    # Simulated slot write with validation
    print("4. Slot write simulation")
    print("-" * 40)

    setup_sim = """
import torch
d = {}
schemas = {"boxes": True}
t = torch.randn(100, 4, dtype=torch.float32)
"""
    stmt_sim_off = """
if "boxes" not in schemas:
    d["boxes"] = t
else:
    d["boxes"] = t
"""
    time_sim_off = timeit.timeit(stmt_sim_off, setup=setup_sim, number=ITERATIONS)
    print(f"   validation off:  {time_sim_off * 1e6 / ITERATIONS:.3f} µs")

    stmt_sim_warn = """
if "boxes" not in schemas:
    d["boxes"] = t
else:
    if not isinstance(t, torch.Tensor):
        pass
    elif t.dtype != torch.float32:
        pass
    elif len(t.shape) != 2 or t.shape[1] != 4:
        pass
    else:
        d["boxes"] = t
"""
    time_sim_warn = timeit.timeit(stmt_sim_warn, setup=setup_sim, number=ITERATIONS)
    print(f"   validation warn: {time_sim_warn * 1e6 / ITERATIONS:.3f} µs")

    print()

# Summary
print("=" * 60)
print("SUMMARY")
print("-" * 40)

if TORCH_AVAILABLE:
    baseline = time_dict_set * 1e6 / ITERATIONS
    overhead_warn = time_sim_warn * 1e6 / ITERATIONS - baseline
    overhead_full = time_full * 1e6 / ITERATIONS

    print(f"Baseline dict set:     {baseline:.3f} µs")
    print(f"Write with validation: {time_sim_warn * 1e6 / ITERATIONS:.3f} µs")
    print(f"Overhead:              {overhead_warn:.3f} µs")
    print()
    print(f"Target:                <0.5 µs overhead")
    print(
        f"Status:                {'PASS' if overhead_warn < 0.5 else 'NEEDS OPTIMIZATION'}"
    )
else:
    print("Could not measure tensor validation (torch not available)")

print()
