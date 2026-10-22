"""Minimal flag test: does set_flag/wait_flag work AT ALL on A5?"""
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

# Test 1: simplest possible — set then wait on the SAME flag (should never block)
@pl.jit
def t1_set_then_wait(x: pl.Tensor, o: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [16, 16])
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=0)
        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=0)
        o = pl.store(t, [0, 0], o)
    return o

# Test 2: two different event IDs (slot 0 and slot 1)
@pl.jit
def t2_two_events(x: pl.Tensor, o: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [16, 16])
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=0)
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=1)
        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=0)
        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=1)
        o = pl.store(t, [0, 0], o)
    return o

# Test 3: set MTE1→M direction (simulating "data ready")
@pl.jit
def t3_mte1_to_m(x: pl.Tensor, o: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [16, 16])
        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=0)
        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=0)
        o = pl.store(t, [0, 0], o)
    return o

# Test 4: FIX→M direction
@pl.jit
def t4_fix_to_m(x: pl.Tensor, o: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [16, 16])
        pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, event_id=0)
        pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.M, event_id=0)
        o = pl.store(t, [0, 0], o)
    return o

# Test 5: extract + manual MTE1→M set + wait (the v2 pattern)
@pl.jit
def t5_extract_with_flag(x: pl.Tensor, o: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        # Set flag BEFORE extract (simulating "slot free")
        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, event_id=0)
        t = pl.load(x, [0, 0], [16, 16])
        # Simulate "data ready" — set manually after load
        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=0)
        # Wait for "data ready"
        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, event_id=0)
        o = pl.store(t, [0, 0], o)
    return o

if __name__ == "__main__":
    x = torch.randn(16, 16)
    o = torch.zeros(16, 16)

    tests = [
        ("T1: set M→MTE1 then wait M→MTE1", t1_set_then_wait),
        ("T2: two events M→MTE1", t2_two_events),
        ("T3: set MTE1→M then wait MTE1→M", t3_mte1_to_m),
        ("T4: set FIX→M then wait FIX→M", t4_fix_to_m),
        ("T5: load + manual MTE1→M set + wait", t5_extract_with_flag),
    ]

    for name, fn in tests:
        print(f"{name}...")
        try:
            o = torch.zeros(16, 16)
            fn(x, o, config=RunConfig(platform="a5"))
            print("  PASS")
        except Exception as e:
            print(f"  FAIL: {str(e)[:100]}")
