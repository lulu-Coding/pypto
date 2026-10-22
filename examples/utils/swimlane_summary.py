# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Print a compact summary of a raw ``chip_swimlane_records.json`` file.

This is a post-process helper for the DFX output of a profile run; it does not
run a kernel. It reads the raw AICore task stream (not the merged Perfetto
trace) and reports the AICore makespan, per-core busy time, dispatch skew, and
core utilization. Use it to separate in-kernel idle time from runtime
orchestration overhead.

Produce a fresh profile (the workload is executed and checked, but not
benchmarked):

    SIMPLER_DFX=1 python examples/advanced/10_group_matmul_i8_nz.py -p a5 -d 0 --profile --planner pypto

The example prints ``compiled in ...; artifacts: <dir>``. The raw records are
under ``<dir>/dfx_outputs/chip_swimlane_records.json`` (the example also prints
the exact path). If the old artifacts are gone, rerun the command above.

Summarize the newest record under ``build_output`` (or ``outputs``):

    python examples/utils/swimlane_summary.py

Or pass a file / directory directly:

    python examples/utils/swimlane_summary.py build_output/gmm_nz_profile_pypto
    python examples/utils/swimlane_summary.py <dir>/dfx_outputs/chip_swimlane_records.json

The raw schema is documented in
``runtime/docs/dfx/chip-swimlane-profiling.md``: ``metadata.clock_freq_hz`` is
the cycle-to-microsecond factor (1e9 on A5), and each ``aicore_tasks`` row is
``[core_id, task_token_raw, reg_task_id, start_cycles, end_cycles]``.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _find_record(path: Path) -> Path | None:
    """Return *path* itself if it is a record, else the newest record under it."""
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = sorted(path.rglob("chip_swimlane_records.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _select_record(explicit: str | None) -> Path | None:
    if explicit:
        found = _find_record(Path(explicit))
        if found:
            return found
        print(f"no chip_swimlane_records.json under {explicit}", file=sys.stderr)
        return None

    candidates: list[Path] = []
    for root in ("build_output", "outputs"):
        root_path = Path(root)
        if root_path.exists():
            candidates.extend(root_path.rglob("chip_swimlane_records.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _is_aic(core_id: int, core_types: list[str]) -> bool:
    return not core_types or (0 <= core_id < len(core_types) and core_types[core_id] == "aic")


def summarize(path: Path) -> None:
    """Print a concise summary of the AICore task stream in *path*."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"failed to read {path}: {exc}", file=sys.stderr)
        return

    metadata = data.get("metadata") or {}
    clock_hz = metadata.get("clock_freq_hz") or 1_000_000_000
    core_types = list(metadata.get("core_types") or [])
    rows = data.get("aicore_tasks") or []

    print(f"record: {path}")
    print(f"clock_freq_hz: {clock_hz}  num_cores: {metadata.get('num_cores')}  core_types: {core_types}")
    print(f"aicore_tasks rows: {len(rows)}")

    if not rows:
        print("no aicore_tasks found; this does not look like the raw chip swimlane record")
        print(f"top-level keys: {sorted(data.keys())}")
        return

    aic_rows = [
        row
        for row in rows
        if isinstance(row, (list, tuple)) and len(row) >= 5 and _is_aic(int(row[0]), core_types)
    ]
    if not aic_rows:
        print("no AIC rows found (only AIV rows?)")
        if rows:
            print(f"first raw row: {rows[0]}")
        return

    # The raw record's documented prefix is
    # [core_id, task_token_raw, reg_task_id, start_cycles, end_cycles], but the
    # runtime may append extra fields. Print the first row and index the prefix
    # rather than unpacking a fixed width.
    print(f"first_aic_row: {aic_rows[0]}")
    if len(aic_rows[0]) != 5:
        print(f"note: raw rows carry {len(aic_rows[0])} columns; using the first five")

    starts = [row[3] for row in aic_rows]
    ends = [row[4] for row in aic_rows]
    first_start, last_end = min(starts), max(ends)
    makespan_us = (last_end - first_start) / clock_hz * 1e6
    start_spread_us = (max(starts) - first_start) / clock_hz * 1e6
    end_spread_us = (last_end - min(ends)) / clock_hz * 1e6

    busy_cycles: dict[int, int] = defaultdict(int)
    for row in aic_rows:
        core_id = int(row[0])
        start = row[3]
        end = row[4]
        busy_cycles[core_id] += end - start
    busy_us = sorted(cycles / clock_hz * 1e6 for cycles in busy_cycles.values())
    total_busy_us = sum(busy_us)
    active_cores = len(busy_cycles)
    utilization = total_busy_us / (makespan_us * active_cores) * 100 if makespan_us > 0 else 0.0

    print(f"aicore_makespan_us: {makespan_us:.2f}")
    print(f"start_spread_us: {start_spread_us:.2f}  end_spread_us: {end_spread_us:.2f}")
    print(
        f"per_core_busy_us: n={len(busy_us)} min={busy_us[0]:.2f} "
        f"median={statistics.median(busy_us):.2f} max={busy_us[-1]:.2f}"
    )
    print(f"sum_core_busy_us: {total_busy_us:.2f}  active_cores: {active_cores}")
    print(f"core_busy_utilization_pct: {utilization:.1f}")

    print("\nread:")
    print("- makespan is the AICore kernel span.")
    print("- compare it with the benchmark device_wall: similar means the gap is inside the kernel;")
    print("  much smaller means the rest is runtime/orchestration overhead.")
    print("- start_spread is dispatch skew; a large spread means blocks started in waves.")
    print("- core_busy_utilization_pct is busy/makespan; use it to size the in-kernel idle.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a raw chip_swimlane_records.json")
    parser.add_argument(
        "path",
        nargs="?",
        help="raw chip_swimlane_records.json, or a directory/root to search",
    )
    args = parser.parse_args()

    record = _select_record(args.path)
    if record is None:
        print("no chip_swimlane_records.json found", file=sys.stderr)
        print(
            "run a profile first:\n"
            "  SIMPLER_DFX=1 python examples/advanced/10_group_matmul_i8_nz.py "
            "-p a5 -d 0 --profile --planner pypto",
            file=sys.stderr,
        )
        raise SystemExit(1)
    summarize(record)


if __name__ == "__main__":
    main()
