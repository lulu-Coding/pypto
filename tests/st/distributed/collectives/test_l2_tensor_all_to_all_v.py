# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L2 distributed ST: managed ``pld.tensor.all_to_all_v`` inside a CHIP pipeline.

The CHIP/L2 counterpart of ``test_l3_host_tensor_all_to_all_v.py``. There, the
collective is written in the HOST orchestrator and ``LowerHostTensorCollectives``
fans it out into one ``builtin.tensor.all_to_all_v`` *chip dispatch per device* —
an extra L2 orchestration task whose only job is to submit one AIV kernel. With
a stage and a consume step around it, a rank costs three L3 -> L2 round trips.

Here the same call is written one level down, in the CHIP pipeline itself::

    @pl.jit
    def chip_pipeline(...):
        stage, counts = stage_step(...)
        data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=1)
        return consume_step(data, recv, out, recv_out)

``LowerL2TensorCollectives`` rewrites the collective into a call to a synthesized
AIV kernel backed by the *same* hand-written builtin source the HOST rail
renders, so the wire behaviour is unchanged and only the dispatch structure
differs: one ``chip_pipeline`` dispatch per rank, three AIV tasks inside it,
ordered by ordinary TensorMap dependencies. That the two rails really do render
one byte-identical kernel source is a compile-only property, asserted in
``tests/ut/codegen/distributed/test_builtin_collective_kernel_source.py``.

The exchange uses five window-bound resources, all allocated in one comm-domain
scope: ``stage`` (TPUT source only), ``data`` (the result window), ``signal``
(credit-based barrier), ``counts`` (this rank's send counts — peers pull ONE
scalar word per rank from this window after Barrier A, so the ``[NR]`` INT32
vector is all it needs) and ``recv`` (per-source valid row counts, written by
the collective).

**Why this file used to be ``@pl.program``**: while issue #2638 was open,
``@pl.jit``'s specializer dropped local tensor metadata across a two-level
collective rebinding — ``data = pld.tensor.all_to_all_v(...)`` followed by
``consume_step(data, ...)`` failed with "missing inferred tensor metadata for
parameter 'data'". The specializer was generalized (#2901; #2638 closed
2026-10-10, pinned by ``tests/ut/jit/test_decorator.py``), so this file now
uses the user-facing ``@pl.jit`` surface — the family the rule
(``.claude/rules/testing-and-examples.md``) requires for ``tests/st/``.

ST coverage: P=2 and P=4 (skips when fewer devices are available) — the
uniform golden, plus the same 0 / 1 / capacity / over-capacity / negative
count matrix the InCore and HOST rails run, so all three are held to one
wire golden.

Run on hardware via ``task-submit``::

    task-submit --device auto --device-num 4 --run 'cd <repo> && \
        export PYTHONPATH=<repo>/python:$PYTHONPATH && \
        python -m pytest tests/st/distributed/collectives/test_l2_tensor_all_to_all_v.py \
        -v --platform a2a3 --device $TASK_DEVICE'
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import DataType
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig

SIZE = 64
MAX_RECV = 4


def _admitted_blocks(n_ranks: int, core_num: int) -> int:
    """``B = CalAllToAllVBlocks(P, L)``: ``L`` below ``P``, else the largest multiple of ``P`` <= ``L``.

    An independent restatement, as in the HOST multicore ST, so a case's expected
    gang width does not come from the code under test.
    """
    return core_num if core_num < n_ranks else (core_num // n_ranks) * n_ranks


# Requested block limits ``L``. With P=2 these admit B = 1, 2, 4; with P=4, B = 1, 2, 4 — covering
# the single-block rail, B < P (multi-peer-per-core) and B = P / B > P (peer x length lanes).
_CORE_NUMS = (1, 2, 4)


def _build_l2_all_to_all_v_program(
    n_ranks: int, max_recv: int, core_num: int = 1, *, payload_dtype: DataType = pl.FP32
):
    """Build an N-rank CHIP-pipeline variable-size all-to-all program.

    Signal/counts shapes are per-rank-count, so the program is built by a
    factory — the same pattern the HOST and InCore all_to_all_v STs use.
    """
    nr = n_ranks
    mr = max_recv
    total = nr * mr
    # The signal is [NR, S] with one lane per admitted block, so S = B (the
    # smallest legal stride: the pass rejects S < B).
    stride = _admitted_blocks(nr, core_num)

    @pl.jit.incore
    def stage_step(
        inp: pl.Tensor[[total, SIZE], payload_dtype],
        counts_row: pl.Tensor[[nr, 1], pl.INT32],
        stage: pl.InOut[pld.DistributedTensor[[total, SIZE], payload_dtype]],
        counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> tuple[
        pld.DistributedTensor[[total, SIZE], payload_dtype],
        pld.DistributedTensor[[nr, 1], pl.INT32],
    ]:
        """Publish this rank's payload and send counts into their windows.

        Both are window-bound because every operand of one collective must
        live in the same comm domain — the same narrowing the HOST rail
        imposes, which is why the counts need a staging step at all.
        """
        for row in pl.range(total):
            chunk = pl.load(inp, [row, 0], [1, SIZE])
            stage = pl.store(chunk, [row, 0], stage)
        # Scalar read/write — a [1,1] INT32 tile.load fails ptoas 32-byte
        # row alignment (4 bytes), the pitfall the other all_to_all_v STs
        # avoid the same way.
        for d in pl.range(nr):
            v = pl.read(counts_row, [d, 0])
            pl.write(counts, [d, 0], v)
        return stage, counts

    @pl.jit.incore
    def consume_step(
        data: pl.InOut[pld.DistributedTensor[[total, SIZE], payload_dtype]],
        # Written by the collective (pulled from each source's send_counts
        # window); recv_counts[src, 0] is the count the consumer reads.
        recv_counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        out: pl.Out[pl.Tensor[[total, SIZE], payload_dtype]],
        recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[total, SIZE], payload_dtype], pl.Tensor[[nr, 1], pl.INT32]]:
        """Read back each source's block, bounded by its published count.

        Both loops target ``out`` and their row ranges partition each
        sender's slot, so every row is written exactly once: a ``pl.Out``
        tensor is write-only on the device, so a row the kernel skipped
        would come back as undefined host memory rather than as window
        content. Keeping the valid-row loop bounded by ``recv_counts``
        exercises the intended consumer pattern; the tail loop is what makes
        a bounded-transfer regression visible.
        """
        for src in pl.range(nr):
            n_rows_i32 = pl.read(recv_counts, [src, 0])
            pl.write(recv_out, [src, 0], n_rows_i32)
            n_rows = pl.cast(n_rows_i32, pl.INDEX)
            base = src * mr
            for r in pl.range(n_rows):
                flat_row = base + r
                chunk = pl.load(data, [flat_row, 0], [1, SIZE])
                out = pl.store(chunk, [flat_row, 0], out)
            for r in pl.range(n_rows, mr):
                flat_row = base + r
                chunk = pl.load(data, [flat_row, 0], [1, SIZE])
                out = pl.store(chunk, [flat_row, 0], out)
        return out, recv_out

    @pl.jit
    def chip_pipeline(
        inp: pl.Tensor[[total, SIZE], payload_dtype],
        counts_row: pl.Tensor[[nr, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[total, SIZE], payload_dtype]],
        recv_out: pl.Out[pl.Tensor[[nr, 1], pl.INT32]],
        stage: pl.InOut[pld.DistributedTensor[[total, SIZE], payload_dtype]],
        data: pl.InOut[pld.DistributedTensor[[total, SIZE], payload_dtype]],
        signal: pl.InOut[pld.DistributedTensor[[nr, stride], pl.INT32]],
        counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        recv: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[total, SIZE], payload_dtype], pl.Tensor[[nr, 1], pl.INT32]]:
        """One per-rank pipeline: stage -> collective -> consume.

        The three tasks are ordered by real TensorMap dependencies —
        ``stage`` and ``counts`` are written then read, ``data`` and
        ``recv`` are written by the collective then read by the consumer —
        not by an injected ordering token.
        """
        stage, counts = stage_step(inp, counts_row, stage, counts)
        data = pld.tensor.all_to_all_v(stage, data, signal, counts, recv, core_num=core_num)
        return consume_step(data, recv, out, recv_out)

    @pl.jit.host
    def host_orch(
        inputs: pl.Tensor[[nr, total, SIZE], payload_dtype],
        send_counts: pl.Tensor[[nr, nr, 1], pl.INT32],
        outputs: pl.Out[pl.Tensor[[nr, total, SIZE], payload_dtype]],
        recv_outputs: pl.Out[pl.Tensor[[nr, nr, 1], pl.INT32]],
    ) -> tuple[pl.Tensor[[nr, total, SIZE], payload_dtype], pl.Tensor[[nr, nr, 1], pl.INT32]]:
        """Allocate the five windows once, dispatch one pipeline per rank."""
        stage_buf = pld.alloc_window_buffer(total * SIZE * payload_dtype.get_byte())
        data_buf = pld.alloc_window_buffer(total * SIZE * payload_dtype.get_byte())
        signal_buf = pld.alloc_window_buffer(nr * stride * pl.INT32.get_byte())
        # Peers pull ONE word per rank from this window (scalar ld_dev read),
        # so the [NR, 1] INT32 vector is the whole requirement — no fixed-
        # width TLOAD unit set.
        counts_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())
        recv_buf = pld.alloc_window_buffer(nr * pl.INT32.get_byte())

        for r in pl.range(pld.world_size()):
            stage = pld.window(stage_buf, [total, SIZE], dtype=payload_dtype)
            data = pld.window(data_buf, [total, SIZE], dtype=payload_dtype)
            sig = pld.window(signal_buf, [nr, stride], dtype=pl.INT32)
            counts = pld.window(counts_buf, [nr, 1], dtype=pl.INT32)
            recv = pld.window(recv_buf, [nr, 1], dtype=pl.INT32)
            chip_pipeline(
                inputs[r],
                send_counts[r],
                outputs[r],
                recv_outputs[r],
                stage,
                data,
                sig,
                counts,
                recv,
                device=r,
            )
        return outputs, recv_outputs

    return host_orch


def _golden_inputs(nr: int, mr: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank r sends ``nr - dest`` rows to each destination (variable counts).

    The value formula ``r*1000 + d*100 + k*10 + j%10`` matches the HOST and
    InCore all_to_all_v STs, so a golden mismatch points at the rail under test
    rather than at a different fixture.
    """
    total = nr * mr
    inputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
    send_counts = torch.zeros((nr, nr, 1), dtype=torch.int32)
    for r in range(nr):
        for d in range(nr):
            n_rows = nr - d
            send_counts[r, d, 0] = n_rows
            base = d * mr
            for k in range(n_rows):
                for j in range(SIZE):
                    inputs[r, base + k, j] = float(r * 1000 + d * 100 + k * 10 + j % 10)
    return inputs, send_counts


class TestL2TensorAllToAllV:
    """L2 managed all_to_all_v: one chip pipeline per rank, no nested dispatch."""

    @pytest.mark.parametrize("core_num", _CORE_NUMS)
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_l2_all_to_all_v(self, test_config, device_ids, n_ranks, core_num):
        """Compile and run the CHIP-pipeline all_to_all_v for P in {2, 4}, L in {1, 2, 4}.

        ``L > 1`` must produce results identical to ``L = 1``: the golden below
        does not depend on ``core_num``.
        """
        if len(device_ids) < n_ranks:
            pytest.skip(f"L2 all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr

        inputs, send_counts = _golden_inputs(nr, mr)
        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)

        program = _build_l2_all_to_all_v_program(nr, mr, core_num)
        compiled = program.compile(
            inputs,
            send_counts,
            outputs,
            recv_outputs,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(
                    device_ids=device_ids[:nr],
                    num_sub_workers=0,
                ),
            ),
        )

        # The collective is a kernel of the pipeline's own sub-build, not a
        # separate chip callable: a ``next_levels/builtin.tensor.*`` directory
        # would mean the HOST fan-out rail ran and the L2 dispatch is nested.
        next_levels = compiled.output_dir / "next_levels"
        assert (next_levels / "chip_pipeline").is_dir(), sorted(p.name for p in next_levels.iterdir())
        assert not any(p.name.startswith("builtin.tensor.") for p in next_levels.iterdir()), (
            "managed L2 all_to_all_v must not emit a builtin chip dispatch"
        )
        kernel_src = next_levels / "chip_pipeline" / "kernels" / "aiv" / "__builtin_all_to_all_v__fp32.cpp"
        assert kernel_src.is_file(), f"expected the rendered builtin kernel at {kernel_src}"

        compiled(inputs, send_counts, outputs, recv_outputs, config=RunConfig(platform=test_config.platform))

        # Rank `rank` receives from `src` the chunk that `src` sent to dest=rank.
        for rank in range(nr):
            for src in range(nr):
                n_rows = int(send_counts[src, rank, 0].item())
                assert int(recv_outputs[rank, src, 0].item()) == n_rows, (
                    f"P={nr} rank={rank} src={src}: recv_counts="
                    f"{int(recv_outputs[rank, src, 0].item())} != expected {n_rows}"
                )
                base = src * mr
                for k in range(n_rows):
                    expected_row = inputs[src, rank * mr + k, :]
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, expected_row, atol=1e-5), (
                        f"P={nr} rank={rank} src={src} row={k}: "
                        f"max diff = {(got_row - expected_row).abs().max().item()}"
                    )


def _effective_rows(count: int, max_recv: int) -> int:
    """Rows the kernel actually transfers and publishes: ``clamp(count, 0, MAX_RECV)``.

    All three rails apply the identical two-sided clamp, so this is the single
    golden for any lowering path.
    """
    return max(0, min(count, max_recv))


# Deliberately duplicated from the InCore ST
# (``test_l3_tensor_all_to_all_v_intrinsic.py``) and the HOST ST
# (``../test_l3_host_tensor_all_to_all_v.py``) rather than imported: the point
# is to drive ALL THREE lowering rails independently with the same counts
# against the same golden. If a rail ever diverges on the wire, its own file
# fails. Keep the three case tables in sync.
_SKEW_CASES = {
    "zero_and_full": lambda nr, mr: [[0 if d % 2 == 0 else mr for d in range(nr)] for _ in range(nr)],
    "one_and_full": lambda nr, mr: [[1 if d % 2 == 0 else mr for d in range(nr)] for _ in range(nr)],
    "over_capacity": lambda nr, mr: [[mr + 3 for _ in range(nr)] for _ in range(nr)],
    "negative": lambda nr, mr: [[-2 if d % 2 == 0 else mr for d in range(nr)] for _ in range(nr)],
}


class TestL2TensorAllToAllVSkew:
    """CHIP-rail boundary coverage: 0, 1, capacity, over-capacity, negative counts.

    Mirrors the InCore and HOST cases — same counts, same golden — so a wire
    divergence between the managed CHIP rail and the other two surfaces as a
    golden mismatch in this file.

    This matters more here than the shared kernel source might suggest. The
    transport is byte-identical to the HOST rail's by construction (asserted in
    ``_assert_shared_kernel_source_with_host_rail``), but the *arguments* it
    receives are assembled by an entirely different path: this rail's operands
    come from ``chip_pipeline``'s own parameters through
    ``rt_submit_aiv_task``, not from a HOST dispatch's ``TaskArgs``. A wrong
    operand order, a lost InOut direction, or a mis-sized window would leave the
    kernel intact and still corrupt the exchange — which is exactly what these
    counts probe.
    """

    @pytest.mark.parametrize("case", sorted(_SKEW_CASES))
    @pytest.mark.parametrize("core_num", _CORE_NUMS)
    @pytest.mark.parametrize("n_ranks", [2, 4])
    def test_l2_all_to_all_v_skewed_counts(self, test_config, device_ids, n_ranks, core_num, case):
        if len(device_ids) < n_ranks:
            pytest.skip(f"L2 all_to_all_v P={n_ranks} needs {n_ranks} devices, got {device_ids}")

        nr = n_ranks
        mr = MAX_RECV
        total = nr * mr
        raw = _SKEW_CASES[case](nr, mr)

        # Fill the FULL capacity slot of every destination, not just the rows
        # being sent, so an over-send would deposit recognisable data in the
        # padding rows and the tail assertion below would catch it.
        #
        # ``salt`` makes every (case, n_ranks, core_num) payload unique. Window
        # memory is not zero-initialised on reuse AND persists across tests in
        # one process, so an unwritten row could otherwise still hold the
        # pattern an earlier test wrote — indistinguishable from a real
        # over-send. The core_num term keeps that true across the L
        # parametrizations, which reuse the same (case, n_ranks) slot layout;
        # the 900000 base additionally separates this file's salts from the
        # InCore and HOST files', which share the same formula.
        salt = 900000 + (sorted(_SKEW_CASES).index(case) + 1) * 100000 + nr * 10000 + core_num * 1000
        inputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        send_counts = torch.zeros((nr, nr, 1), dtype=torch.int32)
        for r in range(nr):
            for d in range(nr):
                send_counts[r, d, 0] = raw[r][d]
                base = d * mr
                for k in range(mr):
                    for j in range(SIZE):
                        inputs[r, base + k, j] = float(salt + r * 1000 + d * 100 + k * 10 + j % 10)

        outputs = torch.zeros((nr, total, SIZE), dtype=torch.float32)
        recv_outputs = torch.zeros((nr, nr, 1), dtype=torch.int32)

        program = _build_l2_all_to_all_v_program(nr, mr, core_num)
        compiled = program.compile(
            inputs,
            send_counts,
            outputs,
            recv_outputs,
            config=RunConfig(
                platform=test_config.platform,
                distributed_config=DistributedConfig(device_ids=device_ids[:nr], num_sub_workers=0),
            ),
        )
        compiled(inputs, send_counts, outputs, recv_outputs, config=RunConfig(platform=test_config.platform))

        for rank in range(nr):
            for src in range(nr):
                n_rows = _effective_rows(raw[src][rank], mr)

                # recv_counts publishes the clamped count, never the raw one.
                got_count = int(recv_outputs[rank, src, 0].item())
                assert got_count == n_rows, (
                    f"P={nr} case={case} rank={rank} src={src}: recv_counts={got_count} "
                    f"!= clamped({raw[src][rank]}) = {n_rows}"
                )

                base = src * mr
                for k in range(n_rows):
                    expected_row = inputs[src, rank * mr + k, :]
                    got_row = outputs[rank, base + k, :]
                    assert torch.allclose(got_row, expected_row, atol=1e-5), (
                        f"P={nr} case={case} rank={rank} src={src} row={k}: "
                        f"max diff = {(got_row - expected_row).abs().max().item()}"
                    )

                # The sender's surplus rows must never arrive. ``outputs``
                # mirrors the whole window (consume_step copies the tail rows
                # too), so this inspects window content rather than an unwritten
                # region of a write-only ``pl.Out`` buffer.
                #
                # Asserted as exactly zero: the runtime zeroes a comm-domain
                # window at allocation, before the handle is published to peers
                # (``aclrtMemset`` in ``comm_hccl.cpp``'s ``alloc_domain``), so
                # a row no TPUT ever wrote must still read 0. Every payload is
                # ``salt + ...`` with ``salt > 0``, so a transferred row can
                # never be mistaken for an untouched one.
                #
                # The self slot is included, as on both other rails: ``stage``
                # and ``data`` are distinct windows here, so the rank's own
                # staged input cannot masquerade as an arrival in ``data``.
                for k in range(n_rows, mr):
                    got_row = outputs[rank, base + k, :]
                    assert torch.all(got_row == 0.0), (
                        f"P={nr} case={case} rank={rank} src={src} row={k}: an untransferred row is "
                        f"not zero — the CHIP transfer is not bounded by the runtime count "
                        f"(got {got_row[:4].tolist()}...)"
                    )


@pytest.mark.parametrize("dtype_name", ["int8", "fp32"])
@pytest.mark.parametrize("core_num", [1, 4], ids=["multi-peer", "split-peer"])
@pytest.mark.parametrize(
    "peer_bytes",
    [8 * 1024, 64 * 1024, 192 * 1024 + 512],
    ids=["small", "tile-boundary", "pipeline-tail"],
)
def test_l2_all_to_all_v_staging_chunks(test_config, device_ids, dtype_name, core_num, peer_bytes):
    """Check byte-sized staging, ping/pong reuse, and partial final chunks."""
    nr = 2
    if len(device_ids) < nr:
        pytest.skip(f"Staging regression needs {nr} devices, got {device_ids}")
    payload_dtype, torch_dtype = {
        "int8": (pl.INT8, torch.int8),
        "fp32": (pl.FP32, torch.float32),
    }[dtype_name]
    mr = peer_bytes // (SIZE * payload_dtype.get_byte())
    total = nr * mr
    pattern = torch.arange(nr * total * SIZE, dtype=torch.int32).reshape(nr, total, SIZE)
    inputs = torch.zeros_like(pattern, dtype=torch_dtype).share_memory_()
    # Keep rank 0 -> rank 1 full to test an exact buffer boundary remotely;
    # asymmetric counts on the other paths exercise tails and count routing.
    counts = torch.tensor([[[mr - 1], [mr]], [[mr - 2], [mr - 3]]], dtype=torch.int32).share_memory_()
    outputs = torch.zeros_like(inputs).share_memory_()
    recv = torch.zeros_like(counts).share_memory_()
    program = _build_l2_all_to_all_v_program(nr, mr, core_num, payload_dtype=payload_dtype)
    config = RunConfig(
        platform=test_config.platform,
        distributed_config=DistributedConfig(device_ids=device_ids[:nr], num_sub_workers=0),
    )
    compiled = program.compile(inputs, counts, outputs, recv, config=config)
    # Large cases span multiple ping/pong turns plus a partial final chunk,
    # in both the multi-peer-per-core and the two-lanes-per-peer branches.
    # Different payloads on reused windows expose stale or undrained buffers.
    with compiled.prepare(config=config, persistent=True, reset_persistent_windows=False) as worker:
        for iteration in range(3):
            inputs.copy_(((pattern + iteration * 17) % 251 - 125).to(torch_dtype))
            expected = torch.zeros_like(outputs)
            for dest in range(nr):
                for src in range(nr):
                    rows = int(counts[src, dest, 0])
                    expected[dest, src * mr : src * mr + rows] = inputs[src, dest * mr : dest * mr + rows]
            worker.run(compiled, inputs, counts, outputs, recv, config=config)
            assert torch.equal(outputs, expected), (
                f"Payload or untouched tail differs at iteration {iteration}"
            )
            assert torch.equal(recv, counts.transpose(0, 1)), f"Counts differ at iteration {iteration}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
