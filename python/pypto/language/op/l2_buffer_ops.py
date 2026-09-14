# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.l2.*`` — L2 cache prefetch via A5 SHMEM CMO.

Uses ``aclshmemx_cmo_qp_nbi`` on A5 (Ascend950) for device-side, multi-QP
concurrent L2 cache prefetch. Each AIV prefetches its own data block with
its own QP (``qp_idx = block_idx``). Data stays in DDR but becomes hot in
L2 cache, so subsequent ``tile.load`` operations hit L2 cache and achieve
2~4x bandwidth improvement over cache-miss access.

This is a mock for the future managed L2 SRAM buffer (``MemorySpace::L2Buffer``).
When real managed SRAM is available, these ops will be replaced by actual
L2Buffer load/store/move operations. The DSL API surface will remain the same.

Typical usage::

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, a, b, c):
        pl.l2.prefetch(a, size=M * K * 2)   # fp16 = 2 bytes
        pl.l2.prefetch(b, size=K * N * 2)
        self.kernel(a, b, c)                # kernel_entry injects CMO calls

Requirements:
    - A5 (Ascend950) device
    - CANN 9.1.0+ with SHMEM library
    - SDMA engine enabled in runtime config
"""

from typing import Any

from pypto.ir.op import tensor_ops as _ir_ops

from ..typing.scalar import Scalar
from ..typing.tensor import Tensor


def _unwrap(value: Any) -> Any:
    """Unwrap a DSL wrapper (Tensor / Scalar / ...) to ``ir.Expr``."""
    if hasattr(value, "unwrap"):
        return value.unwrap()
    return value


def prefetch(
    tensor: Tensor,
    offset: int | Scalar = 0,
    size: int | Scalar | None = None,
) -> None:
    """Annotate a GM tensor for L2 cache prefetch via SHMEM CMO.

    On A5, the backend injects ``aclshmemx_cmo_qp_nbi`` into the
    ``kernel_entry`` wrapper of the called InCore function. Each AIV
    prefetches its own block with its own QP. The prefetch completes
    before the kernel body starts.

    The data remains in DDR but is hot in L2 cache — subsequent
    ``tile.load`` operations hit L2 cache. This does not change any
    tensor values, so the computation is numerically identical with
    or without prefetch.

    Must be called inside an Orchestration function, before the
    InCore kernel call.

    Args:
        tensor: GM tensor to prefetch into L2 cache.
        offset: Byte offset within the tensor's device buffer.
        size: Bytes to prefetch. If ``None``, prefetches the entire
            tensor (computed from shape and dtype).
    """
    if size is None:
        size = _tensor_bytes(tensor)
    _ir_ops.annotate_prefetch(
        tensor.unwrap(),
        _unwrap(offset),
        _unwrap(size),
    )


def _tensor_bytes(tensor: Tensor) -> int:
    """Compute total bytes of a tensor from its shape and dtype."""
    total_elements = 1
    for dim in tensor.shape:
        total_elements *= dim
    return total_elements * tensor.dtype.get_byte()


__all__ = ["prefetch"]
