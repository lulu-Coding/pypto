/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/comm.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/lower_composite/lower_composite_builder.h"
#include "src/ir/transforms/lower_composite/lower_composite_common.h"
#include "src/ir/transforms/lower_composite/lower_composite_rules.h"

namespace pypto {
namespace ir {
namespace lower_composite {

// ============================================================================
// ``pld.tensor.all_to_all`` lowering rule
//
// Push-based symmetric all-to-all: every rank sends a distinct chunk to every
// other rank.  2-phase decomposition:
//
//   Phase 1 (push): for dest in 0..NR-1:
//       pld.tile.put(dst=target, peer=dest, src=input, stage,   // push row to peer
//                    dst_offsets=[my_rank, 0],
//                    src_offsets=[dest, 0],
//                    shape=[1, SIZE], atomic=None)
//
//   Phase 2 (barrier):
//       notify-all (AtomicAdd 1)
//       wait-all   (Ge generation)
//
//   Result: target (window-as-result).  After the barrier, target[src, :]
//           holds the chunk received from rank src.
//
// Input layout:  input[dest, :] = chunk destined for rank dest.
//
// Emits tile.create + pld.tile.put directly (the tensor-level pld.tensor.put
// has no codegen and ConvertTensorToTileOps runs before this pass — same
// reason broadcast/allgather emit pld.tile.get directly). The HCCL TPUT engine
// streams input[dest, :] through the shared VEC staging tile into the peer's
// window row [my_rank, 0], so a row larger than the staging tile is auto-chunked
// by pto-isa. The self-rank case (peer == my_rank) falls out of the same TPUT
// path via HCCL identity mapping (CommRemotePtr returns the local ptr), so no
// separate self-copy branch is needed.
// ============================================================================

ExprPtr LowerTensorAllToAllRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 3, span)
      << "pld.tensor.all_to_all rule expects 3 args (input, target, signal), got " << args.size();
  const auto& input = args[0];
  const auto& target = args[1];
  const auto& signal = args[2];

  auto input_type = As<TensorType>(input->GetType());
  INTERNAL_CHECK_SPAN(input_type, span)
      << "pld.tensor.all_to_all input must be TensorType, got " << input->GetType()->TypeName();
  auto target_type = As<DistributedTensorType>(target->GetType());
  INTERNAL_CHECK_SPAN(target_type, span)
      << "pld.tensor.all_to_all target must be DistributedTensorType (deducer-rejected otherwise)";
  INTERNAL_CHECK_SPAN(target_type->shape_.size() == 2, span)
      << "pld.tensor.all_to_all target must be 2D [NR, SIZE]";
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  ValidateMeshSignalShape(signal_type, "pld.tensor.all_to_all", span);

  auto& reg = OpRegistry::GetInstance();
  auto comm = b.EmitCommSetup(target, span);

  // Per-chunk shape: [1, SIZE] where SIZE = target.shape[1].
  auto size_expr = target_type->shape_[1];
  auto chunk_shape = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{std::make_shared<ConstInt>(1, DataType::INDEX, span), size_expr}, span);

  auto zero_idx = std::make_shared<ConstInt>(0, DataType::INDEX, span);
  auto one_idx = std::make_shared<ConstInt>(1, DataType::INDEX, span);

  // Offsets for the push target: write at [my_rank, 0] on the peer's window.
  // Every rank r writes its per-destination chunk to slot [r, 0] on every
  // peer's window, so after the barrier, rank r sees target[src, :] = chunk
  // sent from src to r.
  auto my_rank_offsets = std::make_shared<MakeTuple>(
      std::vector<ExprPtr>{comm.my_rank, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

  // ---- Phase 1: push — write each per-destination row directly into the
  //      peer's window via pld.tile.put (TPUT-based). The HCCL TPUT engine
  //      streams input[dest, :] through the shared VEC staging tile, so a row
  //      larger than the stage is auto-chunked. The self-rank case (peer ==
  //      my_rank) falls out of the same path via HCCL identity mapping.
  //
  // One shared VEC staging tile is reused across all destinations, mirroring
  // allgather's. It is capped to one chunk rather than sized from SIZE: the
  // stage is a bounce buffer the transfer slides through, so a [1, SIZE] stage
  // would only waste UB. chunk_shape stays the transfer extent.
  const auto chunk_geometry = MakeChunkGeometry(target_type->dtype_, span, "pld.tensor.all_to_all");
  auto stage_shape =
      MakeCollectiveStageShape({one_idx, size_expr}, chunk_geometry, span, "pld.tensor.all_to_all");
  auto put_stage =
      b.Bind("aa_stage",
             reg.Create("tile.create", {stage_shape},
                        {{"dtype", target_type->dtype_}, {"target_memory", MemorySpace::Vec}}, span),
             span);

  b.EmitFor(
      "dest", zero_idx, comm.nranks_idx, one_idx,
      [&](LoweringBuilder& body, const VarPtr& dest_var) {
        auto dest_row_offsets = std::make_shared<MakeTuple>(
            std::vector<ExprPtr>{dest_var, std::make_shared<ConstInt>(0, DataType::INDEX, span)}, span);

        // pld.tile.put(dst, peer, src, stage, dst_offsets, src_offsets, shape):
        // read input[dest, :] and write it to the peer's window row [my_rank, 0].
        body.Bind(
            "aa_put",
            reg.Create("pld.tile.put",
                       {target, dest_var, input, put_stage, my_rank_offsets, dest_row_offsets, chunk_shape},
                       {{"atomic", static_cast<int>(AtomicType::kNone)}}, span),
            span);
      },
      span);

  // ---- Phase 2: barrier ----
  const int64_t generation = b.EmitBarrier(signal, comm, "", span);

  // Self-clearing epilogue: exactly one credit per peer this call.
  auto total_i32 = std::make_shared<ConstInt>(generation, DataType::INT32, span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  // Window-as-result: target[src, :] now holds the chunk from rank src.
  // No read-back phase or post-barrier needed — the barrier guarantees all
  // peer writes are complete, and no peer reads the window afterwards.
  return target;
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
