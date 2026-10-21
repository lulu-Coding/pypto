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
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
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
// ``pld.tensor.barrier`` lowering rule
//
// Cross-rank barrier: notify-all (AtomicAdd 1) then wait-all (Ge generation).
// Pure synchronisation — no data movement.  Returns the signal expression so
// the rebind idiom (``sig = pld.tensor.barrier(sig)``) matches allreduce; the
// barrier restarts at generation 1 on every call (self-clearing credit protocol).
// ============================================================================

ExprPtr LowerTensorBarrierRule(const CallPtr& call, const std::vector<ExprPtr>& args, LoweringBuilder& b) {
  const Span& span = call->span_;
  INTERNAL_CHECK_SPAN(args.size() == 1, span) << "pld.tensor.barrier rule expects 1 arg, got " << args.size();
  const auto& signal = args[0];
  auto signal_type = As<DistributedTensorType>(signal->GetType());
  INTERNAL_CHECK_SPAN(signal_type, span)
      << "pld.tensor.barrier signal must be DistributedTensorType (deducer-rejected otherwise)";
  ValidateMeshSignalShape(signal_type, "pld.tensor.barrier", span);

  auto comm = b.EmitCommSetup(signal, span);

  // ---- AtomicAdd cell[my_rank, 0] on each peer, then wait cell[src, 0] >= gen ----
  const int64_t generation = b.EmitBarrier(signal, comm, "", span);

  // Self-clearing epilogue: exactly one credit per peer this call.
  auto total_i32 = std::make_shared<ConstInt>(generation, DataType::INT32, span);
  b.EmitEpilogueReset(signal, comm, total_i32, span);

  // Rebind: return the signal so the LHS Var retains the DistributedTensor view.
  return signal;
}

}  // namespace lower_composite
}  // namespace ir
}  // namespace pypto
