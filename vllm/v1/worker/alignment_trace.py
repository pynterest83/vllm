# SPDX-License-Identifier: Apache-2.0
"""Bulk alignment dumps: scheduled input tokens and routed-expert histograms.

The `VibeSimAlignment*` records the engine emits through the logger are one small
JSON object per iteration or per request. These two dumps are different in kind:
a single iteration's token IDs or per-layer expert histograms are far too large
for the log, and they are only wanted for a handful of hand-picked iterations.

So each is off unless its own path env var is set, and each takes an optional
iteration selector (`3`, `10-20`, `0,5,100-110`) that defaults to every
iteration. Output is one JSON object per line, appended under an exclusive lock
so tensor-parallel ranks writing the same path interleave whole rows.

Iteration indices are `SchedulerOutput.alignment_iteration_index`, the same
dispatch-order index the EngineCore records and the worker NVTX ranges use, so a
dump row joins to a `vllm_iteration(N)` NVTX range and to that iteration's
`VibeSimAlignmentIteration` record without any timestamp matching.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

TOKEN_TRACE_PATH_ENV = "VLLM_VIBESIM_TOKEN_TRACE_PATH"
TOKEN_TRACE_ITERS_ENV = "VLLM_VIBESIM_TOKEN_TRACE_ITERS"
ROUTING_TRACE_PATH_ENV = "VLLM_VIBESIM_ROUTING_TRACE_PATH"
ROUTING_TRACE_ITERS_ENV = "VLLM_VIBESIM_ROUTING_TRACE_ITERS"
ROUTING_TRACE_BLOCK_M_ENV = "VLLM_VIBESIM_ROUTING_TRACE_BLOCK_M"
ROUTING_TRACE_GLOBAL_COUNTS_ENV = "VLLM_VIBESIM_ROUTING_TRACE_GLOBAL_COUNTS"
ROUTING_TRACE_TOPK_IDS_ENV = "VLLM_VIBESIM_ROUTING_TRACE_TOPK_IDS"


def is_token_trace_enabled() -> bool:
    return bool(os.environ.get(TOKEN_TRACE_PATH_ENV))


def is_routing_trace_enabled() -> bool:
    return bool(os.environ.get(ROUTING_TRACE_PATH_ENV))


def parse_iterations(raw: str) -> set[int] | None:
    """Parse an iteration selector. `None` means "every iteration"."""
    raw = raw.strip()
    if not raw:
        return None

    selected: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            selected.update(range(int(start_text), int(end_text) + 1))
        else:
            selected.add(int(part))
    return selected


def _iteration_selected(iteration_index: int | None, iters_env: str) -> bool:
    selected = parse_iterations(os.environ.get(iters_env, ""))
    if selected is None:
        return True
    return iteration_index in selected


def should_trace_token_iteration(iteration_index: int | None) -> bool:
    if not is_token_trace_enabled():
        return False
    return _iteration_selected(iteration_index, TOKEN_TRACE_ITERS_ENV)


def should_trace_routing_iteration(iteration_index: int | None) -> bool:
    if not is_routing_trace_enabled():
        return False
    return _iteration_selected(iteration_index, ROUTING_TRACE_ITERS_ENV)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        # Ranks share one path; the lock is what keeps rows from interleaving.
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _tensor_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]


def _sequence_int_list(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]


def _device_provenance() -> dict[str, Any]:
    return {
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "local_cuda_device": (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
    }


def build_token_input_row(
    *,
    iteration_index: int | None,
    token_ids: list[int],
    request_ids: list[str],
    num_scheduled_tokens: list[int],
    positions: list[int] | None,
) -> dict[str, Any]:
    """Split a flat token buffer back into per-request spans.

    Kept free of torch and of the environment so the span arithmetic -- the only
    part that can silently mis-attribute tokens to the wrong request -- is
    directly testable.
    """
    requests: list[dict[str, Any]] = []
    offset = 0
    for request_id, count in zip(request_ids, num_scheduled_tokens):
        end = offset + count
        request_row: dict[str, Any] = {
            "req_id": request_id,
            "start": offset,
            "end": end,
            "num_scheduled_tokens": count,
            "token_ids": token_ids[offset:end],
        }
        if positions is not None:
            request_row["positions"] = positions[offset:end]
        requests.append(request_row)
        offset = end

    row: dict[str, Any] = {
        "schema_version": 1,
        "iteration": iteration_index,
        "tokens": len(token_ids),
        "num_reqs": len(request_ids),
        "req_ids": request_ids,
        "num_scheduled_tokens": num_scheduled_tokens,
        "token_ids_flat": token_ids,
        "requests": requests,
    }
    if positions is not None:
        row["positions_flat"] = positions
    return row


def dump_token_inputs(
    *,
    iteration_index: int | None,
    input_ids: torch.Tensor | None,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_tokens: int,
    positions: Any | None = None,
) -> None:
    """Dump the scheduled input token IDs for alignment-only reruns.

    This is what lets a replay feed the simulator the exact token stream vLLM
    saw, rather than a re-tokenization of the prompt text.
    """
    if not should_trace_token_iteration(iteration_index):
        return
    if input_ids is None or num_tokens <= 0:
        return
    trace_path = os.environ.get(TOKEN_TRACE_PATH_ENV)
    if not trace_path:
        return

    num_tokens = int(num_tokens)
    row = build_token_input_row(
        iteration_index=iteration_index,
        token_ids=_tensor_int_list(input_ids[:num_tokens]),
        request_ids=list(req_ids),
        num_scheduled_tokens=_sequence_int_list(num_scheduled_tokens),
        positions=(
            None if positions is None else _sequence_int_list(positions[:num_tokens])
        ),
    )
    row.update(_device_provenance())
    _append_jsonl(Path(trace_path), row)


def _iter_moe_layers(static_forward_context: dict[str, Any]):
    for layer_name, layer in static_forward_context.items():
        if not hasattr(layer, "global_num_experts"):
            continue
        if not hasattr(layer, "local_num_experts"):
            continue
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is None:
            continue
        yield int(layer_id), layer_name, layer


def grouped_gemm_block_stats(
    *,
    local_counts: list[int],
    total_assignments: int,
    global_num_experts: int,
    block_m: int,
) -> dict[str, Any]:
    """Padding cost of one grouped GEMM launch, given a per-expert load vector.

    The kernel sizes its launch for the worst case (`total_assignments` plus one
    short block per expert) while only `local_padded` rows carry work, so the
    ratio between them is how much of the launch is wasted on padding. Pure
    arithmetic, so the simulator's own grouped-GEMM cost model can be checked
    against it directly.
    """
    nonzero_counts = [count for count in local_counts if count > 0]
    local_padded = sum(
        int(math.ceil(count / block_m) * block_m) for count in nonzero_counts
    )
    sorted_token_ids_len = total_assignments + global_num_experts * (block_m - 1)
    launch_m_blocks = int(math.ceil(sorted_token_ids_len / block_m))
    effective_m_blocks = int(math.ceil(local_padded / block_m)) if local_padded else 0
    return {
        "block_m_assumed": block_m,
        "local_padded": local_padded,
        "sorted_token_ids_len": sorted_token_ids_len,
        "launch_m_blocks": launch_m_blocks,
        "effective_m_blocks": effective_m_blocks,
        "m_block_overlaunch": (
            launch_m_blocks / effective_m_blocks if effective_m_blocks else None
        ),
    }


def dump_routing_summary(
    *,
    capturer: Any,
    static_forward_context: dict[str, Any],
    iteration_index: int | None,
    num_tokens: int,
) -> None:
    """Dump per-layer routed expert histograms from the capturer device buffer.

    Complements the `VibeSimAlignmentExpertLoad` record, which reads EPLB's
    accumulated load and therefore needs EPLB enabled and only fires on its
    rearrangement steps. This one reads the raw `topk_ids` capture for one
    nominated iteration, and keeps the physical/local split and the expert map.
    """
    if not should_trace_routing_iteration(iteration_index):
        return
    if capturer is None:
        return
    device_buffer = getattr(capturer, "device_buffer", None)
    if device_buffer is None or num_tokens <= 0:
        return
    trace_path_raw = os.environ.get(ROUTING_TRACE_PATH_ENV)
    if not trace_path_raw:
        return
    trace_path = Path(trace_path_raw)

    num_tokens = min(int(num_tokens), int(device_buffer.shape[0]))
    block_m = int(os.environ.get(ROUTING_TRACE_BLOCK_M_ENV, "64"))
    include_global_counts = os.environ.get(ROUTING_TRACE_GLOBAL_COUNTS_ENV, "1") != "0"
    include_topk_ids = os.environ.get(ROUTING_TRACE_TOPK_IDS_ENV, "0") == "1"
    provenance = _device_provenance()

    for layer_id, layer_name, layer in _iter_moe_layers(static_forward_context):
        if layer_id >= int(device_buffer.shape[1]):
            continue
        topk_ids = device_buffer[:num_tokens, layer_id, :]
        if topk_ids.numel() == 0:
            continue

        global_num_experts = int(layer.global_num_experts)
        flat = topk_ids.reshape(-1)
        # A padded row is -1; it is not an assignment to expert 0.
        valid_mask = flat >= 0
        valid_flat = flat[valid_mask].to(torch.int64)
        global_counts = _tensor_int_list(
            torch.bincount(valid_flat, minlength=global_num_experts)[
                :global_num_experts
            ]
        )

        expert_map = layer.expert_map
        if expert_map is not None:
            expert_map_tensor = expert_map.to(device=flat.device)
            local_ids_full = expert_map_tensor[flat.clamp(min=0).to(torch.int64)]
            local_ids = local_ids_full[valid_mask & (local_ids_full >= 0)]
            local_num_experts = int(layer.local_num_experts)
            local_counts = _tensor_int_list(
                torch.bincount(local_ids.to(torch.int64), minlength=local_num_experts)[
                    :local_num_experts
                ]
            )
            local_global_experts = _tensor_int_list(
                torch.where(expert_map_tensor >= 0)[0]
            )
        else:
            local_counts = global_counts
            local_global_experts = list(range(global_num_experts))

        total_assignments = int(valid_flat.numel())
        row: dict[str, Any] = {
            "schema_version": 1,
            "iteration": iteration_index,
            "layer_id": layer_id,
            "layer_name": layer_name,
            "tokens": num_tokens,
            "top_k": int(topk_ids.shape[1]),
            "total_assignments": total_assignments,
            "global_num_experts": global_num_experts,
            "ep_size": int(getattr(layer, "ep_size", 1)),
            "ep_rank": int(getattr(layer, "ep_rank", 0)),
            "tp_size": int(getattr(layer, "tp_size", 1)),
            "tp_rank": int(getattr(layer, "tp_rank", 0)),
            "local_num_experts": int(layer.local_num_experts),
            "local_global_experts": local_global_experts,
            "local_counts": local_counts,
            "local_assignments": int(sum(local_counts)),
            "local_count_min": min(local_counts) if local_counts else 0,
            "local_count_p50": _percentile(local_counts, 0.5),
            "local_count_p90": _percentile(local_counts, 0.9),
            "local_count_max": max(local_counts) if local_counts else 0,
            "local_nonzero_experts": len([c for c in local_counts if c > 0]),
        }
        row.update(
            grouped_gemm_block_stats(
                local_counts=local_counts,
                total_assignments=total_assignments,
                global_num_experts=global_num_experts,
                block_m=block_m,
            )
        )
        row.update(provenance)
        if include_global_counts:
            row["global_counts"] = global_counts
        if include_topk_ids:
            # This is deliberately opt-in: one selected decode iteration is
            # small, while an unbounded prefill dump would be unnecessarily
            # large. Keep the token-major layout so the companion token trace's
            # per-request spans can recover within-request routing correlation.
            row["global_topk_ids"] = [
                [int(expert_id) for expert_id in token_experts]
                for token_experts in topk_ids.detach().cpu().tolist()
            ]

        _append_jsonl(trace_path, row)
