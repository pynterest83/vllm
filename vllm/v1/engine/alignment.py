# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level alignment observations before scheduler output is committed."""

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request


def iteration_request_details(
    scheduled: "SchedulerOutput", output: "ModelRunnerOutput | None", *,
    requests: "Mapping[str, Request]",
    request_ids_randomized: bool = False,
) -> dict[str, Any]:
    """Describe scheduled shapes and sampled output without changing requests.

    Output counts precede scheduler stop/length truncation. They describe the
    verification result, not the final client-visible completion count.
    Read committed progress at output consumption: cached scheduler counts can
    include placeholders for earlier async batches. An already removed request
    has no remaining progress observation, but its executed work is retained.
    """
    prefills = [
        [request.num_computed_tokens, scheduled.num_scheduled_tokens[request.req_id]]
        for request in scheduled.scheduled_new_reqs
        if scheduled.num_scheduled_tokens.get(request.req_id, 0) > 0
    ]
    cached = scheduled.scheduled_cached_reqs
    progress = []
    for index, request_id in enumerate(cached.req_ids):
        query_len = scheduled.num_scheduled_tokens.get(request_id, 0)
        if query_len <= 0:
            continue
        kv_len = cached.num_computed_tokens[index]
        if cached.is_context_phase(request_id):
            prefills.append([kv_len, query_len])
            continue
        if output is None:
            raise ValueError("alignment decode observation requires model output")
        output_index = output.req_id_to_index[request_id]
        external_id = request_id
        if request_ids_randomized:
            external_id, separator, suffix = request_id.rpartition("-")
            if (
                not separator
                or not external_id
                or re.fullmatch(r"[0-9a-f]{8}", suffix) is None
            ):
                raise ValueError(f"invalid randomized request id: {request_id!r}")
        emitted = len(output.sampled_token_ids[output_index])
        drafted = len(scheduled.scheduled_spec_decode_tokens.get(request_id, ()))
        accepted = max(emitted - 1, 0) if drafted else 0
        request = requests.get(request_id)
        if accepted > drafted or (not drafted and emitted > 1):
            raise ValueError(
                f"invalid alignment decode output for {request_id!r}: "
                f"drafted={drafted}, emitted={emitted}"
            )
        progress.append(
            {
                "engine_request_id": request_id,
                "external_request_id": external_id,
                "kv_len": kv_len,
                "query_len": query_len,
                "output_tokens_before": (
                    request.num_output_tokens if request is not None else None
                ),
                "request_finished_before": request is None or request.is_finished(),
                "drafted_tokens": drafted,
                "accepted_draft_tokens": accepted,
                "emitted_tokens": emitted,
            }
        )
    return {
        "prefill_chunk_pairs": prefills,
        "decode_kv_lens": [request["kv_len"] for request in progress],
        "decode_query_lens": [request["query_len"] for request in progress],
        "decode_request_progress": progress,
    }
