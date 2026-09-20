# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from types import SimpleNamespace

import pytest

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats


class FakeEngineCore:
    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        return SchedulerStats(iteration_details=iteration_details)


def make_iteration_details() -> SchedulerIterationDetails:
    return SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=2,
        num_ctx_tokens=3,
        num_generation_requests=4,
        num_generation_tokens=5,
        elapsed_ms=6.7,
    )


def make_fake_engine(log_stats: bool = True, requests: dict | None = None):
    return SimpleNamespace(
        log_stats=log_stats,
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(
                enable_logging_iteration_details=True,
            )
        ),
        scheduler=SimpleNamespace(requests=requests or {}),
    )


def test_capture_iteration_details_disabled_without_log_stats():
    engine = make_fake_engine(log_stats=False)

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is None

    assert not hasattr(engine, "_iteration_index")


def test_capture_iteration_details_fills_elapsed_time():
    engine = make_fake_engine()

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is not None
        assert iteration_details.elapsed_ms == 0.0
        assert iteration_details.is_dummy
        time.sleep(0.001)

    assert iteration_details is not None
    assert iteration_details.elapsed_ms > 0.0
    assert engine._iteration_index == 1


def test_attach_iteration_details_uses_existing_output():
    iteration_details = make_iteration_details()
    outputs = {
        2: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
        1: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
    }

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert 0 not in outputs
    assert outputs[2].scheduler_stats is not None
    assert outputs[2].scheduler_stats.iteration_details == iteration_details
    assert outputs[1].scheduler_stats is not None
    assert outputs[1].scheduler_stats.iteration_details is None


def test_attach_iteration_details_falls_back_to_client_zero_without_outputs():
    iteration_details = make_iteration_details()
    outputs: dict[int, EngineCoreOutputs] = {}

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert set(outputs) == {0}
    assert outputs[0].scheduler_stats is not None
    assert outputs[0].scheduler_stats.iteration_details == iteration_details


def test_alignment_iteration_preserves_speculative_progress(monkeypatch):
    records = []
    monkeypatch.setattr(
        "vllm.v1.engine.core.logger.info", lambda *args: records.append(args)
    )
    # The scheduler's cached count can still hold an async placeholder, so the
    # committed count is read from the live request instead.
    engine = make_fake_engine(
        log_stats=False,
        requests={
            "decode-8e9ffed8": SimpleNamespace(
                num_output_tokens=9, is_finished=lambda: False
            )
        },
    )
    scheduled = SimpleNamespace(
        total_num_scheduled_tokens=8,
        scheduled_new_reqs=[],
        num_scheduled_tokens={"prefill": 2, "decode-8e9ffed8": 6},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["prefill", "decode-8e9ffed8"],
            num_computed_tokens=[16, 100],
            num_output_tokens=[0, 12],
            is_context_phase=lambda req_id: req_id == "prefill",
        ),
        scheduled_spec_decode_tokens={"decode-8e9ffed8": [1, 2, 3, 4, 5]},
        scheduled_encoder_input_stats=None,
    )
    EngineCore.assign_alignment_iteration_index(engine, scheduled)
    with EngineCore.log_iteration_details(engine, scheduled) as observation:
        observation["model_output"] = SimpleNamespace(
            req_id_to_index={"decode-8e9ffed8": 0}, sampled_token_ids=[[7, 8, 9]]
        )
    encoded = next(
        args[1] for args in records if args[0] == "VibeSimAlignmentIteration %s"
    )
    record = json.loads(encoded)
    assert record["schema_version"] == 4
    assert record["iteration_index"] == scheduled.alignment_iteration_index == 0
    assert record["prefill_chunk_pairs"] == [[16, 2]]
    assert record["decode_query_lens"] == [6]
    assert record["decode_request_progress"] == [
        {
            "engine_request_id": "decode-8e9ffed8",
            "external_request_id": "decode",
            "kv_len": 100,
            "query_len": 6,
            "output_tokens_before": 9,
            "request_finished_before": False,
            "drafted_tokens": 5,
            "accepted_draft_tokens": 2,
            "emitted_tokens": 3,
        }
    ]
    assert record["observed_end_monotonic_ns"] >= record["observed_start_monotonic_ns"]
    EngineCore.assign_alignment_iteration_index(engine, scheduled)
    assert scheduled.alignment_iteration_index == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_alignment_zero_token_step_does_not_emit_an_iteration(monkeypatch, enabled):
    records = []
    monkeypatch.setattr(
        "vllm.v1.engine.core.logger.info", lambda *args: records.append(args)
    )
    engine = make_fake_engine()
    engine.vllm_config.observability_config.enable_logging_iteration_details = enabled
    with EngineCore.log_iteration_details(
        engine, SimpleNamespace(total_num_scheduled_tokens=0)
    ) as observation:
        assert observation == {}
    assert records == []
