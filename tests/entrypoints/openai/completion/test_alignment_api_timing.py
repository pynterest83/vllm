import pytest

from vllm.entrypoints.serve.utils.api_utils import (
    build_alignment_api_timing_record,
)


def test_alignment_api_timing_uses_only_within_process_durations():
    record = build_alignment_api_timing_record(
        request_id="cmpl-vibesim_7",
        request_start_monotonic=100.0,
        generators_ready_monotonic=100.003,
        first_output_received_monotonic=100.020,
        last_output_received_monotonic=100.220,
        first_token_yield_monotonic=100.021,
        last_token_yield_monotonic=100.225,
        done_yield_monotonic=100.226,
        # Deliberately use a different absolute origin for EngineCore. Only
        # within-EngineCore differences are meaningful across processes.
        engine_queued_monotonic=10.0,
        engine_first_token_monotonic=10.015,
        engine_last_token_monotonic=10.215,
        generate_start_monotonic=100.005,
        add_request_done_monotonic=100.007,
        first_engine_output_received_monotonic=100.016,
        first_output_collector_put_monotonic=100.018,
        first_output_dequeued_monotonic=100.019,
        output_tokens=32,
        token_events=31,
        first_token_event_tokens=2,
    )

    assert record["schema_version"] == 3
    assert record["api_request_id"] == "cmpl-vibesim_7"
    assert record["api_frontend_prepare_ms"] == pytest.approx(3.0)
    assert record["api_first_output_wait_ms"] == pytest.approx(17.0)
    assert record["api_stream_activation_ms"] == pytest.approx(2.0)
    assert record["api_add_request_ms"] == pytest.approx(2.0)
    assert record["api_collector_wait_ms"] == pytest.approx(11.0)
    assert record["api_engine_output_wait_ms"] == pytest.approx(9.0)
    assert record["api_output_fanout_ms"] == pytest.approx(2.0)
    assert record["api_collector_wakeup_ms"] == pytest.approx(1.0)
    assert record["api_generator_resume_ms"] == pytest.approx(1.0)
    assert record["api_first_output_serialize_ms"] == pytest.approx(1.0)
    assert record["api_token_output_receive_span_ms"] == pytest.approx(200.0)
    assert record["api_token_sse_yield_span_ms"] == pytest.approx(204.0)
    assert record["api_terminal_tail_ms"] == pytest.approx(1.0)
    assert record["engine_core_ttft_ms"] == pytest.approx(15.0)
    assert record["engine_core_decode_ms"] == pytest.approx(200.0)
    assert record["first_token_event_tokens"] == 2
