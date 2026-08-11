# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure parts of the bulk alignment dumps.

Only the arithmetic is covered here: the span split that can silently attribute
tokens to the wrong request, the iteration selector, and the grouped-GEMM
padding math. The torch/env-touching wrappers are exercised by a real run.
"""

import pytest

from vllm.v1.worker.alignment_trace import (
    ROUTING_TRACE_PATH_ENV,
    TOKEN_TRACE_ITERS_ENV,
    TOKEN_TRACE_PATH_ENV,
    build_token_input_row,
    grouped_gemm_block_stats,
    parse_iterations,
    should_trace_token_iteration,
)


def test_token_spans_follow_the_scheduled_counts_not_equal_shares():
    row = build_token_input_row(
        iteration_index=7,
        token_ids=[10, 11, 12, 13, 14, 15],
        request_ids=["a", "b", "c"],
        # Deliberately ragged: one prefill chunk and two single-token decodes.
        num_scheduled_tokens=[4, 1, 1],
        positions=[0, 1, 2, 3, 91, 502],
    )

    assert [request["req_id"] for request in row["requests"]] == ["a", "b", "c"]
    assert [(r["start"], r["end"]) for r in row["requests"]] == [(0, 4), (4, 5), (5, 6)]
    assert row["requests"][0]["token_ids"] == [10, 11, 12, 13]
    assert row["requests"][1]["token_ids"] == [14]
    assert row["requests"][2]["token_ids"] == [15]
    # Positions are what distinguish a chunk's place in its sequence, so they
    # have to be split on the same boundaries rather than re-derived.
    assert row["requests"][1]["positions"] == [91]
    assert row["requests"][2]["positions"] == [502]
    assert row["tokens"] == 6
    assert row["num_reqs"] == 3
    assert row["iteration"] == 7


def test_positions_stay_absent_when_not_captured():
    row = build_token_input_row(
        iteration_index=0,
        token_ids=[1, 2],
        request_ids=["a"],
        num_scheduled_tokens=[2],
        positions=None,
    )

    assert "positions_flat" not in row
    assert "positions" not in row["requests"][0]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", None),
        ("   ", None),
        ("3", {3}),
        ("0,5", {0, 5}),
        ("10-12", {10, 11, 12}),
        ("1, 4-6 ,9", {1, 4, 5, 6, 9}),
        # A degenerate range is one iteration, not an empty set.
        ("7-7", {7}),
    ],
)
def test_iteration_selector_parsing(raw, expected):
    assert parse_iterations(raw) == expected


def test_token_trace_is_off_until_a_path_is_set(monkeypatch):
    monkeypatch.delenv(TOKEN_TRACE_PATH_ENV, raising=False)
    monkeypatch.delenv(ROUTING_TRACE_PATH_ENV, raising=False)
    monkeypatch.setenv(TOKEN_TRACE_ITERS_ENV, "5")

    # An iteration selector alone must not switch the dump on.
    assert should_trace_token_iteration(5) is False

    monkeypatch.setenv(TOKEN_TRACE_PATH_ENV, "/dev/null")
    assert should_trace_token_iteration(5) is True
    assert should_trace_token_iteration(4) is False

    # No selector means every iteration, including the one with no index.
    monkeypatch.delenv(TOKEN_TRACE_ITERS_ENV)
    assert should_trace_token_iteration(4) is True
    assert should_trace_token_iteration(None) is True


def test_grouped_gemm_padding_counts_only_experts_that_got_work():
    # Two experts hold 65 and 1 rows; at block_m=64 that is 2 blocks and 1 block.
    stats = grouped_gemm_block_stats(
        local_counts=[65, 1, 0, 0],
        total_assignments=66,
        global_num_experts=4,
        block_m=64,
    )

    assert stats["local_padded"] == 128 + 64
    # The launch is sized for the worst case: every expert may need a short block.
    assert stats["sorted_token_ids_len"] == 66 + 4 * 63
    assert stats["launch_m_blocks"] == 5
    assert stats["effective_m_blocks"] == 3
    assert stats["m_block_overlaunch"] == pytest.approx(5 / 3)


def test_grouped_gemm_overlaunch_is_none_when_no_expert_is_local():
    stats = grouped_gemm_block_stats(
        local_counts=[0, 0],
        total_assignments=0,
        global_num_experts=2,
        block_m=64,
    )

    assert stats["local_padded"] == 0
    assert stats["effective_m_blocks"] == 0
    # Not zero and not a division by zero: there is simply no ratio to report.
    assert stats["m_block_overlaunch"] is None
