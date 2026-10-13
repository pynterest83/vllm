# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-KV-group block-pool counters (VLLM_VIBESIM_BLOCK_POOL_COUNTERS)."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.block_pool_counters import (
    BLOCK_POOL_COUNTERS_ENV,
    BlockPoolCounters,
    group_labels,
    take_block_pool_record,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


class _FakeManager:
    """Stands in for a single-type manager: owns a group id, calls the pool."""

    def __init__(self, block_pool: BlockPool, kv_cache_group_id: int):
        self.block_pool = block_pool
        self.kv_cache_group_id = kv_cache_group_id

    def alloc(self, num_blocks: int):
        return self.block_pool.get_new_blocks(num_blocks)


def _cache(pool: BlockPool, block, tag: bytes, group_id: int) -> None:
    pool._insert_block_hash(
        make_block_hash_with_group_id(BlockHash(tag), group_id), block, num_tokens=16
    )


def _check_gauge(pool: BlockPool) -> None:
    counters = pool.vibesim_counters
    if counters is None:
        return
    incremental = {g: n for g, n in counters._cached_free.items() if n}
    assert incremental == BlockPoolCounters.recount_cached_free(pool.free_block_queue)


def _run_scenario(pool: BlockPool) -> list[dict | None]:
    """Allocate / cache / free / touch / reuse-evict / explicit-evict."""
    counters = pool.vibesim_counters
    records = []

    def snap():
        _check_gauge(pool)
        records.append(
            counters.take_record(pool.get_num_free_blocks()) if counters else None
        )

    attn = _FakeManager(pool, 0)
    mamba = _FakeManager(pool, 1)
    a = attn.alloc(3)
    m = mamba.alloc(2)
    _cache(pool, a[0], b"a0", 0)
    _cache(pool, a[1], b"a1", 0)
    _cache(pool, m[0], b"m0", 1)
    snap()  # 0: alloc + store

    pool.free_blocks(reversed(a))
    pool.free_blocks(reversed(m))
    snap()  # 1: free cached / uncached

    pool.touch([a[0]])
    pool.touch([a[0]])
    snap()  # 2: hit_free then hit_shared

    mamba.alloc(pool.get_num_free_blocks())
    snap()  # 3: reuse evictions of a[1] (group 0) and m[0] (group 1)

    pool.evict_blocks({a[0].block_id})
    snap()  # 4: explicit eviction of a referenced block
    return records


def _pool_state(pool: BlockPool):
    return (
        [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        [(b.ref_cnt, b.block_hash) for b in pool.blocks],
        len(pool.cached_block_hash_to_block),
    )


def test_counters_attribute_events_per_group(monkeypatch):
    monkeypatch.setenv(BLOCK_POOL_COUNTERS_ENV, "1")
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    assert pool.vibesim_counters is not None
    r = _run_scenario(pool)

    assert r[0]["free"] == 2
    assert r[0]["groups"] == {
        "0": {"alloc": 3, "store": 2},
        "1": {"alloc": 2, "store": 1},
    }
    assert r[0]["cached_free"] == {}

    assert r[1]["free"] == 7
    assert r[1]["groups"] == {
        "0": {"free_cached": 2, "free_uncached": 1},
        "1": {"free_cached": 1, "free_uncached": 1},
    }
    assert r[1]["cached_free"] == {"0": 2, "1": 1}

    assert r[2]["groups"] == {"0": {"hit_free": 1, "hit_shared": 1}}
    assert r[2]["cached_free"] == {"0": 1, "1": 1}

    assert r[3]["free"] == 0
    assert r[3]["groups"] == {
        "0": {"evict_reuse": 1, "evict_hashes": 1},
        "1": {"alloc": 6, "evict_reuse": 1, "evict_by": 2, "evict_hashes": 1},
    }
    assert r[3]["cached_free"] == {}

    assert r[4]["groups"] == {"0": {"evict_other": 1, "evict_hashes": 1}}
    assert all("errors" not in rec for rec in r)


def test_counters_unset_is_inert_and_behavior_identical(monkeypatch):
    monkeypatch.delenv(BLOCK_POOL_COUNTERS_ENV, raising=False)
    off = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    assert off.vibesim_counters is None
    assert _run_scenario(off) == [None] * 5
    scheduler = SimpleNamespace(kv_cache_manager=SimpleNamespace(block_pool=off))
    assert take_block_pool_record(scheduler) is None

    monkeypatch.setenv(BLOCK_POOL_COUNTERS_ENV, "1")
    on = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    _run_scenario(on)
    assert _pool_state(on) == _pool_state(off)


def test_counter_errors_never_raise(monkeypatch):
    monkeypatch.setenv(BLOCK_POOL_COUNTERS_ENV, "1")
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    counters = pool.vibesim_counters
    counters._counts = None  # Corrupt the counter state on purpose.
    blocks = _FakeManager(pool, 0).alloc(2)
    pool.touch(blocks)
    pool.free_blocks(blocks)
    pool.free_blocks(blocks)
    assert counters.errors > 0
    assert pool.get_num_free_blocks() == 3


def _make_request(request_id: str, token_ids: list[int], block_size: int) -> Request:
    params = SamplingParams(max_tokens=17)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def test_hybrid_manager_attributes_attention_and_mamba_groups(monkeypatch):
    monkeypatch.setenv(BLOCK_POOL_COUNTERS_ENV, "1")
    init_none_hash(sha256)
    block_size = 16
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attn"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=block_size,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=1024,
        enable_caching=True,
        hash_block_size=block_size,
        scheduler_block_size=block_size,
    )
    pool = manager.block_pool
    scheduler = SimpleNamespace(kv_cache_manager=manager)
    assert group_labels(config) == {"0": "FullAttentionSpec", "1": "MambaSpec"}

    tokens = list(range(4 * block_size))
    req0 = _make_request("0", tokens + [7], block_size)
    computed, num_computed, _ = manager.get_computed_blocks(req0)
    assert num_computed == 0
    # Align mode checkpoints Mamba state at block-aligned chunk ends.
    for end in (len(tokens), len(req0.prompt_token_ids)):
        num_new = end - req0.num_computed_tokens
        assert manager.allocate_slots(req0, num_new, 0, computed)
        computed = manager.empty_kv_cache_blocks
        req0.num_computed_tokens = end
        manager.new_step_starts()
    first = take_block_pool_record(scheduler)
    assert first["groups"]["0"]["alloc"] > 0
    assert first["groups"]["1"]["alloc"] > 0
    assert "-1" not in first["groups"]
    json.dumps(first)
    assert first["meta"]["labels"] == {"0": "FullAttentionSpec", "1": "MambaSpec"}
    _check_gauge(pool)

    manager.free(req0)
    freed = take_block_pool_record(scheduler)
    assert "meta" not in freed
    assert freed["groups"]["0"]["free_cached"] > 0
    assert freed["cached_free"]["0"] > 0
    _check_gauge(pool)

    req1 = _make_request("1", tokens + [8], block_size)
    computed, num_computed, _ = manager.get_computed_blocks(req1)
    assert num_computed > 0
    assert manager.allocate_slots(
        req1, len(req1.prompt_token_ids) - num_computed, num_computed, computed
    )
    hit = take_block_pool_record(scheduler)
    assert hit["groups"]["0"]["hit_free"] > 0
    assert hit["groups"]["1"]["hit_free"] > 0
    assert hit["cached_free"] == {}
    _check_gauge(pool)
    manager.free(req1)
    _check_gauge(pool)
