# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, per-KV-cache-group block-pool event counters (diagnostic only).

Enabled by ``VLLM_VIBESIM_BLOCK_POOL_COUNTERS=1``. When unset, ``BlockPool``
never constructs a ``BlockPoolCounters`` and every hook is a single
``is not None`` check, so scheduling behavior and logs are unchanged.

The counters only observe; they never touch block state. Every hook swallows
its own exceptions (counted in ``errors``) so instrumentation can never raise
into the scheduler.

Group attribution:

- a cached block (it has a primary hash) belongs to the group encoded in its
  ``BlockHashWithGroupId``;
- an uncached block belongs to the group that last allocated it (the
  ``kv_cache_group_id`` of the single-type manager that called
  ``get_new_blocks``); ``-1`` when the caller is not a single-type manager.

Per-group counters (deltas since the previous ``take_record``):

- ``alloc``: blocks handed out by ``get_new_blocks`` (requester group);
- ``free_cached`` / ``free_uncached``: blocks whose ref count reached zero in
  ``free_blocks`` / ``unpin_blocks``, split by whether they keep a prefix-cache
  hash (appended to the LRU tail) or not (prepended to the head);
- ``evict_reuse``: cached blocks whose hashes were dropped because
  ``get_new_blocks`` reused them (``_maybe_evict_cached_block`` on the
  allocation path; victim group);
- ``evict_by``: the same evictions attributed to the requester group;
- ``evict_other``: ``_maybe_evict_cached_block`` outside allocation
  (connector ``evict_blocks``, Mamba checkpoint replacement; victim group);
- ``evict_hashes``: hash keys removed by both evict paths (a block can own
  partial-prefix keys beside its primary hash);
- ``rehash``: blocks whose hashes were replaced in place (partial->full
  promotion, partial-entry upgrade/replace; old group);
- ``hit_free``: blocks touched while free (a cached free block revived by a
  prefix hit); ``hit_shared``: blocks touched while already referenced;
- ``store``: blocks that became cached (gained a primary hash).

Gauges (absolute at ``take_record`` time): ``free`` (free-queue length),
``cached_free`` per group (free blocks that still own a prefix-cache hash).
"""

import os
import sys
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import get_group_id

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)

BLOCK_POOL_COUNTERS_ENV = "VLLM_VIBESIM_BLOCK_POOL_COUNTERS"
# Optional: every N records, also emit an exact free-list recount of
# ``cached_free`` (O(num_blocks)) to validate the incremental gauge.
BLOCK_POOL_RECOUNT_EVERY_ENV = "VLLM_VIBESIM_BLOCK_POOL_RECOUNT_EVERY"

RECORD_SCHEMA_VERSION = 1

COUNTER_FIELDS = (
    "alloc",
    "free_cached",
    "free_uncached",
    "evict_reuse",
    "evict_by",
    "evict_other",
    "evict_hashes",
    "rehash",
    "hit_free",
    "hit_shared",
    "store",
)
_IDX = {name: i for i, name in enumerate(COUNTER_FIELDS)}
_ALLOC = _IDX["alloc"]
_FREE_CACHED = _IDX["free_cached"]
_FREE_UNCACHED = _IDX["free_uncached"]
_EVICT_REUSE = _IDX["evict_reuse"]
_EVICT_BY = _IDX["evict_by"]
_EVICT_OTHER = _IDX["evict_other"]
_EVICT_HASHES = _IDX["evict_hashes"]
_REHASH = _IDX["rehash"]
_HIT_FREE = _IDX["hit_free"]
_HIT_SHARED = _IDX["hit_shared"]
_STORE = _IDX["store"]

UNKNOWN_GROUP = -1


def block_pool_counters_enabled() -> bool:
    return os.environ.get(BLOCK_POOL_COUNTERS_ENV, "0") not in ("", "0")


def _caller_group_id(depth: int) -> int:
    """Return ``self.kv_cache_group_id`` of the frame ``depth`` levels up."""
    try:
        owner = sys._getframe(depth + 1).f_locals.get("self")
        group_id = getattr(owner, "kv_cache_group_id", None)
        return group_id if isinstance(group_id, int) else UNKNOWN_GROUP
    except Exception:
        return UNKNOWN_GROUP


def _in_free_queue(block: "KVCacheBlock") -> bool:
    # ``FreeKVCacheBlockQueue`` links exactly its members (sentinels at both
    # ends) and clears both links on pop/remove.
    return block.prev_free_block is not None


class BlockPoolCounters:
    """Per-group block-pool counters; see the module docstring."""

    def __init__(self) -> None:
        self._counts: dict[int, list[int]] = {}
        self._cached_free: dict[int, int] = {}
        # Last allocating group of each block id (attribution of uncached
        # blocks). Bounded by the pool size.
        self._owner: dict[int, int] = {}
        # Requester group while ``get_new_blocks`` runs (its evictions are
        # reuse evictions); None otherwise.
        self.alloc_group: int | None = None
        self.errors = 0
        self.num_records = 0
        try:
            self._recount_every = int(
                os.environ.get(BLOCK_POOL_RECOUNT_EVERY_ENV, "0") or 0
            )
        except ValueError:
            self._recount_every = 0

    # ----------------------------------------------------------------- helpers
    def _row(self, group_id: int) -> list[int]:
        row = self._counts.get(group_id)
        if row is None:
            row = [0] * len(COUNTER_FIELDS)
            self._counts[group_id] = row
        return row

    def group_of(self, block: "KVCacheBlock") -> int:
        try:
            block_hash = block.block_hash
            if block_hash is not None:
                return get_group_id(block_hash)
            return self._owner.get(block.block_id, UNKNOWN_GROUP)
        except Exception:
            self.errors += 1
            return UNKNOWN_GROUP

    def _add_cached_free(self, group_id: int, delta: int) -> None:
        self._cached_free[group_id] = self._cached_free.get(group_id, 0) + delta

    # ------------------------------------------------------------------- hooks
    def begin_alloc(self, depth: int) -> None:
        """Called by ``get_new_blocks``; ``depth`` frames up is the requester."""
        try:
            self.alloc_group = _caller_group_id(depth + 1)
        except Exception:
            self.errors += 1
            self.alloc_group = UNKNOWN_GROUP

    def end_alloc(self, blocks: list["KVCacheBlock"]) -> None:
        try:
            group_id = (
                self.alloc_group if self.alloc_group is not None else UNKNOWN_GROUP
            )
            self._row(group_id)[_ALLOC] += len(blocks)
            owner = self._owner
            for block in blocks:
                owner[block.block_id] = group_id
        except Exception:
            self.errors += 1
        finally:
            self.alloc_group = None

    def on_evict(self, victim_group: int, num_hashes: int) -> None:
        try:
            if self.alloc_group is not None:
                self._row(victim_group)[_EVICT_REUSE] += 1
                self._row(self.alloc_group)[_EVICT_BY] += 1
                self._row(victim_group)[_EVICT_HASHES] += num_hashes
            else:
                row = self._row(victim_group)
                row[_EVICT_OTHER] += 1
                row[_EVICT_HASHES] += num_hashes
        except Exception:
            self.errors += 1

    def on_rehash(self, group_id: int) -> None:
        try:
            self._row(group_id)[_REHASH] += 1
        except Exception:
            self.errors += 1

    def on_hashes_removed(self, block: "KVCacheBlock", group_id: int) -> None:
        """A block lost its primary hash (any path).

        It leaves ``cached_free`` if it sits in the free queue, or if
        ``get_new_blocks`` just popped it (reuse eviction).
        """
        try:
            if self.alloc_group is not None or _in_free_queue(block):
                self._add_cached_free(group_id, -1)
        except Exception:
            self.errors += 1

    def on_hash_inserted(self, block: "KVCacheBlock", group_id: int) -> None:
        """A block gained its primary hash (it was uncached before)."""
        try:
            self._row(group_id)[_STORE] += 1
            if _in_free_queue(block):
                self._add_cached_free(group_id, 1)
        except Exception:
            self.errors += 1

    def on_released(self, blocks: list["KVCacheBlock"]) -> None:
        """Blocks whose ref count reached zero and that joined the free queue."""
        try:
            for block in blocks:
                group_id = self.group_of(block)
                if block.block_hash is not None:
                    self._row(group_id)[_FREE_CACHED] += 1
                    self._add_cached_free(group_id, 1)
                else:
                    self._row(group_id)[_FREE_UNCACHED] += 1
        except Exception:
            self.errors += 1

    def on_touch(self, block: "KVCacheBlock", was_free: bool) -> None:
        try:
            group_id = self.group_of(block)
            if was_free:
                self._row(group_id)[_HIT_FREE] += 1
                if block.block_hash is not None:
                    self._add_cached_free(group_id, -1)
            else:
                self._row(group_id)[_HIT_SHARED] += 1
        except Exception:
            self.errors += 1

    def on_reset(self) -> None:
        try:
            self._cached_free.clear()
        except Exception:
            self.errors += 1

    # ------------------------------------------------------------------ export
    @staticmethod
    def recount_cached_free(free_block_queue: Any) -> dict[int, int]:
        """Exact O(num_free_blocks) recount of cached free blocks per group."""
        counts: dict[int, int] = {}
        head = free_block_queue.fake_free_list_head
        tail = free_block_queue.fake_free_list_tail
        block = head.next_free_block
        while block is not None and block is not tail:
            block_hash = block.block_hash
            if block_hash is not None:
                group_id = get_group_id(block_hash)
                counts[group_id] = counts.get(group_id, 0) + 1
            block = block.next_free_block
        return counts

    def take_record(
        self, num_free_blocks: int, free_block_queue: Any = None
    ) -> dict[str, Any] | None:
        """Return the per-step record and reset the delta counters."""
        try:
            groups: dict[str, dict[str, int]] = {}
            for group_id in sorted(self._counts):
                row = self._counts[group_id]
                nonzero = {
                    name: value for name, value in zip(COUNTER_FIELDS, row) if value
                }
                if nonzero:
                    groups[str(group_id)] = nonzero
            self._counts.clear()
            record: dict[str, Any] = {
                "v": RECORD_SCHEMA_VERSION,
                "free": num_free_blocks,
                "cached_free": {
                    str(group_id): value
                    for group_id, value in sorted(self._cached_free.items())
                    if value
                },
                "groups": groups,
            }
            self.num_records += 1
            if (
                free_block_queue is not None
                and self._recount_every > 0
                and self.num_records % self._recount_every == 0
            ):
                record["cached_free_exact"] = {
                    str(group_id): value
                    for group_id, value in sorted(
                        self.recount_cached_free(free_block_queue).items()
                    )
                }
            if self.errors:
                record["errors"] = self.errors
            return record
        except Exception:
            self.errors += 1
            return None


def group_labels(kv_cache_config: Any) -> dict[str, str]:
    """Map group id -> spec class name (with an ``eagle`` suffix)."""
    labels: dict[str, str] = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        label = type(group.kv_cache_spec).__name__
        inner = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        if isinstance(inner, dict) and inner:
            names = sorted({type(spec).__name__ for spec in inner.values()})
            label = f"{label}[{','.join(names)}]"
        if getattr(group, "is_eagle_group", False):
            label += "+eagle"
        labels[str(group_id)] = label
    return labels


def take_block_pool_record(scheduler: Any) -> dict[str, Any] | None:
    """EngineCore helper: the scheduler pool's record, or None when disabled.

    The first call also logs a ``VibeSimBlockPoolGroups`` line with the group
    labels and attaches the same metadata as ``meta`` to the first record.
    Never raises.
    """
    try:
        kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        counters = getattr(block_pool, "vibesim_counters", None)
        if counters is None:
            return None
        meta: dict[str, Any] | None = None
        if counters.num_records == 0:
            try:
                import json

                meta = {
                    "v": RECORD_SCHEMA_VERSION,
                    "num_blocks": block_pool.num_gpu_blocks,
                    "labels": group_labels(kv_cache_manager.kv_cache_config),
                    "fields": list(COUNTER_FIELDS),
                }
                logger.info(
                    "VibeSimBlockPoolGroups %s",
                    json.dumps(meta, separators=(",", ":")),
                )
            except Exception:
                counters.errors += 1
        record = counters.take_record(
            block_pool.get_num_free_blocks(), block_pool.free_block_queue
        )
        if record is not None and meta is not None:
            # Also carry the labels in the first record itself.
            record["meta"] = meta
        return record
    except Exception:
        return None
