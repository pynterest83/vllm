# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8-KV prefill rows on a BF16 workspace (FLASHINFER_MLA_SPARSE_SM90).

With ``VLLM_FP8_KV_PREFILL_BF16_WORKSPACE=1`` the prefill rows of a step read
a BF16 copy of their requests' contexts through a second FlashInfer plan while
decode rows keep the FP8 plan. The CPU tests pin the workspace sizing and
fallback, the index remapping (request-local position -> workspace row), the
plan selection in the builder, and output equivalence with the FP8 path using
a reference attention in place of FlashInfer. The CUDA tests check the real
remap/gather kernels and, on SM90, the real FP8 vs BF16-workspace plans.
"""

from types import SimpleNamespace

import pytest
import torch

# isort: off
import vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 as sm90_mod
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
    FlashInferMLASparseSM90Builder,
    FlashInferMLASparseSM90Impl,
    plan_prefill_workspace,
    prefill_workspace_tokens,
)
# isort: on

HEAD = 512
TOPK = 128  # triton convert requires width % 128 == 0
BS = 4


def test_e4m3_to_bf16_is_exact():
    """Every finite E4M3 value (incl. subnormals) is representable in BF16,
    so dequantizing into the workspace loses nothing at k_scale == 1 (or any
    power-of-two scale in range)."""
    bits = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    vals = bits.view(torch.float8_e4m3fn).float()
    finite = torch.isfinite(vals)
    assert int(finite.sum()) == 254  # 0x7f / 0xff are NaN
    v = vals[finite]
    assert torch.equal(v.to(torch.bfloat16).float(), v)
    for scale in (0.5, 2.0, 0.125):
        assert torch.equal((v * scale).to(torch.bfloat16).float(), v * scale)


@pytest.mark.parametrize(
    "enabled,fp8,mml,mnbt,override,expected",
    [
        (False, True, 262144, 8192, 0, 0),
        (True, False, 262144, 8192, 0, 0),
        (True, True, 262144, 8192, 0, 262144 + 8192),
        (True, True, 131072, 8192, 0, 131072 + 8192),
        (True, True, 1 << 20, 8192, 0, 262144 + 8192),
        (True, True, 262144, 8192, 50000, 50000),
    ],
)
def test_prefill_workspace_tokens(enabled, fp8, mml, mnbt, override, expected):
    assert prefill_workspace_tokens(enabled, fp8, mml, mnbt, override) == expected
    # 512-wide BF16 latent: the default for 262144/8192 is 264 MiB.
    if expected == 262144 + 8192:
        assert expected * HEAD * 2 == 264 * 2**20


def test_plan_prefill_workspace_layout_and_budget():
    # 2 decodes (q=1), prefill A (q=5, ctx 100), prefill B (q=3, ctx 3000).
    qsl = torch.tensor([0, 1, 2, 7, 10], dtype=torch.int32)
    sl = torch.tensor([100, 3000], dtype=torch.int32)
    layout = plan_prefill_workspace(qsl, sl, 2, budget_tokens=3100)
    assert layout is not None
    assert layout.num_decode_tokens == 2 and layout.num_tokens == 3100
    assert layout.cu_seq_lens.dtype == torch.int32
    assert layout.cu_seq_lens.tolist() == [0, 100, 3100]
    assert layout.request_ids.dtype == torch.int32
    assert layout.request_ids.tolist() == [-1, -1, 0, 0, 0, 0, 0, 1, 1, 1]
    # One token over the budget: the whole batch falls back to the FP8 plan.
    assert plan_prefill_workspace(qsl, sl, 2, budget_tokens=3099) is None
    # No prefill requests.
    assert (
        plan_prefill_workspace(qsl[:3], torch.zeros(0, dtype=torch.int32), 2, 10**6)
        is None
    )
    # Prefill-only batch.
    layout = plan_prefill_workspace(
        torch.tensor([0, 4], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        0,
        10,
    )
    assert layout is not None and layout.num_decode_tokens == 0
    assert layout.request_ids.tolist() == [0, 0, 0, 0]


# --------------------------------------------------------------------------
# Builder plan selection
# --------------------------------------------------------------------------


class RecState:
    def __init__(self, width=TOPK):
        self.device = torch.device("cpu")
        self.plan_calls: list = []
        self.index_topk = 2048
        self.index_kpool = 4
        self.max_valid = 2051
        self.topk_width = width

    def plan(self, num_tokens, kv_lens, cam, req_id_per_token, row_offset=0):
        self.plan_calls.append(
            (num_tokens, kv_lens.tolist(), cam is not None, row_offset)
        )


def _builder(ws_tokens, use_mha=False, dcp=1):
    builder = object.__new__(FlashInferMLASparseSM90Builder)
    builder._adaptive_verification = False
    builder.state = RecState()
    builder.prefill_state = RecState() if ws_tokens else None
    builder.prefill_ws_tokens = ws_tokens
    builder.dcp_world_size = dcp
    builder._attention_layer = SimpleNamespace(_use_sparse_mha=lambda _: use_mha)
    return builder


def _metadata(monkeypatch, num_decodes, num_prefills, num_decode_tokens, num_reqs):
    metadata = object.__new__(sm90_mod.FlashInferMLASparseSM90Metadata)
    metadata.state = None
    metadata.req_id_per_token = torch.zeros(16, dtype=torch.int32)
    metadata.block_table = torch.arange(num_reqs * 8, dtype=torch.int32).view(
        num_reqs, 8
    )
    metadata.num_decodes = num_decodes
    metadata.num_prefills = num_prefills
    metadata.num_decode_tokens = num_decode_tokens
    monkeypatch.setattr(
        sm90_mod.FlashInferMLASparseMetadataBuilder,
        "build",
        lambda *_args, **_kwargs: metadata,
    )
    return metadata


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize(
    "case", ["off", "ws", "over_budget", "mha", "dcp", "prefill_only"]
)
def test_builder_plan_selection(monkeypatch, case, exact):
    """Decode rows always go to the FP8 state. Prefill rows go to the BF16
    state (row offset = first prefill row) only when the workspace is on,
    the prefills are MQA-routed, there is no DCP, and the contexts fit."""
    prefill_only = case == "prefill_only"
    ws_tokens = {"off": 0, "over_budget": 100}.get(case, 10_000)
    builder = _builder(ws_tokens, use_mha=case == "mha", dcp=2 if case == "dcp" else 1)
    if prefill_only:
        # One prefill request: q=4 at ctx 3001 (rows 2998..3001).
        qsl = torch.tensor([0, 4], dtype=torch.int32)
        seq_lens = torch.tensor([3001], dtype=torch.int32)
        metadata = _metadata(monkeypatch, 0, 1, 0, 1)
    else:
        # Decode (ctx 1402), prefill (q=5, ctx 105).
        qsl = torch.tensor([0, 1, 6], dtype=torch.int32)
        seq_lens = torch.tensor([1402, 105], dtype=torch.int32)
        metadata = _metadata(monkeypatch, 1, 1, 1, 2)
    cam = SimpleNamespace(
        num_reqs=qsl.numel() - 1,
        query_start_loc_cpu=qsl,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=None if exact else seq_lens,
    )

    result = builder.build(0, cam)

    assert result.state is builder.state
    if prefill_only:
        # Exact valid counts past index_topk are topk + ctx % kpool; host
        # upper bounds are capped at the largest valid count (device clamp).
        decode_lens = []
        prefill_lens = [2050, 2051, 2048, 2049] if exact else [2051] * 4
    else:
        decode_lens, prefill_lens = [1402], [101, 102, 103, 104, 105]
    main = builder.state.plan_calls
    assert len(main) == 1
    if case in ("ws", "prefill_only"):
        nd = len(decode_lens)
        assert main[0][:2] == (nd, decode_lens)
        sub = builder.prefill_state.plan_calls
        assert len(sub) == 1
        assert sub[0][:2] == (len(prefill_lens), prefill_lens)
        assert sub[0][2] == (not exact)
        assert sub[0][3] == (0 if exact else nd)
        ws = result.prefill_ws
        assert ws is not None and result.prefill_state is builder.prefill_state
        assert ws.num_decode_tokens == nd
        assert ws.num_tokens == int(seq_lens[metadata.num_decodes :].sum())
        assert ws.request_ids.tolist() == [-1] * nd + [0] * len(prefill_lens)
        assert ws.workspace_starts.tolist() == [0]
        assert torch.equal(ws.block_table, metadata.block_table[metadata.num_decodes :])
    elif case == "mha":
        assert main[0][:2] == (1, decode_lens)
        assert result.prefill_ws is None and result.prefill_state is None
        assert builder.prefill_state.plan_calls == []
    else:  # off / over_budget / dcp: every row on the FP8 plan
        assert main[0][:2] == (int(qsl[-1]), decode_lens + prefill_lens)
        assert result.prefill_ws is None and result.prefill_state is None
        if builder.prefill_state is not None:
            assert builder.prefill_state.plan_calls == []


# --------------------------------------------------------------------------
# Forward equivalence with a reference attention
# --------------------------------------------------------------------------


def ref_convert(
    req_id,
    block_table,
    token_indices,
    BLOCK_SIZE=64,
    HAS_PREFILL_WORKSPACE=False,
    prefill_workspace_request_ids=None,
    prefill_workspace_starts=None,
    **_,
):
    """CPU mirror of sparse_mla_index_remap_kernel with compaction."""
    out = torch.full_like(token_indices, -1)
    counts = torch.zeros(token_indices.shape[0], dtype=torch.int32)
    for t in range(token_indices.shape[0]):
        vals = []
        pr = int(prefill_workspace_request_ids[t]) if HAS_PREFILL_WORKSPACE else -1
        for j in range(token_indices.shape[1]):
            pos = int(token_indices[t, j])
            if pos < 0 or pos // BLOCK_SIZE >= block_table.shape[1]:
                continue
            if pr >= 0:
                vals.append(int(prefill_workspace_starts[pr]) + pos)
            else:
                blk = int(block_table[int(req_id[t]), pos // BLOCK_SIZE])
                vals.append(blk * BLOCK_SIZE + pos % BLOCK_SIZE)
        out[t, : len(vals)] = torch.tensor(vals, dtype=out.dtype)
        counts[t] = len(vals)
    return out, counts


def ref_gather(
    src_cache,
    dst,
    block_table,
    cu_seq_lens,
    token_to_seq,
    num_tokens,
    kv_cache_dtype,
    scale,
    seq_starts=None,
):
    assert kv_cache_dtype in ("fp8", "fp8_e4m3") and seq_starts is None
    assert int(cu_seq_lens[-1]) == num_tokens == dst.shape[0]
    cache = src_cache.view(torch.float8_e4m3fn)
    bs = cache.shape[1]
    for i in range(cu_seq_lens.numel() - 1):
        lo, hi = int(cu_seq_lens[i]), int(cu_seq_lens[i + 1])
        pos = torch.arange(hi - lo)
        blocks = block_table[i, pos // bs].long()
        vals = cache[blocks, pos % bs].float() * float(scale)
        dst[lo:hi] = vals.to(dst.dtype)


class RefWrapper:
    """Sparse MLA attention over the state's packed CSR rows (fp32)."""

    def __init__(self, state, kv_dtype, sm_scale):
        self.state = state
        self.kv_dtype = kv_dtype
        self.sm_scale = sm_scale
        self.calls: list = []

    def run(self, q_nope, q_pe, ckv, kpe, out=None, **scales):
        assert ckv.dtype == self.kv_dtype
        if self.kv_dtype == torch.float8_e4m3fn:
            assert set(scales) == {"ckv_scale", "kpe_scale"}
        else:
            assert scales == {}, "FlashInfer rejects scales for BF16 KV"
        self.calls.append((q_nope.shape[0], ckv.dtype, ckv.shape[0], scales))
        ckv_s = scales.get("ckv_scale", 1.0)
        kpe_s = scales.get("kpe_scale", 1.0)
        indptr, indices = self.state.kv_indptr, self.state.kv_indices
        res = torch.empty(q_nope.shape, dtype=torch.float32)
        for r in range(q_nope.shape[0]):
            idx = indices[int(indptr[r]) : int(indptr[r + 1])].long()
            k = ckv[idx, 0].float() * ckv_s
            kp = kpe[idx, 0].float() * kpe_s
            logits = (q_nope[r].float() @ k.T + q_pe[r].float() @ kp.T) * self.sm_scale
            res[r] = torch.softmax(logits, dim=-1) @ k
        if out is None:
            return res.to(q_nope.dtype)
        out.copy_(res.to(out.dtype))
        return out


class RefState:
    def __init__(self, kv_dtype, sm_scale, max_tokens=16, kv_indices=None):
        self.kv_indptr = torch.zeros(max_tokens + 1, dtype=torch.int32)
        self.kv_indices = (
            kv_indices
            if kv_indices is not None
            else torch.zeros(max_tokens * TOPK, dtype=torch.int32)
        )
        self.wrapper = RefWrapper(self, kv_dtype, sm_scale)

    def set_lens(self, lens):
        self.kv_indptr.zero_()
        self.kv_indptr[1 : len(lens) + 1] = torch.tensor(lens).cumsum(0)
        self.kv_indptr[len(lens) + 1 :] = int(sum(lens))

    def pack_indices(self, slots):
        for row in range(slots.shape[0]):
            start, end = self.kv_indptr[row : row + 2].tolist()
            self.kv_indices[start:end] = slots[row, : end - start].clamp(min=0)


def _make_batch(gen):
    """Two decodes then two prefills; per-row top-k with interior -1 gaps."""
    seq_lens = [9, 6, 13, 7]
    q_lens = [1, 1, 3, 2]
    num_blocks = 32
    perm = torch.randperm(num_blocks, generator=gen)
    block_table = torch.zeros(4, 8, dtype=torch.int32)
    used = 0
    for i, sl in enumerate(seq_lens):
        nb = -(-sl // BS)
        block_table[i, :nb] = perm[used : used + nb]
        used += nb
    req_id, topk = [], []
    for i, (sl, ql) in enumerate(zip(seq_lens, q_lens)):
        for j in range(ql):
            ctx = sl - ql + j + 1
            row = torch.full((TOPK,), -1, dtype=torch.int32)
            pick = torch.randperm(ctx, generator=gen)[: max(1, ctx - 2)]
            cols = torch.randperm(TOPK // 2, generator=gen)[: pick.numel()]
            row[cols.sort().values] = pick.to(torch.int32)
            topk.append(row)
            req_id.append(i)
    cache = (
        (torch.randn(num_blocks, BS, HEAD, generator=gen) * 4)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return (
        torch.tensor(seq_lens),
        torch.tensor(q_lens),
        block_table,
        torch.tensor(req_id, dtype=torch.int32),
        torch.stack(topk),
        cache,
    )


@pytest.mark.parametrize("k_scale,exact", [(1.0, True), (0.5, True), (0.3, False)])
def test_forward_prefill_ws_matches_fp8_path(monkeypatch, k_scale, exact):
    gen = torch.Generator().manual_seed(0)
    seq_lens, q_lens, block_table, req_id, topk, cache = _make_batch(gen)
    rows, heads = topk.shape[0], 2
    num_decodes, nd = 2, 2
    monkeypatch.setattr(
        sm90_mod, "triton_convert_req_index_to_global_index", ref_convert
    )
    monkeypatch.setattr(sm90_mod.ops, "gather_and_maybe_dequant_cache", ref_gather)

    impl = object.__new__(FlashInferMLASparseSM90Impl)
    impl.num_heads = heads
    impl.head_size = HEAD
    impl.scale = HEAD**-0.5
    impl.kv_lora_rank = HEAD
    impl.qk_rope_head_dim = 0
    impl.kv_cache_dtype = "fp8"
    impl.use_fp8_kv_cache = True
    impl.topk_indices_buffer = topk.clone()
    impl.prefill_ws_tokens = 64
    ws_buf = torch.full((64, HEAD), float("nan"), dtype=torch.bfloat16)
    monkeypatch.setattr(impl, "_prefill_workspace", lambda: ws_buf)

    layer = SimpleNamespace(_k_scale_float=k_scale, _k_scale=torch.tensor(k_scale))
    q_nope = torch.randn(rows, heads, HEAD, generator=gen).to(torch.bfloat16)
    q_rope = q_nope.new_empty(rows, heads, 0)
    _, counts = ref_convert(req_id, block_table, topk, BLOCK_SIZE=BS)
    counts = counts.tolist()

    # Baseline: every row on the FP8 plan.
    fp8_all = RefState(torch.float8_e4m3fn, impl.scale)
    fp8_all.set_lens(counts)
    base_meta = SimpleNamespace(
        req_id_per_token=req_id, block_table=block_table, block_size=BS, state=fp8_all
    )
    ref_out, _ = impl.forward_mqa((q_nope, q_rope), cache, base_meta, layer)

    # Workspace: decode rows on FP8, prefill rows on BF16 (shared indices).
    layout = plan_prefill_workspace(
        torch.cat([torch.zeros(1, dtype=torch.long), q_lens.cumsum(0)]).to(torch.int32),
        seq_lens[num_decodes:].to(torch.int32),
        num_decodes,
        impl.prefill_ws_tokens,
    )
    assert layout is not None and layout.num_decode_tokens == nd
    ws = sm90_mod.PrefillWorkspaceMetadata(
        num_decode_tokens=nd,
        num_tokens=layout.num_tokens,
        cu_seq_lens=layout.cu_seq_lens,
        request_ids=layout.request_ids,
        block_table=block_table[num_decodes:],
    )
    # Remapping keeps every row's valid count (= the planned lengths).
    _, ws_counts = ref_convert(
        req_id,
        block_table,
        topk,
        BLOCK_SIZE=BS,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=ws.request_ids,
        prefill_workspace_starts=ws.workspace_starts,
    )
    assert ws_counts.tolist() == counts
    dec = RefState(torch.float8_e4m3fn, impl.scale)
    dec.set_lens(counts[:nd])
    pre = RefState(torch.bfloat16, impl.scale, kv_indices=dec.kv_indices)
    pre.set_lens(counts[nd:])
    ws_meta = SimpleNamespace(
        req_id_per_token=req_id,
        block_table=block_table,
        block_size=BS,
        state=dec,
        prefill_state=pre,
        prefill_ws=ws,
    )
    out, lse = impl.forward_mqa((q_nope, q_rope), cache, ws_meta, layer)

    assert lse is None and out.shape == ref_out.shape
    assert [c[0] for c in dec.wrapper.calls] == [nd]
    assert dec.wrapper.calls[0][3] == {"ckv_scale": k_scale, "kpe_scale": 1.0}
    assert [c[:3] for c in pre.wrapper.calls] == [
        (rows - nd, torch.bfloat16, layout.num_tokens)
    ]
    # Workspace rows hold each prefill context in position order.
    assert not ws_buf[: layout.num_tokens].isnan().any()
    if exact:
        assert torch.equal(out.float(), ref_out.float())
    else:
        # Non-power-of-two scale: dequantized values round to BF16.
        torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
        assert torch.equal(out[:nd].float(), ref_out[:nd].float())


# --------------------------------------------------------------------------
# CUDA: real remap / gather kernels, and (SM90) the real FlashInfer plans
# --------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA"
)


@requires_cuda
@pytest.mark.parametrize("width", [128, 2176])
def test_cuda_convert_with_workspace_matches_ref(width):
    gen = torch.Generator().manual_seed(1)
    num_reqs, max_blocks, bs = 4, 64, 64
    q_lens = [1, 1, 5, 3]
    seq_lens = [500, 300, 3000, 4000]
    block_table = torch.randperm(1024, generator=gen)[: num_reqs * max_blocks]
    block_table = block_table.view(num_reqs, max_blocks).to(torch.int32)
    req_id, topk = [], []
    for i, (sl, ql) in enumerate(zip(seq_lens, q_lens)):
        for j in range(ql):
            ctx = sl - ql + j + 1
            row = torch.full((width,), -1, dtype=torch.int32)
            n = min(ctx, width - 64)
            cols = torch.randperm(width, generator=gen)[:n].sort().values
            row[cols] = torch.randperm(ctx, generator=gen)[:n].to(torch.int32)
            topk.append(row)
            req_id.append(i)
    req_id = torch.tensor(req_id, dtype=torch.int32)
    topk = torch.stack(topk)
    layout = plan_prefill_workspace(
        torch.tensor([0, 1, 2, 7, 10], dtype=torch.int32),
        torch.tensor(seq_lens[2:], dtype=torch.int32),
        2,
        10**6,
    )
    assert layout is not None
    kwargs = dict(
        BLOCK_SIZE=bs,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=layout.request_ids,
        prefill_workspace_starts=layout.cu_seq_lens[:-1],
    )
    ref, ref_counts = ref_convert(req_id, block_table, topk, **kwargs)
    cuda_kwargs = {
        k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()
    }
    out, counts = sm90_mod.triton_convert_req_index_to_global_index(
        req_id.cuda(),
        block_table.cuda(),
        topk.cuda(),
        NUM_TOPK_TOKENS=width,
        return_valid_counts=True,
        **cuda_kwargs,
    )
    assert counts.cpu().tolist() == ref_counts.tolist()
    out = out.cpu()
    for t in range(topk.shape[0]):
        n = int(ref_counts[t])
        # Multi-tile compaction order is unspecified; the set is not.
        assert sorted(out[t, :n].tolist()) == sorted(ref[t, :n].tolist())
        assert (out[t, n:] == -1).all()


@requires_cuda
@pytest.mark.parametrize("k_scale", [1.0, 0.5])
def test_cuda_gather_dequant_is_exact(k_scale):
    from vllm import _custom_ops as ops

    gen = torch.Generator().manual_seed(2)
    num_blocks, bs = 256, 64
    cache = (
        (torch.randn(num_blocks, bs, HEAD, generator=gen) * 8)
        .to(torch.float8_e4m3fn)
        .cuda()
    )
    seq_lens = torch.tensor([3000, 77, 4097], dtype=torch.int32)
    cu = torch.zeros(4, dtype=torch.int32)
    cu[1:] = seq_lens.cumsum(0)
    max_blocks = 80
    block_table = torch.randperm(num_blocks, generator=gen)[: 3 * max_blocks]
    block_table = block_table.view(3, max_blocks).to(torch.int32)
    total = int(cu[-1])
    dst = torch.full((total + 5, HEAD), float("nan"), dtype=torch.bfloat16).cuda()
    ops.gather_and_maybe_dequant_cache(
        src_cache=cache,
        dst=dst[:total],
        block_table=block_table.cuda(),
        cu_seq_lens=cu.cuda(),
        token_to_seq=cu.cuda(),
        num_tokens=total,
        kv_cache_dtype="fp8",
        scale=torch.tensor(k_scale, dtype=torch.float32).cuda(),
    )
    ref = torch.empty(total, HEAD, dtype=torch.bfloat16)
    ref_gather(cache.cpu(), ref, block_table, cu, cu, total, "fp8", k_scale)
    assert torch.equal(dst[:total].cpu(), ref)
    assert dst[total:].isnan().all()  # never writes past the planned rows


def _sm90_flashinfer_available():
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
        and sm90_mod.has_flashinfer_sm90_nope_mla()
    )


@pytest.mark.skipif(
    not _sm90_flashinfer_available(), reason="Requires SM90 FlashInfer MLA"
)
@pytest.mark.parametrize("num_decodes,prefill_q", [(8, 256), (0, 512)])
def test_sm90_prefill_ws_matches_fp8_plan(monkeypatch, num_decodes, prefill_q):
    """Real FlashInfer: FP8 plan for all rows vs FP8 decode + BF16 workspace
    prefill. Same K/V values (k_scale 1), so only kernel accumulation order
    differs."""
    torch.manual_seed(0)
    heads, topk_n, kpool, width, bs = 16, 2048, 4, 2176, 64
    dev = torch.device("cuda")
    prefill_ctx = 5000
    seq_lens = [3000 + 37 * i for i in range(num_decodes)] + [prefill_ctx]
    q_lens = [1] * num_decodes + [prefill_q]
    num_reqs = len(seq_lens)
    max_blocks = -(-max(seq_lens) // bs)
    num_blocks = num_reqs * max_blocks + 8
    block_table = (
        torch.randperm(num_blocks)[: num_reqs * max_blocks]
        .view(num_reqs, max_blocks)
        .to(torch.int32)
    )
    cache = (torch.randn(num_blocks, bs, HEAD) * 2).to(torch.float8_e4m3fn)
    req_id, topk, counts = [], [], []
    for i, (sl, ql) in enumerate(zip(seq_lens, q_lens)):
        for j in range(ql):
            ctx = sl - ql + j + 1
            n = ctx if ctx <= topk_n else topk_n + ctx % kpool
            row = torch.full((width,), -1, dtype=torch.int32)
            row[:n] = torch.randperm(ctx)[:n].to(torch.int32)
            topk.append(row)
            req_id.append(i)
            counts.append(n)
    rows = len(req_id)
    req_id_t = torch.tensor(req_id, dtype=torch.int32, device=dev)
    topk_t = torch.stack(topk).to(dev)
    counts_t = torch.tensor(counts, dtype=torch.int32)

    def make_state(kv_dtype, share=None):
        return sm90_mod._SM90State(
            dev,
            heads,
            kv_dtype,
            rows,
            width,
            HEAD,
            0,
            HEAD**-0.5,
            topk_n,
            kpool,
            float_workspace=None if share is None else share.workspace,
            kv_indices=None if share is None else share.kv_indices,
        )

    impl = object.__new__(FlashInferMLASparseSM90Impl)
    impl.num_heads = heads
    impl.head_size = HEAD
    impl.scale = HEAD**-0.5
    impl.kv_lora_rank = HEAD
    impl.qk_rope_head_dim = 0
    impl.kv_cache_dtype = "fp8"
    impl.use_fp8_kv_cache = True
    impl.topk_indices_buffer = topk_t
    impl.prefill_ws_tokens = prefill_ctx + 8192
    ws_buf = torch.empty(impl.prefill_ws_tokens, HEAD, dtype=torch.bfloat16, device=dev)
    monkeypatch.setattr(impl, "_prefill_workspace", lambda: ws_buf)
    layer = SimpleNamespace(_k_scale_float=1.0, _k_scale=torch.tensor(1.0, device=dev))
    q_nope = torch.randn(rows, heads, HEAD, device=dev, dtype=torch.bfloat16)
    q_rope = q_nope.new_empty(rows, heads, 0)
    cache_dev = cache.to(dev)

    fp8_all = make_state(torch.float8_e4m3fn)
    fp8_all.plan(rows, counts_t, None, None)
    meta = SimpleNamespace(
        req_id_per_token=req_id_t,
        block_table=block_table.to(dev),
        block_size=bs,
        state=fp8_all,
    )
    ref_out, _ = impl.forward_mqa((q_nope, q_rope), cache_dev, meta, layer)

    nd = num_decodes
    qsl = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
    layout = plan_prefill_workspace(
        qsl, torch.tensor([prefill_ctx], dtype=torch.int32), nd, impl.prefill_ws_tokens
    )
    assert layout is not None
    dec = make_state(torch.float8_e4m3fn)
    dec.plan(nd, counts_t[:nd], None, None)
    pre = make_state(torch.bfloat16, share=dec)
    pre.plan(rows - nd, counts_t[nd:], None, None)
    ws_meta = SimpleNamespace(
        req_id_per_token=req_id_t,
        block_table=meta.block_table,
        block_size=bs,
        state=dec,
        prefill_state=pre,
        prefill_ws=sm90_mod.PrefillWorkspaceMetadata(
            num_decode_tokens=nd,
            num_tokens=layout.num_tokens,
            cu_seq_lens=layout.cu_seq_lens.to(dev),
            request_ids=layout.request_ids.to(dev),
            block_table=meta.block_table[nd:],
        ),
    )
    out, _ = impl.forward_mqa((q_nope, q_rope), cache_dev, ws_meta, layer)
    assert not out.isnan().any()
    # Decode rows: same FP8 kernel, but a decode-only plan may split KV
    # differently than the all-rows plan.
    if nd:
        torch.testing.assert_close(out[:nd], ref_out[:nd], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out[nd:], ref_out[nd:], atol=1e-2, rtol=1e-2)
