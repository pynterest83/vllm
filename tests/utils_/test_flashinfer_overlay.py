# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the opt-in FlashInfer include overlay (utils/flashinfer_overlay)."""

import json
import logging
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from vllm.utils import flashinfer_overlay as fo

REPO_ROOT = Path(__file__).resolve().parents[2]


def _include_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


@pytest.fixture
def stock(tmp_path):
    return _include_tree(
        tmp_path / "stock",
        {"flashinfer/attention/mla_hopper.cuh": "old", "flashinfer/vec.cuh": "same"},
    )


@pytest.fixture
def overlay(tmp_path):
    return _include_tree(
        tmp_path / "overlay",
        {"flashinfer/attention/mla_hopper.cuh": "new", "flashinfer/vec.cuh": "same"},
    )


def test_off_by_default():
    assert fo.resolve_overlay({}) is None
    assert fo.resolve_overlay({fo.OVERLAY_ENV: ""}) is None
    environ: dict[str, str] = {}
    assert fo.maybe_apply_flashinfer_include_overlay(environ) is None
    assert environ == {}


def test_rejects_non_include_root(tmp_path):
    with pytest.raises(ValueError, match="include root"):
        fo.resolve_overlay({fo.OVERLAY_ENV: str(tmp_path)})


def test_default_workspace_is_per_overlay_under_cache_root(
    overlay, tmp_path, monkeypatch
):
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    plan = fo.resolve_overlay({fo.OVERLAY_ENV: str(overlay)})
    assert plan.include_dir == overlay.resolve()
    assert (
        plan.workspace.parent == (tmp_path / "cache" / "flashinfer_overlay").resolve()
    )
    other = fo.default_workspace(tmp_path / "other", tmp_path / "cache")
    assert other != plan.workspace


def test_workspace_override_and_conflicts(overlay, tmp_path):
    ws = tmp_path / "ws"
    env = {fo.OVERLAY_ENV: str(overlay), fo.WORKSPACE_ENV: str(ws)}
    assert fo.resolve_overlay(env).workspace == ws.resolve()
    # FLASHINFER_WORKSPACE_BASE naming the same directory is accepted ...
    env[fo.FLASHINFER_WORKSPACE_ENV] = str(ws)
    assert fo.resolve_overlay(env).workspace == ws.resolve()
    # ... a different one, or the default ~/.cache/flashinfer location, is not.
    env[fo.FLASHINFER_WORKSPACE_ENV] = str(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="differs"):
        fo.resolve_overlay(env)
    with pytest.raises(ValueError, match="HOME"):
        fo.resolve_overlay(
            {fo.OVERLAY_ENV: str(overlay), fo.WORKSPACE_ENV: str(Path.home())}
        )


def test_changed_headers(overlay, stock):
    (overlay / "flashinfer" / "extra.cuh").write_text("x")
    assert fo.changed_headers(overlay, stock) == [
        "flashinfer/attention/mla_hopper.cuh",
        "flashinfer/extra.cuh",
    ]


def _stub_flashinfer(monkeypatch, stock: Path, base: Path | None):
    env = types.ModuleType("flashinfer.jit.env")
    env.FLASHINFER_INCLUDE_DIR = stock
    env.FLASHINFER_JIT_DIR = Path("/unused")
    env.FLASHINFER_AOT_PROVIDERS = ()
    if base is not None:
        env.FLASHINFER_BASE_DIR = base
    jit = types.ModuleType("flashinfer.jit")
    jit.env = env
    pkg = types.ModuleType("flashinfer")
    pkg.jit = jit
    for name, mod in (("flashinfer", pkg), ("flashinfer.jit", jit)):
        monkeypatch.setitem(sys.modules, name, mod)
    return env


def test_apply_sets_workspace_and_include(overlay, stock, tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    # flashinfer imported with the overlay workspace already in place (a child
    # process inherits FLASHINFER_WORKSPACE_BASE from its parent).
    env_mod = _stub_flashinfer(monkeypatch, stock, base=ws.resolve())
    monkeypatch.setitem(sys.modules, "flashinfer.jit.env", env_mod)
    environ = {fo.OVERLAY_ENV: str(overlay), fo.WORKSPACE_ENV: str(ws)}
    plan = fo.maybe_apply_flashinfer_include_overlay(environ)
    assert environ[fo.FLASHINFER_WORKSPACE_ENV] == str(ws.resolve())
    assert ws.is_dir()
    assert env_mod.FLASHINFER_INCLUDE_DIR == plan.include_dir == overlay.resolve()


def test_apply_after_flashinfer_import_with_other_workspace_fails(
    overlay, stock, tmp_path, monkeypatch
):
    env_mod = _stub_flashinfer(monkeypatch, stock, base=Path.home())
    monkeypatch.setitem(sys.modules, "flashinfer.jit.env", env_mod)
    environ = {fo.OVERLAY_ENV: str(overlay), fo.WORKSPACE_ENV: str(tmp_path / "ws")}
    with pytest.raises(RuntimeError, match="before flashinfer is imported"):
        fo.maybe_apply_flashinfer_include_overlay(environ)
    assert stock == env_mod.FLASHINFER_INCLUDE_DIR


_PROBE = (
    "import json, os, vllm, flashinfer.jit.env as e; "
    "from vllm.utils.flashinfer_overlay import active_overlay; a = active_overlay(); "
    "print(json.dumps({'include': str(e.FLASHINFER_INCLUDE_DIR), "
    "'jit': str(e.FLASHINFER_JIT_DIR), "
    "'base': os.environ.get('FLASHINFER_WORKSPACE_BASE'), "
    "'changed': list(a.changed) if a else None}))"
)


def _probe(extra_env: dict[str, str]) -> dict:
    pytest.importorskip("flashinfer")
    env = {k: v for k, v in os.environ.items() if not k.startswith("FLASHINFER_")}
    env.pop(fo.OVERLAY_ENV, None)
    env.pop(fo.WORKSPACE_ENV, None)
    env.update(extra_env, PYTHONPATH=str(REPO_ROOT), CUDA_VISIBLE_DEVICES="")
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_import_vllm_end_to_end(tmp_path):
    """``import vllm`` applies the overlay in a fresh process, and only when set."""
    stock = _probe({})
    assert stock["base"] is None and stock["changed"] is None
    assert Path(stock["include"]).name == "include"

    overlay = tmp_path / "include"
    shutil.copytree(stock["include"], overlay)
    mla = overlay / "flashinfer" / "attention" / "mla_hopper.cuh"
    mla.write_text(mla.read_text() + "\n// overlay\n")
    ws = tmp_path / "ws"
    got = _probe({fo.OVERLAY_ENV: str(overlay), fo.WORKSPACE_ENV: str(ws)})
    assert got["include"] == str(overlay.resolve())
    assert got["base"] == str(ws.resolve())
    assert got["jit"].startswith(str(ws.resolve()))
    assert got["changed"] == ["flashinfer/attention/mla_hopper.cuh"]


def test_log_active_overlay_once(overlay, stock, tmp_path, monkeypatch, caplog):
    ws = tmp_path / "ws"
    env_mod = _stub_flashinfer(monkeypatch, stock, base=ws.resolve())
    monkeypatch.setitem(sys.modules, "flashinfer.jit.env", env_mod)
    monkeypatch.setattr(fo, "_active", None)
    monkeypatch.setattr(fo, "_logged", False)
    fo.log_active_overlay()  # off: nothing
    fo.maybe_apply_flashinfer_include_overlay(
        {fo.OVERLAY_ENV: str(overlay), fo.WORKSPACE_ENV: str(ws)}
    )
    with caplog.at_level(logging.INFO):
        logging.getLogger("vllm").propagate = True
        fo.log_active_overlay()
        fo.log_active_overlay()
    lines = [
        r.getMessage() for r in caplog.records if "overlay active" in r.getMessage()
    ]
    assert len(lines) == 1
    assert "changed=flashinfer/attention/mla_hopper.cuh" in lines[0]
