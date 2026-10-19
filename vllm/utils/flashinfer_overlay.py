# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in FlashInfer header overlay for JIT-compiled kernels.

``VLLM_FLASHINFER_INCLUDE_OVERLAY=<dir>`` points FlashInfer's JIT at another
include root (a full ``include/`` tree containing ``flashinfer/``), for example a
patched checkout, without modifying the installed package. The overlay builds
into its own JIT workspace so modules compiled from it never share a cache entry
with the stock headers: ``VLLM_FLASHINFER_OVERLAY_WORKSPACE`` when set, else
``$VLLM_CACHE_ROOT/flashinfer_overlay/<sha256(overlay path)[:12]>``. The
workspace becomes ``FLASHINFER_WORKSPACE_BASE`` (inherited by child processes);
an explicitly set ``FLASHINFER_WORKSPACE_BASE`` must name that same directory.

Unset (the default), nothing is imported or changed.

The overlay must be applied before ``flashinfer`` is imported in the process
(FlashInfer reads its workspace once, at import); ``vllm.env_override`` applies it
on ``import vllm``. Logging is not configured that early, so the users of
FlashInfer kernels call ``log_active_overlay()`` once logging is up; it logs one
line per process naming the include root, the JIT directory and the overlay files
that differ from the stock headers.
"""

from __future__ import annotations

import filecmp
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

OVERLAY_ENV = "VLLM_FLASHINFER_INCLUDE_OVERLAY"
WORKSPACE_ENV = "VLLM_FLASHINFER_OVERLAY_WORKSPACE"
FLASHINFER_WORKSPACE_ENV = "FLASHINFER_WORKSPACE_BASE"


@dataclass(frozen=True)
class OverlayPlan:
    include_dir: Path
    workspace: Path


@dataclass(frozen=True)
class ActiveOverlay:
    include_dir: Path
    jit_dir: Path
    changed: tuple[str, ...]  # overlay files new or different from the stock headers
    aot_providers: bool  # prebuilt jit-cache packages bypass the overlay


_active: ActiveOverlay | None = None
_logged = False


def default_workspace(include_dir: Path, cache_root: str | os.PathLike) -> Path:
    digest = hashlib.sha256(str(include_dir).encode()).hexdigest()[:12]
    return Path(cache_root).expanduser() / "flashinfer_overlay" / digest


def resolve_overlay(environ: os._Environ | dict[str, str]) -> OverlayPlan | None:
    """Validate the overlay settings in ``environ``; None when the overlay is off."""
    raw = environ.get(OVERLAY_ENV, "")
    if not raw:
        return None
    include_dir = Path(raw).expanduser().resolve()
    if not (include_dir / "flashinfer").is_dir():
        raise ValueError(
            f"{OVERLAY_ENV}={raw} must be a FlashInfer include root "
            f"(a directory containing flashinfer/)"
        )
    if environ.get(WORKSPACE_ENV):
        workspace = Path(environ[WORKSPACE_ENV]).expanduser().resolve()
    else:
        import vllm.envs as envs

        workspace = default_workspace(include_dir, envs.VLLM_CACHE_ROOT).resolve()
    if workspace == Path.home().resolve():
        raise ValueError(
            f"the FlashInfer overlay workspace must not be $HOME (FlashInfer's "
            f"default JIT cache); set {WORKSPACE_ENV} to a dedicated directory"
        )
    explicit = environ.get(FLASHINFER_WORKSPACE_ENV)
    if explicit and Path(explicit).expanduser().resolve() != workspace:
        raise ValueError(
            f"{OVERLAY_ENV} is set but {FLASHINFER_WORKSPACE_ENV}={explicit} differs "
            f"from the overlay workspace {workspace}; unset one of them"
        )
    return OverlayPlan(include_dir, workspace)


def changed_headers(overlay: Path, stock: Path) -> list[str]:
    """Overlay files (relative paths) that are new or differ from ``stock``."""
    changed = []
    for path in sorted(overlay.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(overlay)
        ref = stock / rel
        if not ref.is_file() or not filecmp.cmp(path, ref, shallow=False):
            changed.append(rel.as_posix())
    return changed


def maybe_apply_flashinfer_include_overlay(
    environ: os._Environ | dict[str, str] | None = None,
) -> OverlayPlan | None:
    """Apply the overlay described by ``environ`` (default ``os.environ``)."""
    environ = os.environ if environ is None else environ
    plan = resolve_overlay(environ)
    if plan is None:
        return None
    jit_env_mod = sys.modules.get("flashinfer.jit.env")
    if jit_env_mod is not None:
        base = Path(jit_env_mod.FLASHINFER_BASE_DIR).resolve()
        if base != plan.workspace:
            raise RuntimeError(
                f"{OVERLAY_ENV} must be applied before flashinfer is imported "
                f"(flashinfer already uses the JIT workspace {base})"
            )
    environ[FLASHINFER_WORKSPACE_ENV] = str(plan.workspace)
    plan.workspace.mkdir(parents=True, exist_ok=True)

    import flashinfer.jit.env as jit_env

    stock = Path(jit_env.FLASHINFER_INCLUDE_DIR)
    if stock.resolve() != plan.include_dir:
        changed = changed_headers(plan.include_dir, stock)
        jit_env.FLASHINFER_INCLUDE_DIR = plan.include_dir
    else:
        changed = []
    global _active
    _active = ActiveOverlay(
        include_dir=plan.include_dir,
        jit_dir=Path(jit_env.FLASHINFER_JIT_DIR),
        changed=tuple(changed),
        aot_providers=bool(getattr(jit_env, "FLASHINFER_AOT_PROVIDERS", ())),
    )
    return plan


def active_overlay() -> ActiveOverlay | None:
    """The overlay applied in this process, if any."""
    return _active


def log_active_overlay() -> None:
    """Log the applied overlay once per process (no-op when the overlay is off)."""
    global _logged
    if _active is None or _logged:
        return
    _logged = True
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    logger.info(
        "FlashInfer include overlay active: include=%s jit_dir=%s changed=%s",
        _active.include_dir,
        _active.jit_dir,
        ",".join(_active.changed) or "none",
    )
    if _active.aot_providers:
        logger.warning(
            "FlashInfer AOT jit-cache packages are installed; modules they provide "
            "load prebuilt and ignore %s",
            OVERLAY_ENV,
        )
