"""Runtime paths shared by source runs and packaged desktop builds.

The application has two kinds of files:

* bundled resources are read-only and travel with the application (models,
  frozen contracts, demo data, and static images);
* user data must remain writable and must not be written into a PyInstaller
  bundle or an installed program directory (generated signals, plots, logs,
  and caches).

Keep this module free of Qt so it can be used by startup checks, tests, and
future packaging scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping


APP_DATA_NAME = "AirWatch"
DATA_ROOT_ENV = "AIRWATCH_DATA_DIR"


def is_frozen() -> bool:
    """Return whether the current process is running from a frozen bundle."""

    return bool(getattr(sys, "frozen", False))


def resource_root(*, frozen: bool | None = None, meipass: str | Path | None = None) -> Path:
    """Return the read-only root containing application resources."""

    if frozen is None:
        frozen = is_frozen()
    if frozen:
        candidate = meipass or getattr(sys, "_MEIPASS", None)
        if candidate:
            return Path(candidate).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _windows_data_base() -> Path:
    """Return the conventional per-user writable base directory."""

    value = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if value:
        return Path(value)
    return Path.home() / "AppData" / "Local"


def writable_root(*, frozen: bool | None = None, resource: str | Path | None = None, override: str | Path | None = None) -> Path:
    """Return the root for files the application is allowed to write.

    AIRWATCH_DATA_DIR is supported for portable demos, automated tests, and
    administrators who want an explicit data location. A packaged app
    otherwise writes under the per-user local application data directory.
    A source run keeps the repository-local output layout for compatibility.
    """

    if override is None:
        override = os.environ.get(DATA_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    if frozen is None:
        frozen = is_frozen()
    if resource is None:
        resource = resource_root(frozen=frozen)
    if frozen:
        return (_windows_data_base() / APP_DATA_NAME).resolve()
    return Path(resource).resolve()


def _safe_child(root: str | Path, *parts: str | Path) -> Path:
    """Resolve a relative child and reject paths escaping root."""

    base = Path(root).resolve()
    relative = Path(*parts)
    if relative.is_absolute():
        raise ValueError("runtime path parts must be relative")
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("runtime path must stay below its root") from exc
    return resolved


def resource_path(*parts: str | Path) -> Path:
    """Resolve a read-only file or directory shipped with the application."""

    return _safe_child(resource_root(), *parts)


def writable_path(*parts: str | Path) -> Path:
    """Resolve a file or directory under the writable application root."""

    return _safe_child(writable_root(), *parts)


def ensure_user_directories() -> Mapping[str, Path]:
    """Create the small set of writable directories used by the desktop app."""

    directories = {
        "root": writable_root(),
        "output": writable_path("result"),
        "generation": writable_path("result", "generation"),
        "comparison": writable_path("result", "comparison"),
        "logs": writable_path("logs"),
        "cache": writable_path("cache"),
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def runtime_info() -> dict[str, str | bool]:
    """Return human-readable paths for startup diagnostics and support logs."""

    return {
        "frozen": is_frozen(),
        "resource_root": str(resource_root()),
        "writable_root": str(writable_root()),
        "output_dir": str(writable_path("result")),
        "generation_dir": str(writable_path("result", "generation")),
        "comparison_dir": str(writable_path("result", "comparison")),
    }


__all__ = [
    "APP_DATA_NAME",
    "DATA_ROOT_ENV",
    "ensure_user_directories",
    "is_frozen",
    "resource_path",
    "resource_root",
    "runtime_info",
    "writable_path",
    "writable_root",
]
