"""User-facing frozen application bootstrap with visible startup diagnostics."""
from __future__ import annotations

import ctypes
import os
import sys
import traceback
from pathlib import Path


def _writable_log_dir() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "AirWatch"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _show_error(title: str, message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
    except Exception:
        print(message, file=sys.stderr)


def main() -> int:
    try:
        if "--runtime-smoke" in sys.argv:
            from airwatch.runtime_smoke import write_runtime_model_smoke_report

            output = os.environ.get("AIRWATCH_SMOKE_REPORT")
            if not output:
                raise ValueError("AIRWATCH_SMOKE_REPORT is required for --runtime-smoke")
            write_runtime_model_smoke_report(output)
            return 0
        if "--ui-stress" in sys.argv:
            # Explicit acceptance-only mode. The same Qt driver is exercised
            # in source and frozen builds; ordinary startup never imports it.
            from importlib import import_module

            sys.argv.remove("--ui-stress")
            try:
                import_module("tools.ui_stress_session")
            except SystemExit as result:
                return int(result.code or 0)
            return 0
        from main import main as application_main
        return int(application_main() or 0)
    except BaseException as exc:
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log_path = _writable_log_dir() / "startup_error.log"
        log_path.write_text(detail, encoding="utf-8")
        _show_error(
            "空域电波哨兵启动失败",
            "软件没有成功启动。\n\n"
            f"详细错误已保存到：\n{log_path}\n\n"
            f"错误摘要：{type(exc).__name__}: {exc}",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
