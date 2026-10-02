"""Make bundled PyTorch DLL dependencies discoverable before torch import."""
import os
import sys
from pathlib import Path

root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
torch_lib = root / "torch" / "lib"
if torch_lib.is_dir():
    # Python 3.8+ uses explicit DLL directories on Windows. Keep PATH too
    # because some native extensions resolve secondary dependencies that way.
    try:
        os.add_dll_directory(str(torch_lib))
    except (AttributeError, OSError):
        pass
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")

# PyTorch's own loader can report c10.dll even when a later dependency is the
# actual initializer failure. Preload the core libraries in dependency order.
if torch_lib.is_dir():
    import ctypes
    for _name in ("c10.dll", "c10_cuda.dll", "torch_cpu.dll", "torch_cuda.dll"):
        _path = torch_lib / _name
        if _path.is_file():
            try:
                ctypes.WinDLL(str(_path))
            except OSError:
                pass
