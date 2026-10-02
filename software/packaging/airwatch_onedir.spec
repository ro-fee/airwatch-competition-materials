# PyInstaller onedir specification. Build with the dedicated environment.
from pathlib import Path
import sys
root = Path(SPECPATH).parent
sys.path.insert(0, str(root))
from airwatch.runtime_resources import pyinstaller_datas
vc = Path(r"E:/CondaEnvs/rofee-bearing")
binaries = [(str(vc / name), "torch/lib") for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll")]
datas = pyinstaller_datas(root, root / "build" / "runtime-resources.json")
# The bootstrap imports ``main`` dynamically. Everything below ``main`` is
# discovered from real imports; collecting every airwatch maintenance module
# would pull offline audit/materialization code into the desktop package.
hiddenimports = [
    "main",
    "tools.ui_stress_session",  # explicit --ui-stress acceptance mode only
    # Frozen inference implementations intentionally keep their historical
    # package-level imports because their SHA-256 hashes are contract-bound.
    # These are the two concrete model modules resolved by the lazy package
    # Interface at runtime.
    "airwatch.models.bearing_cnn",
    "airwatch.models.uav_baselines",
]
# Optional development/training packages are not used by any maintained
# desktop flow.  scikit-learn treats pandas as optional, and the application
# uses eager PyTorch inference only (not compile/export/vision pipelines).
excludes = [
    # PyInstaller appends this entry in-place when absent, which makes the
    # saved Analysis guts differ from the next spec evaluation and defeats
    # incremental builds with a false "excludes changed".
    "__main__",
    "pandas",
    "torchvision",
    "h5py",
    "pytest",
    "tensorboard",
    "IPython",
    "fsspec",
    "sympy",
    "jinja2",
    # OpenCV was reachable only through retired disk-image rendering code.
    # Maintained plots use pyqtgraph or in-memory Matplotlib renderers.
    "cv2",
]
a = Analysis([str(root / "packaging" / "airwatch_bootstrap.py")], pathex=[str(root)], binaries=binaries, datas=datas, hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[str(root / "packaging" / "torch_dll_path_hook.py")], excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AirWatch", debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=False, icon=str(root / "assets" / "airwatch.ico"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name="AirWatch")

