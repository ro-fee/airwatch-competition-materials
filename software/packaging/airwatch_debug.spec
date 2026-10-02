# PyInstaller onedir specification. Build with the dedicated environment.
from pathlib import Path
import sys
root = Path(SPECPATH).parent
sys.path.insert(0, str(root))
from airwatch.runtime_resources import pyinstaller_datas
vc = Path(r"E:/CondaEnvs/rofee-bearing")
binaries = [(str(vc / name), "torch/lib") for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll")]
datas = pyinstaller_datas(root, root / "build" / "runtime-resources-debug.json")
hiddenimports = [
    "airwatch.models.bearing_cnn",
    "airwatch.models.uav_baselines",
]
excludes = [
    # Keep Analysis inputs stable across cached builds; PyInstaller otherwise
    # appends this entry in-place after the guts comparison.
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
    "cv2",
]
a = Analysis([str(root / "main.py")], pathex=[str(root)], binaries=binaries, datas=datas, hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[str(root / "packaging" / "torch_dll_path_hook.py")], excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AirWatchDebug", debug=True, bootloader_ignore_signals=False, strip=False, upx=True, console=True, icon=str(root / "assets" / "airwatch.ico"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name="AirWatchDebug")

