from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules
root = Path(SPECPATH).parent
vc = Path(r"E:/CondaEnvs/rofee-bearing")
binaries = [(str(vc / name), "torch/lib") for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll")]
datas = []
hiddenimports = collect_submodules("torch")
a = Analysis([str(root / "packaging" / "torch_dll_diagnostic.py")], pathex=[str(root)], binaries=binaries, datas=datas, hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[str(root / "packaging" / "torch_dll_path_hook.py")], excludes=[], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, name="AirWatchTorchDiagnostic", debug=False, console=True, icon=str(root / "assets" / "airwatch.ico"))

