from pathlib import Path
root=Path(SPECPATH).parent
vc=Path(r"E:/CondaEnvs/rofee-bearing")
torch=Path(r"E:/CondaEnvs/rofee-bearing/Lib/site-packages/torch/lib")
binaries=[(str(torch/n),"torch/lib") for n in [p.name for p in torch.glob("*.dll")]]+[(str(vc/n),"torch/lib") for n in ["msvcp140.dll","vcruntime140.dll","vcruntime140_1.dll","vcruntime140_threads.dll"]]
a=Analysis([str(root/"packaging/dll_probe.py")],pathex=[str(root)],binaries=binaries,datas=[],hiddenimports=[],runtime_hooks=[],hookspath=[],hooksconfig={},excludes=[]); pyz=PYZ(a.pure); exe=EXE(pyz,a.scripts,a.binaries,a.datas,name="AirWatchDllProbe",console=True);