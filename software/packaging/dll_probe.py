import ctypes, os, sys, traceback
from pathlib import Path
root=Path(getattr(sys,"_MEIPASS",Path(sys.executable).parent)); lib=root/"torch"/"lib"
os.environ["PATH"]=str(lib)+os.pathsep+os.environ.get("PATH","")
if hasattr(os,"add_dll_directory"): os.add_dll_directory(str(lib))
print("ROOT",root,flush=True)
for name in ["msvcp140.dll","vcruntime140.dll","vcruntime140_1.dll","vcruntime140_threads.dll","c10.dll","c10_cuda.dll","torch_cpu.dll"]:
 p=lib/name
 try:
  ctypes.WinDLL(str(p)); print("OK",name,flush=True)
 except BaseException as e:
  print("FAIL",name,repr(e),flush=True); traceback.print_exc()
input("Press Enter...")
