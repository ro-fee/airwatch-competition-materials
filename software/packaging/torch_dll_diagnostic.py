import os, sys, traceback
from pathlib import Path
print('=== AirWatch PyTorch DLL diagnostic ===', flush=True)
print('exe:', sys.executable, flush=True)
print('_MEIPASS:', getattr(sys, '_MEIPASS', None), flush=True)
root=Path(getattr(sys, '_MEIPASS', Path(sys.executable).resolve().parent))
lib=root/'torch'/'lib'
print('torch_lib:', lib, 'exists=', lib.is_dir(), flush=True)
for name in ('c10.dll','c10_cuda.dll','torch_cpu.dll','torch_cuda.dll','msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll','vcruntime140_threads.dll'):
 p=lib/name
 print(name, 'exists=', p.is_file(), 'size=', p.stat().st_size if p.is_file() else None, flush=True)
if hasattr(os,'add_dll_directory') and lib.is_dir():
 try:
  os.add_dll_directory(str(lib)); print('add_dll_directory: OK', flush=True)
 except Exception as e: print('add_dll_directory ERROR:', repr(e), flush=True)
os.environ['PATH']=str(lib)+os.pathsep+os.environ.get('PATH','')
print('PATH prepended', flush=True)
try:
 import torch
 print('torch import: OK', torch.__version__, flush=True)
 print('torch cuda compiled:', torch.version.cuda, flush=True)
 print('cuda available:', torch.cuda.is_available(), flush=True)
except BaseException as e:
 print('torch import: FAILED', repr(e), flush=True)
 traceback.print_exc()
 input('Press Enter to close...')
 raise
input('Torch loaded. Press Enter to close...')
