"""Read-only, amplitude-preserving home preview input (not inference input)."""
from pathlib import Path
import numpy as np


def load_home_signal(path):
    """Validate completely before the caller publishes a new file identity.

    Accept real/complex vectors, legacy real [1,N], and real [2,N]/[N,2]
    IQ matrices. Plotting accepts zero and short signals; expert analyses apply
    their own stricter length/real-only contracts later.
    """
    source = Path(path)
    if source.suffix.lower() != '.npy':
        raise ValueError('主页仅支持 NumPy .npy 信号文件。')
    array = np.load(source, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype.kind not in 'iufc':
        raise ValueError('信号必须是实数或复数采样数组。')
    valid_shape = array.ndim == 1 or (
        array.ndim == 2 and not np.iscomplexobj(array)
        and (array.shape[0] in (1, 2) or array.shape[1] == 2)
    )
    if not valid_shape:
        raise ValueError('信号形状须为 [N]、[1,N]、[2,N] 或 [N,2]；复数 IQ 须为一维。')
    if not array.size or not np.isfinite(array).all():
        raise ValueError('信号不能为空或含 NaN/Inf；本次文件未载入。')
    array.setflags(write=False)
    return array
