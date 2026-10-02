"""Feature-vector selection and index-domain spectra; no Qt or physical Hz claims."""
from dataclasses import dataclass
import numpy as np
from .signal_transforms import SignalViewError, SpectrogramView, compute_spectrogram


@dataclass(frozen=True)
class FeatureVectorView:
    values: np.ndarray
    row_index: int
    rows_in_file: int


def select_feature_vector(data, *, file_index=0, file_count=1, row_counts=None):
    """Choose the first window for a file; never confuse file and flattened row IDs."""
    matrix = np.asarray(data)
    if matrix.ndim != 2 or not matrix.size:
        raise SignalViewError('需要非空的二维特征矩阵（窗口数 × 特征维数）。')
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix):
        raise SignalViewError('模型特征必须是实数矩阵。')
    if not np.isfinite(matrix).all():
        raise SignalViewError('特征含 NaN 或无穷值。')
    if file_count < 1:
        raise SignalViewError('没有对应的输入文件。')
    index = max(0, file_index)
    if index >= file_count:
        raise SignalViewError('文件选择超出范围。')
    if file_count == 1:
        start, count = 0, matrix.shape[0]
    else:
        if row_counts is None or len(row_counts) != file_count:
            raise SignalViewError('缺少文件与特征窗口的对应关系，请重新识别。')
        if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in row_counts):
            raise SignalViewError('特征窗口数量无效。')
        if sum(row_counts) != matrix.shape[0]:
            raise SignalViewError('特征窗口数量与文件对应关系不一致，请重新识别。')
        start, count = sum(row_counts[:index]), row_counts[index]
    values = np.asarray(matrix[start], dtype=float).copy()
    values.setflags(write=False)
    return FeatureVectorView(values, start, count)


def compute_feature_spectrogram(view: FeatureVectorView, window: int) -> SpectrogramView:
    """Hann/50%-overlap PSD along feature index; one unit is one feature, not second."""
    return compute_spectrogram(view.values, sample_rate_hz=1, nperseg=window)
