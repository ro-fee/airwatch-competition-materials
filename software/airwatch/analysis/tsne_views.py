"""Deterministic feature embedding; no Qt, files, or accuracy claims."""
from dataclasses import dataclass
import numpy as np
from sklearn.manifold import TSNE


@dataclass(frozen=True)
class TsneInput:
    features: np.ndarray
    file_ids: np.ndarray
    file_names: tuple


@dataclass(frozen=True)
class TsneView:
    coordinates: np.ndarray
    file_ids: np.ndarray
    file_names: tuple
    perplexity: float
    seed: int


def prepare_tsne(features, file_names, row_counts):
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError('t-SNE 至少需要两个窗口的二维特征矩阵。')
    if matrix.shape[0] > 2000 or matrix.shape[1] > 4096:
        raise ValueError('本次可视化上限为 2000 个窗口、4096 维；请缩小输入范围后重新识别。')
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix) or not np.isfinite(matrix).all():
        raise ValueError('特征须为有限实数，不允许 NaN/Inf。')
    if np.all(matrix == matrix[0]):
        raise ValueError('所有窗口特征完全相同，无法形成有意义的 t-SNE 分布。')
    names = tuple(str(n) for n in file_names)
    if not names:
        raise ValueError('缺少输入文件信息，请重新识别。')
    counts = [len(matrix)] if len(names) == 1 else row_counts
    if counts is None or len(counts) != len(names):
        raise ValueError('缺少文件与特征窗口映射，请重新识别。')
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in counts):
        raise ValueError('文件窗口数无效。')
    if sum(counts) != len(matrix):
        raise ValueError('文件窗口数与特征行数不一致，请重新识别。')
    snapshot = np.array(matrix, dtype=np.float64, copy=True)
    snapshot.setflags(write=False)
    ids = np.repeat(np.arange(len(names)), counts)
    ids.setflags(write=False)
    return TsneInput(snapshot, ids, names)


def compute_tsne(source):
    perplexity = min(30.0, (len(source.features) - 1) / 3.0)
    # Explicit initialization, seed and perplexity work with small sample counts.
    # No normalization: this is a view of the supplied model features.
    coordinates = TSNE(n_components=2, perplexity=perplexity, init='random',
                       random_state=42, learning_rate='auto', n_jobs=1).fit_transform(source.features)
    if not np.isfinite(coordinates).all():
        raise ValueError('t-SNE 返回了无效坐标。')
    coordinates.setflags(write=False)
    return TsneView(coordinates, source.file_ids, source.file_names, perplexity, 42)
