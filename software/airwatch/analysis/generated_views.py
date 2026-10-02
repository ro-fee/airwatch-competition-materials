"""Select a generated sample from the current session, without file IO or Qt."""
import numpy as np
from .signal_transforms import prepare_signal, SignalViewError


def select_generated_sample(batches, labels, class_name, sample_number):
    if not labels or len(batches) != len(labels):
        raise SignalViewError('请先生成信号；当前没有完整的生成样本。')
    if len(set(labels)) != len(labels):
        raise SignalViewError('生成类别标签重复。')
    if not class_name or class_name == '全选':
        class_name = labels[0]
    if class_name not in labels:
        raise SignalViewError('所选类别不在本次生成结果中，请重新选择。')
    if isinstance(sample_number, bool) or not isinstance(sample_number, (int, np.integer)):
        raise SignalViewError('样本序号必须是整数。')
    batch = np.asarray(batches[labels.index(class_name)])
    if batch.ndim != 3 or batch.shape[1] not in (1, 2):
        raise SignalViewError('生成样本形状须为 (样本数, 1或2通道, 采样点数)。')
    if not 1 <= sample_number <= batch.shape[0]:
        raise SignalViewError(f'样本序号应在 1–{batch.shape[0]} 之间。')
    sample = batch[sample_number - 1]
    if sample.shape[0] == 1:
        sample = sample[0]
    return class_name, prepare_signal(sample).samples
