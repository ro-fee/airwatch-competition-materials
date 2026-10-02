"""Read-only historical inference workflow; no Qt dependency or training logic."""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch
from airwatch.general_recognition_contract import TASKS, task_spec
from airwatch.data.recognition_input import RecognitionSnapshot

CLASSES = {name: spec.classes for name, spec in TASKS.items()}

@dataclass(frozen=True)
class RecognitionResult:
    task_name: str
    filenames: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    window_predictions: tuple[tuple[int, ...], ...]
    features: np.ndarray
    feature_row_counts: tuple[int, ...]
    checkpoint_name: str
    elapsed_seconds: float
    has_ground_truth: bool
    discarded_samples: tuple[int, ...] = ()
    model_name: str = ''
    repeat_count: int = 1
    repeat_labels: tuple[tuple[str, ...], ...] = ()
    def __post_init__(self):
        features = np.asarray(self.features, dtype=np.float32)
        object.__setattr__(self, 'features', np.frombuffer(features.tobytes(), dtype=np.float32).reshape(features.shape))

def predict_one(model_info, snapshot: RecognitionSnapshot, cancelled, progress=None, *, batch_size=32):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError('批大小必须为正整数')
    def check():
        if cancelled.is_set():
            raise InterruptedError('识别已取消，不保留部分结果')
    check()
    model, checkpoint = model_info
    spec = task_spec(snapshot.task_name)
    if snapshot.window_size != spec.window_size:
        raise ValueError('窗口长度与历史模型输入契约不一致')
    arrays = snapshot.data if isinstance(snapshot.data, tuple) else (snapshot.data,)
    counts = [raw.shape[-1] // spec.window_size for raw in arrays]
    for name, raw in zip(snapshot.filenames, arrays):
        channels = 1 if raw.ndim == 1 else raw.shape[0]
        if channels != spec.channels:
            raise ValueError(f'{Path(name).name}：所选任务需要 {spec.channels} 通道，请重新按该任务导入')
    one_pass_total = sum(counts)
    total = one_pass_total * snapshot.repeat_count
    if progress:
        progress(0, total)
    device = next(model.parameters()).device
    started = time.perf_counter()
    all_features, all_predictions, repeat_labels = [], [], []
    done = 0
    with torch.inference_mode():
        for repeat_index in range(snapshot.repeat_count):
            pass_features = [] if repeat_index == 0 else None
            pass_predictions, pass_labels = [], []
            for file_index, (raw, count) in enumerate(zip(arrays, counts)):
                check()
                data = raw.reshape(spec.channels, -1)
                predictions = []
                for start in range(0, count, batch_size):
                    check()
                    stop = min(count, start + batch_size)
                    windows = data[:, start * spec.window_size:stop * spec.window_size]
                    windows = windows.reshape(spec.channels, stop - start, spec.window_size).transpose(1, 0, 2)
                    batch = torch.from_numpy(np.array(windows, dtype=np.float32, copy=True)).to(device)
                    feature, logits = model(batch)
                    if not torch.isfinite(feature).all() or not torch.isfinite(logits).all():
                        raise ValueError(f'{Path(snapshot.filenames[file_index]).name}：模型产生非有限值')
                    if logits.shape != (stop - start, len(spec.classes)):
                        raise ValueError('模型输出类别数量与标签映射不一致')
                    features = feature.detach().cpu().numpy()
                    if features.ndim != 2 or features.shape[0] != stop - start:
                        raise ValueError('模型特征形状不符合窗口×特征约定')
                    if pass_features is not None:
                        pass_features.append(features)
                    predictions.extend(logits.argmax(1).cpu().tolist())
                    done += stop - start
                    if progress:
                        progress(done, total)
                pass_predictions.append(tuple(predictions))
                winner = int(np.bincount(predictions, minlength=len(spec.classes)).argmax())
                pass_labels.append(spec.classes[winner])
            # Keep the first pass feature map for existing feature-view semantics.
            if repeat_index == 0:
                all_features = pass_features
                all_predictions = pass_predictions
            repeat_labels.append(tuple(pass_labels))
    check()
    final_labels = []
    for file_index in range(len(arrays)):
        votes = [spec.classes.index(labels[file_index]) for labels in repeat_labels]
        final_labels.append(spec.classes[int(np.bincount(votes, minlength=len(spec.classes)).argmax())])
    return RecognitionResult(
        snapshot.task_name, snapshot.filenames, tuple(final_labels), tuple(all_predictions),
        np.concatenate(all_features), tuple(counts), Path(checkpoint).name,
        time.perf_counter() - started, bool(snapshot.labels),
        tuple(raw.shape[-1] % spec.window_size for raw in arrays), type(model).__name__,
        snapshot.repeat_count, tuple(repeat_labels))
