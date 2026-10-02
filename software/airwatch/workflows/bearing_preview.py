"""Read-only bearing plot preparation, independent of Qt and model execution."""
from dataclasses import dataclass
from pathlib import Path
from airwatch.data.cwru import load_cwru_signal
from airwatch.data.bearing_metadata import find_bearing_metadata, metadata_sensor_key
from airwatch.analysis.signal_transforms import compute_waveform, compute_spectrum, _validate_sample_rate

MAX_PREVIEW_SAMPLES = 65536


@dataclass(frozen=True)
class BearingPreview:
    path: str
    sensor_key: str
    total_samples: int
    shown_samples: int
    rate_source: str
    waveform: object
    spectrum: object


def load_bearing_preview(path, *, sample_rate_hz=None, manifest_path=None,
                         project_root=None, cancelled=lambda: False,
                         progress=lambda message: None):
    def check():
        if cancelled():
            raise InterruptedError('轴承图预览已取消')
    check()
    progress('核对通道和采样元数据…')
    if manifest_path is None:
        # Resolve paths through the same verified frozen contract as diagnosis.
        # This checks resources but never constructs a network or loads a checkpoint.
        from airwatch.inference.bearing_contract import load_frozen_bearing_runtime_contract
        runtime = load_frozen_bearing_runtime_contract()
        manifest_path, project_root = runtime.manifest_path, runtime.project_root
    row = find_bearing_metadata(manifest_path, project_root, path)
    sensor = metadata_sensor_key(row)
    rate = None
    rate_source = '未知采样率（仅采样点波形）'
    if row and row.get('sample_rate_hz'):
        rate = _validate_sample_rate(row['sample_rate_hz'])
        rate_source = '已审计清单'
        if sample_rate_hz is not None and _validate_sample_rate(sample_rate_hz) != rate:
            raise ValueError('输入采样率与已审计清单不一致；请清空手动采样率后重试。')
    elif sample_rate_hz is not None:
        rate = _validate_sample_rate(sample_rate_hz)
        rate_source = '用户填写（仅绘图，不改变诊断契约）'
    check()
    progress('读取所选 MAT 驱动端振动信号…')
    record = load_cwru_signal(path, sensor_key=sensor, normalization='none')
    check()
    samples = record.signal[:MAX_PREVIEW_SAMPLES]
    progress('计算波形与频谱预览…')
    waveform = compute_waveform(samples, sample_rate_hz=rate)
    check()
    spectrum = compute_spectrum(samples, sample_rate_hz=rate) if rate else None
    check()
    return BearingPreview(str(Path(path).resolve()), record.sensor_key,
                          len(record.signal), len(samples), rate_source, waveform, spectrum)
