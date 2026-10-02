"""Conservative input-quality checks for UAV signal previews.

These checks are intake gates, not model confidence and not a calibrated SNR
estimator. Thresholds are explicit and results are descriptive.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np

class UAVQualityStatus(str, Enum):
    ACCEPTED = "accepted"
    CAUTION = "caution"
    REJECTED = "rejected"

@dataclass(frozen=True)
class UAVQualityReport:
    status: UAVQualityStatus
    message: str
    sample_count: int
    channels: int
    peak_abs: float
    rms: float
    clipped_fraction: float
    finite: bool
    has_signal_variation: bool


def assess_uav_signal(samples, *, clip_fraction_limit=0.01, min_samples=16):
    array=np.asarray(samples)
    if array.ndim not in (1,2) or (array.ndim==2 and array.shape[0] not in (1,2)):
        raise ValueError('质量检查只支持采样点数组或通道×采样点数组')
    channels=1 if array.ndim==1 else array.shape[0]
    count=int(array.shape[-1])
    if count == 0:
        return UAVQualityReport(UAVQualityStatus.REJECTED,'信号为空，无法继续分析。',0,channels,0.,0.,0.,True,False)
    finite=bool(np.isfinite(array).all())
    if not finite:
        return UAVQualityReport(UAVQualityStatus.REJECTED,'信号包含 NaN 或无穷值，请检查文件。',count,channels,float('nan'),float('nan'),0.,False,False)
    values=np.asarray(array,dtype=np.float64)
    peak=float(np.max(np.abs(values)))
    rms=float(np.sqrt(np.mean(values*values)))
    variation=bool(np.ptp(values)>0)
    # Saturation is only flagged for normalized/full-scale-like data; it is not
    # inferred as clipping for arbitrary physical units.
    full_scale=peak > 0 and peak <= 1.000001
    clipped=float(np.mean(np.isclose(np.abs(values),1.0,rtol=0,atol=1e-6))) if full_scale else 0.
    if count < min_samples:
        status,msg=UAVQualityStatus.REJECTED,'信号过短，至少需要 16 个采样点。'
    elif not variation or rms == 0:
        status,msg=UAVQualityStatus.REJECTED,'信号没有可分析的幅值变化。'
    elif clipped > clip_fraction_limit:
        status,msg=UAVQualityStatus.CAUTION,f'疑似幅值饱和（约 {clipped:.1%}），结果需谨慎解释。'
    else:
        status,msg=UAVQualityStatus.ACCEPTED,'基础检查通过，可继续进行图形分析；这不是模型置信度。'
    return UAVQualityReport(status,msg,count,channels,peak,rms,clipped,finite,variation)

__all__=['UAVQualityStatus','UAVQualityReport','assess_uav_signal']
