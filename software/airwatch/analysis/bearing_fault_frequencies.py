"""Auditable theoretical bearing characteristic-frequency calculations.

Assumes a stationary outer race, rotating inner race, steady RPM, uniform
geometry, and no slip. These lines are references, not automatic diagnoses.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class BearingGeometry:
    roller_count: int
    roller_diameter_mm: float
    pitch_diameter_mm: float
    shaft_rpm: float
    contact_angle_deg: float = 0.0

@dataclass(frozen=True)
class BearingFrequencies:
    shaft_hz: float
    ftf_hz: float
    bpfo_hz: float
    bpfi_hz: float
    bsf_hz: float

def compute_characteristic_frequencies(geometry: BearingGeometry) -> BearingFrequencies:
    if isinstance(geometry.roller_count, bool) or not isinstance(geometry.roller_count, int) or geometry.roller_count <= 0:
        raise ValueError('滚动体数量必须是正整数（不能是小数）')
    for name in ('roller_diameter_mm','pitch_diameter_mm','shaft_rpm'):
        try:
            value = float(getattr(geometry, name))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} 必须是大于 0 的有限数值') from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} 必须是大于 0 的有限数值')
    if geometry.roller_diameter_mm >= geometry.pitch_diameter_mm:
        raise ValueError('滚动体直径必须小于节径')
    try:
        angle = float(geometry.contact_angle_deg)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('接触角必须是有限数值') from exc
    if not math.isfinite(angle) or not 0 <= angle < 90:
        raise ValueError('接触角必须在 0 到 90 度之间')
    fr = geometry.shaft_rpm / 60.0
    ratio = geometry.roller_diameter_mm / geometry.pitch_diameter_mm * math.cos(math.radians(angle))
    n = geometry.roller_count
    return BearingFrequencies(
        shaft_hz=fr,
        ftf_hz=0.5 * fr * (1-ratio),
        bpfo_hz=0.5 * n * fr * (1-ratio),
        bpfi_hz=0.5 * n * fr * (1+ratio),
        bsf_hz=geometry.pitch_diameter_mm/(2*geometry.roller_diameter_mm) * fr * (1-ratio**2),
    )

__all__=['BearingGeometry','BearingFrequencies','compute_characteristic_frequencies']
