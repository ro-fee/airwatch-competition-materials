"""Lazy exports for competition model definitions."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BearingCNN": (".bearing_cnn", "BearingCNN"),
    "DroneRFCNN": (".uav_cnn", "DroneRFCNN"),
    "DroneRFResNet18": (".uav_baselines", "DroneRFResNet18"),
    "DroneRFTCN": (".uav_baselines", "DroneRFTCN"),
    "DroneRFDualBranch": (".uav_dual_branch", "DroneRFDualBranch"),
    "DroneRFSpectralBackbone": (".uav_dual_branch", "DroneRFSpectralBackbone"),
    "DroneRFSpectralCNN": (".uav_dual_branch", "DroneRFSpectralCNN"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
