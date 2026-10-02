"""Deterministic, controlled perturbations for two-channel complex IQ windows.

The functions in this module operate on one real tensor shaped ``[2, N]``,
where row zero is I and row one is Q.  They are intended for reproducible
validation-set robustness experiments; no dataset is opened by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset


class IQPerturbationError(ValueError):
    """Raised when an IQ perturbation request violates its numeric contract."""


# Frozen profiles are represented by exact (delay, relative amplitude dB)
# protocol fields. The public accessor converts them to positive-real base taps,
# then normalizes total tap energy before applying any seeded delayed-path phase.
_MULTIPATH_PROFILES: dict[str, tuple[tuple[int, float], ...]] = {
    "mild": (
        (0, 0.0),
        (5, -6.0),
        (12, -12.0),
    ),
    "severe": (
        (0, 0.0),
        (8, -3.0),
        (24, -6.0),
        (64, -9.0),
    ),
}

MULTIPATH_PROFILE_NAMES = tuple(_MULTIPATH_PROFILES)


def _profile_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for name, taps in _MULTIPATH_PROFILES.items():
        gains = [(delay, 10.0 ** (relative_db / 20.0)) for delay, relative_db in taps]
        energy = sum(gain**2 for _, gain in gains)
        normalized = [(delay, gain / math.sqrt(energy)) for delay, gain in gains]
        metadata[name] = {
            "normalization": "sum_squared_complex_gain_equals_one",
            "boundary_model": "independent_window_causal_zero_padding",
            "phase_policy": "direct_path_zero_delayed_paths_sha256_seeded",
            "channel_seed_fields": ["profile", "recording_id", "repeat_seed"],
            "taps": [
                {
                    "delay_samples": delay,
                    "normalized_gain_real": gain.real,
                    "normalized_gain_imag": 0.0,
                    "relative_amplitude_db": relative_db,
                }
                for (delay, gain), (_, relative_db) in zip(normalized, taps)
            ],
        }
    return metadata


# JSON-serializable evidence metadata.  The actual implementation continues to
# use the private immutable tuples above, so callers cannot mutate channel taps.
MULTIPATH_PROFILES = _profile_metadata()


def _validated_iq(signal: torch.Tensor) -> torch.Tensor:
    if not isinstance(signal, torch.Tensor):
        raise IQPerturbationError("signal must be a torch.Tensor")
    if signal.ndim != 2 or signal.shape[0] != 2 or signal.shape[1] < 2:
        raise IQPerturbationError(
            f"signal must have shape [2, N] with N >= 2, got {tuple(signal.shape)}"
        )
    if signal.dtype not in (torch.float32, torch.float64):
        raise IQPerturbationError("signal dtype must be torch.float32 or torch.float64")
    if not bool(torch.isfinite(signal).all().item()):
        raise IQPerturbationError("signal must contain only finite values")
    return signal


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise IQPerturbationError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IQPerturbationError(f"{name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise IQPerturbationError(f"{name} must be a finite number")
    return converted


def _validated_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise IQPerturbationError("seed must be an integer")
    return int(seed) % (2**63 - 1)


def complex_signal_power(signal: torch.Tensor) -> torch.Tensor:
    """Return mean complex power ``mean(I**2 + Q**2)`` as a scalar tensor."""

    values = _validated_iq(signal)
    return values.square().sum(dim=0).mean()


def normalize_complex_iq_window(
    signal: torch.Tensor, *, rms_epsilon: float = 1e-8
) -> torch.Tensor:
    """Remove per-window complex DC and normalize joint complex RMS to one."""

    values = _validated_iq(signal)
    epsilon = _finite_number(rms_epsilon, "rms_epsilon")
    if epsilon <= 0.0:
        raise IQPerturbationError("rms_epsilon must be greater than zero")
    mean = values.to(dtype=torch.float64).mean(dim=1, keepdim=True).to(values.dtype)
    centered = values - mean
    power = centered.to(dtype=torch.float64).square().sum(dim=0).mean()
    rms = torch.sqrt(power)
    if not bool(torch.isfinite(rms).item()) or rms.item() <= epsilon:
        raise IQPerturbationError("IQ window has insufficient energy after DC removal")
    result = centered / rms.to(device=values.device, dtype=values.dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise IQPerturbationError("IQ normalization produced non-finite values")
    return result


def add_complex_awgn(
    signal: torch.Tensor,
    *,
    snr_db: float,
    seed: int,
) -> torch.Tensor:
    """Add deterministic circular complex AWGN at an exact empirical SNR.

    The requested SNR is relative to the complete complex-window power
    ``mean(I**2 + Q**2)``.  Both noise components are sampled independently
    from the same Gaussian law and receive one common scale, preserving the
    circular complex-noise model.
    """

    values = _validated_iq(signal)
    requested_snr = _finite_number(snr_db, "snr_db")
    if not -100.0 <= requested_snr <= 100.0:
        raise IQPerturbationError("snr_db must be between -100 and 100")
    normalized_seed = _validated_seed(seed)

    # CPU float64 generation makes a seed independent of the input device and
    # gives the empirical power scaling enough precision before the final cast.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(normalized_seed)
    noise = torch.randn(tuple(values.shape), generator=generator, dtype=torch.float64)
    noise -= noise.mean(dim=1, keepdim=True)
    component_power = noise.square().mean(dim=1, keepdim=True)
    if (
        not bool(torch.isfinite(component_power).all().item())
        or bool(torch.any(component_power <= 0.0).item())
    ):
        raise IQPerturbationError("generated noise has invalid power")

    # One joint scale preserves rotational symmetry; separately forcing equal
    # empirical I/Q powers would distort the circular noise distribution.
    noise /= torch.sqrt(component_power.mean())

    work_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    work_signal = values.to(dtype=work_dtype)
    signal_power = work_signal.square().sum(dim=0).mean()
    if not bool(torch.isfinite(signal_power).item()) or signal_power.item() <= 0.0:
        raise IQPerturbationError("signal must have positive finite complex power")

    target_noise_power = signal_power / (10.0 ** (requested_snr / 10.0))
    scaled_noise = noise.to(device=values.device, dtype=work_dtype)
    scaled_noise *= torch.sqrt(target_noise_power / 2.0)
    result = work_signal + scaled_noise
    if not bool(torch.isfinite(result).all().item()):
        raise IQPerturbationError("AWGN perturbation produced non-finite values")
    return result.to(dtype=values.dtype)


def apply_carrier_frequency_offset(
    signal: torch.Tensor,
    *,
    offset_hz: float,
    sample_rate_hz: float,
    start_sample: int = 0,
) -> torch.Tensor:
    """Apply a CFO rotation using the recording-absolute sample position."""

    values = _validated_iq(signal)
    offset = _finite_number(offset_hz, "offset_hz")
    sample_rate = _finite_number(sample_rate_hz, "sample_rate_hz")
    if sample_rate <= 0.0:
        raise IQPerturbationError("sample_rate_hz must be greater than zero")
    if isinstance(start_sample, bool) or not isinstance(start_sample, Integral):
        raise IQPerturbationError("start_sample must be an integer")
    absolute_start = int(start_sample)
    if absolute_start < 0:
        raise IQPerturbationError("start_sample must be non-negative")

    # Build recording-absolute phase in float64 even for float32 IQ. This avoids
    # losing single-sample phase increments at large recording offsets.
    indices = torch.arange(values.shape[1], device=values.device, dtype=torch.float64)
    indices += absolute_start
    phase = indices * (2.0 * math.pi * offset / sample_rate)
    cosine = torch.cos(phase).to(dtype=values.dtype)
    sine = torch.sin(phase).to(dtype=values.dtype)
    in_phase, quadrature = values[0], values[1]
    result = torch.stack(
        (
            in_phase * cosine - quadrature * sine,
            in_phase * sine + quadrature * cosine,
        )
    )
    if not bool(torch.isfinite(result).all().item()):
        raise IQPerturbationError("carrier-frequency offset produced non-finite values")
    return result


def multipath_profile_taps(
    profile: str,
    *,
    dtype: torch.dtype = torch.complex128,
    device: torch.device | str | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Return one fixed dense multipath impulse response with unit energy."""

    if profile not in _MULTIPATH_PROFILES:
        names = ", ".join(MULTIPATH_PROFILE_NAMES)
        raise IQPerturbationError(f"unknown multipath profile {profile!r}; expected: {names}")
    if dtype not in (torch.complex64, torch.complex128):
        raise IQPerturbationError("multipath tap dtype must be complex64 or complex128")
    protocol_taps = _MULTIPATH_PROFILES[profile]
    sparse_taps = tuple(
        (delay, 10.0 ** (relative_db / 20.0))
        for delay, relative_db in protocol_taps
    )
    dense = torch.zeros(sparse_taps[-1][0] + 1, dtype=dtype, device=device)
    for delay, gain in sparse_taps:
        dense[delay] = gain
    if seed is not None:
        normalized_seed = _validated_seed(seed)
        # Derive each delayed-path phase directly from SHA-256 rather than a
        # library RNG, making the channel definition stable across processes.
        for delay, _ in sparse_taps:
            if delay == 0:
                continue
            payload = f"tap-phase\0{normalized_seed}\0{delay}".encode("utf-8")
            phase_code = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            phase = 2.0 * math.pi * phase_code / float(2**64)
            rotation = complex(math.cos(phase), math.sin(phase))
            dense[delay] *= rotation
    energy = dense.abs().square().sum()
    if not bool(torch.isfinite(energy).item()) or energy.item() <= 0.0:
        raise IQPerturbationError(f"multipath profile {profile!r} has invalid energy")
    return dense / torch.sqrt(energy)


def apply_multipath_channel(
    signal: torch.Tensor, *, profile: str, channel_seed: int | None = None
) -> torch.Tensor:
    """Apply a unit-energy causal channel and keep the original length.

    Each window is convolved independently, using zero samples before its first
    element. Consequently, delayed energy from the preceding real recording
    window is unavailable: this is an explicit window-boundary approximation.
    A recording-stable ``channel_seed`` rotates path phases without changing the
    registered delays or amplitudes.
    """

    values = _validated_iq(signal)
    complex_dtype = torch.complex128 if values.dtype == torch.float64 else torch.complex64
    taps = multipath_profile_taps(
        profile, dtype=complex_dtype, device=values.device, seed=channel_seed
    )
    iq = torch.complex(values[0], values[1])
    result = torch.zeros_like(iq)
    sample_count = iq.shape[0]
    for delay, tap in enumerate(taps):
        if delay >= sample_count:
            break
        result[delay:] += tap * iq[: sample_count - delay]
    stacked = torch.stack((result.real, result.imag)).to(dtype=values.dtype)
    if not bool(torch.isfinite(stacked).all().item()):
        raise IQPerturbationError("multipath perturbation produced non-finite values")
    return stacked


@dataclass(frozen=True)
class IQPerturbationSpec:
    """Flat, JSON-compatible description of exactly one perturbation kind."""

    kind: str
    snr_db: float | None = None
    offset_hz: float | None = None
    sample_rate_hz: float | None = None
    profile: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"awgn", "carrier_frequency_offset", "multipath"}:
            raise IQPerturbationError(
                "kind must be 'awgn', 'carrier_frequency_offset', or 'multipath'"
            )
        parameters = {
            "snr_db": self.snr_db,
            "offset_hz": self.offset_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "profile": self.profile,
        }
        required = {
            "awgn": {"snr_db"},
            "carrier_frequency_offset": {"offset_hz", "sample_rate_hz"},
            "multipath": {"profile"},
        }[self.kind]
        supplied = {name for name, value in parameters.items() if value is not None}
        if supplied != required:
            raise IQPerturbationError(
                f"{self.kind} requires exactly these parameters: {sorted(required)}"
            )
        if self.kind == "awgn":
            value = _finite_number(self.snr_db, "snr_db")
            if not -100.0 <= value <= 100.0:
                raise IQPerturbationError("snr_db must be between -100 and 100")
            object.__setattr__(self, "snr_db", value)
        elif self.kind == "carrier_frequency_offset":
            offset = _finite_number(self.offset_hz, "offset_hz")
            sample_rate = _finite_number(self.sample_rate_hz, "sample_rate_hz")
            if sample_rate <= 0.0:
                raise IQPerturbationError("sample_rate_hz must be greater than zero")
            object.__setattr__(self, "offset_hz", offset)
            object.__setattr__(self, "sample_rate_hz", sample_rate)
        else:
            if self.profile not in _MULTIPATH_PROFILES:
                names = ", ".join(MULTIPATH_PROFILE_NAMES)
                raise IQPerturbationError(
                    f"unknown multipath profile {self.profile!r}; expected: {names}"
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "IQPerturbationSpec":
        """Construct a validated spec from flat JSON-decoded fields."""

        if not isinstance(payload, Mapping):
            raise IQPerturbationError("perturbation spec must be a mapping")
        try:
            return cls(**dict(payload))
        except TypeError as exc:
            raise IQPerturbationError(f"invalid perturbation spec fields: {exc}") from exc

    def as_dict(self) -> dict[str, str | float]:
        payload: dict[str, str | float] = {"kind": self.kind}
        for name in ("snr_db", "offset_hz", "sample_rate_hz", "profile"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


def sample_iq_perturbation_seed(base_seed: int, sample_index: int) -> int:
    """Mix a base seed and stable dataset index independently of loader order."""

    normalized_base = _validated_seed(base_seed)
    if isinstance(sample_index, bool) or not isinstance(sample_index, Integral):
        raise IQPerturbationError("sample_index must be an integer")
    normalized_index = int(sample_index)
    if normalized_index < 0:
        raise IQPerturbationError("sample_index must be non-negative")
    payload = f"sample\0{normalized_base}\0{normalized_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def window_iq_perturbation_seed(base_seed: int, window_id: str) -> int:
    """Return a stable AWGN seed from repeat seed and immutable window id.

    No SNR value is included, so a repeated window uses one noise direction at
    every SNR; only the deterministic power scale changes.
    """

    normalized_base = _validated_seed(base_seed)
    if not isinstance(window_id, str) or not window_id.strip():
        raise IQPerturbationError("window_id must be a non-empty string")
    payload = f"window\0{normalized_base}\0{window_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def recording_iq_perturbation_seed(
    base_seed: int, recording_id: str, profile: str
) -> int:
    """Return a stable channel seed shared by every window of one recording."""

    normalized_base = _validated_seed(base_seed)
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise IQPerturbationError("recording_id must be a non-empty string")
    if profile not in _MULTIPATH_PROFILES:
        raise IQPerturbationError(f"unknown multipath profile {profile!r}")
    payload = f"multipath\0{profile}\0{recording_id}\0{normalized_base}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def apply_iq_perturbation(
    signal: torch.Tensor,
    spec: IQPerturbationSpec | Mapping[str, Any],
    *,
    seed: int = 0,
    start_sample: int = 0,
    channel_seed: int | None = None,
    postprocess: bool = True,
) -> torch.Tensor:
    """Apply one effect, then repeat per-window complex DC/RMS preprocessing.

    ``postprocess=False`` exists for numeric audits of the injected impairment,
    such as measuring AWGN power before the mandatory normalization stage.
    Dataset-backed evaluations always keep the default ``True`` value.
    """

    if not isinstance(postprocess, bool):
        raise IQPerturbationError("postprocess must be a boolean")
    normalized = (
        spec
        if isinstance(spec, IQPerturbationSpec)
        else IQPerturbationSpec.from_mapping(spec)
    )
    if normalized.kind == "awgn":
        assert normalized.snr_db is not None
        result = add_complex_awgn(signal, snr_db=normalized.snr_db, seed=seed)
    elif normalized.kind == "carrier_frequency_offset":
        assert (
            normalized.offset_hz is not None
            and normalized.sample_rate_hz is not None
        )
        result = apply_carrier_frequency_offset(
            signal,
            offset_hz=normalized.offset_hz,
            sample_rate_hz=normalized.sample_rate_hz,
            start_sample=start_sample,
        )
    else:
        assert normalized.profile is not None
        result = apply_multipath_channel(
            signal, profile=normalized.profile, channel_seed=channel_seed
        )
    return normalize_complex_iq_window(result) if postprocess else result


def _metadata_value(metadata: Any, field: str) -> Any:
    if isinstance(metadata, Mapping):
        value = metadata.get(field)
    else:
        value = getattr(metadata, field, None)
    if value is None:
        raise IQPerturbationError(f"sample metadata must provide {field}")
    return value


class DeterministicIQPerturbationDataset(Dataset):
    """Read-only dataset view applying one perturbation per stable sample index."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        perturbation: IQPerturbationSpec | Mapping[str, Any],
        base_seed: int,
    ) -> None:
        self.dataset = dataset
        self.perturbation = (
            perturbation
            if isinstance(perturbation, IQPerturbationSpec)
            else IQPerturbationSpec.from_mapping(perturbation)
        )
        self.base_seed = _validated_seed(base_seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        signal, target, dataset_index = self.dataset[index]
        metadata = self.dataset.sample_metadata(index)
        window_id = _metadata_value(metadata, "window_id")
        seed = window_iq_perturbation_seed(self.base_seed, window_id)
        start_sample = 0
        channel_seed = None
        if self.perturbation.kind in {"carrier_frequency_offset", "multipath"}:
            if self.perturbation.kind == "carrier_frequency_offset":
                start_sample = _metadata_value(metadata, "start_sample")
            else:
                recording_id = _metadata_value(metadata, "recording_id")
                assert self.perturbation.profile is not None
                channel_seed = recording_iq_perturbation_seed(
                    self.base_seed, recording_id, self.perturbation.profile
                )
        return (
            apply_iq_perturbation(
                signal,
                self.perturbation,
                seed=seed,
                start_sample=start_sample,
                channel_seed=channel_seed,
                postprocess=True,
            ),
            target,
            dataset_index,
        )

    def sample_metadata(self, index: int) -> Any:
        return self.dataset.sample_metadata(index)

    @property
    def label_map(self) -> Any:
        return self.dataset.label_map

    @property
    def recording_ids(self) -> Any:
        return self.dataset.recording_ids


__all__ = [
    "DeterministicIQPerturbationDataset",
    "IQPerturbationError",
    "IQPerturbationSpec",
    "MULTIPATH_PROFILE_NAMES",
    "MULTIPATH_PROFILES",
    "add_complex_awgn",
    "apply_carrier_frequency_offset",
    "apply_iq_perturbation",
    "apply_multipath_channel",
    "complex_signal_power",
    "multipath_profile_taps",
    "normalize_complex_iq_window",
    "recording_iq_perturbation_seed",
    "sample_iq_perturbation_seed",
    "window_iq_perturbation_seed",
]
