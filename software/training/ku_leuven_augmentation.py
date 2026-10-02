"""Frozen, training-only mixture of clean IQ, noise, and multipath."""
from airwatch.analysis.iq_perturbations import (
    DeterministicIQPerturbationDataset, window_iq_perturbation_seed,
)

PROTOCOL_ID = "ku-leuven-clean-awgn-multipath-training-v1"
STRONG_NOISE_PROTOCOL_ID = "ku-leuven-clean-strong-awgn-multipath-training-v2"
WEIGHTED_WORST_NOISE_PROTOCOL_ID = (
    "ku-leuven-clean-weighted-minus5-awgn-multipath-training-v3"
)

PROTOCOLS = {
    PROTOCOL_ID: {
        "awgn_snr_db": (0, 5, 10, 15, 20),
        "multipath_profiles": ("mild", "severe"),
    },
    STRONG_NOISE_PROTOCOL_ID: {
        "awgn_snr_db": (-5, 0, 5, 10, 15, 20),
        "awgn_weights": (1, 1, 1, 1, 1, 1),
        "multipath_profiles": ("mild", "severe"),
    },
    WEIGHTED_WORST_NOISE_PROTOCOL_ID: {
        "awgn_snr_db": (-5, 0, 5, 10, 15, 20),
        "awgn_weights": (5, 1, 1, 1, 1, 1),
        "multipath_profiles": ("mild", "severe"),
    },
}


class AugmentedIQTrainingDataset:
    """Deterministic 50% clean, 25% AWGN, 25% multipath mixture per epoch."""

    def __init__(self, dataset, *, seed, epoch, protocol=PROTOCOL_ID):
        if dataset.split != "train" or epoch < 1:
            raise ValueError("augmentation requires train split and a positive epoch")
        if protocol not in PROTOCOLS:
            raise ValueError(f"unsupported IQ augmentation protocol: {protocol!r}")
        self.dataset = dataset
        self.protocol = protocol
        self.seed = int(seed) + int(epoch) * 1000033
        definition = PROTOCOLS[protocol]
        self.awgn_snr_db = definition["awgn_snr_db"]
        weights = definition.get("awgn_weights", (1,) * len(self.awgn_snr_db))
        if len(weights) != len(self.awgn_snr_db) or any(
            isinstance(weight, bool) or not isinstance(weight, int) or weight < 1
            for weight in weights
        ):
            raise ValueError("invalid frozen AWGN sampling weights")
        self.awgn_weights = tuple(weights)
        self.awgn_sampling_levels = tuple(
            level
            for level, weight in zip(self.awgn_snr_db, self.awgn_weights)
            for _ in range(weight)
        )
        self.multipath_profiles = definition["multipath_profiles"]
        specs = [
            {"kind": "awgn", "snr_db": level}
            for level in self.awgn_sampling_levels
        ] + [
            {"kind": "multipath", "profile": profile}
            for profile in self.multipath_profiles
        ]
        self.views = [DeterministicIQPerturbationDataset(
            dataset, perturbation=spec, base_seed=self.seed
        ) for spec in specs]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key = window_iq_perturbation_seed(
            self.seed, self.dataset.sample_metadata(index).window_id
        )
        branch = key % 4
        if branch < 2:
            return self.dataset[index]
        awgn_count = len(self.awgn_sampling_levels)
        if branch == 2:
            view = (key // 4) % awgn_count
        else:
            view = awgn_count + (key // 4) % len(self.multipath_profiles)
        return self.views[view][index]


__all__ = [
    "AugmentedIQTrainingDataset",
    "PROTOCOL_ID",
    "PROTOCOLS",
    "STRONG_NOISE_PROTOCOL_ID",
    "WEIGHTED_WORST_NOISE_PROTOCOL_ID",
]
