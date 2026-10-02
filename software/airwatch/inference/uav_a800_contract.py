"""Hash-verified FP32 runtime for three known KU Leuven RF control sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from airwatch.data.data_provenance import sha256_file
from airwatch.data.ku_leuven_preprocessing import preprocess_iq_window, PREPROCESSING_ID
from airwatch.inference.uav_open_set import energy_knownness
from airwatch.models.uav_multiscale_tcn import build_a800_model

CODE_FILES = (
    'airwatch/__init__.py', 'airwatch/models/__init__.py', 'airwatch/models/uav_baselines.py',
    'airwatch/models/uav_multiscale_tcn.py', 'airwatch/data/__init__.py',
    'airwatch/data/data_provenance.py', 'airwatch/data/ku_leuven_preprocessing.py',
    'airwatch/inference/__init__.py', 'airwatch/inference/uav_open_set.py',
    'airwatch/inference/uav_a800_contract.py',
)


def member_scores(logits, features, prototypes):
    """The same 32-window reduction used by frozen evaluation and deployment."""
    logits, features, prototypes = [np.asarray(x, dtype=np.float64) for x in (logits, features, prototypes)]
    if logits.shape != (32, 3) or features.ndim != 2 or features.shape[0] != 32 or prototypes.shape != (3, features.shape[1]):
        raise ValueError('Expected 32 complete windows, three logits and matching prototypes')
    if not all(np.isfinite(x).all() for x in (logits, features, prototypes)):
        raise ValueError('Nonfinite runtime model output')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    proto_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if (norms <= 0).any() or (proto_norms <= 0).any():
        raise ValueError('Zero-norm runtime feature/prototype')
    representation = (features / norms).mean(axis=0)
    norm = np.linalg.norm(representation)
    if norm <= 0:
        raise ValueError('Zero-norm member representation')
    representation /= norm
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities.mean(axis=0)
    return dict(prediction=int(probabilities.argmax()), probabilities=probabilities.tolist(), msp=float(probabilities.max()),
                energy=float(energy_knownness(logits, temperature=1.).mean()),
                cosine=float((representation @ (prototypes / proto_norms).T).max()))


def _local(root, relative):
    from pathlib import PureWindowsPath
    if not isinstance(relative, str) or Path(relative).is_absolute() or PureWindowsPath(relative).drive or '..' in Path(relative).parts:
        raise ValueError('Contract artifact path is not portable/local')
    value = (root / relative).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError('Contract path escapes release')
    return value


def load_contract(path, *, device='cpu'):
    path = Path(path).resolve()
    contract = json.loads(path.read_text(encoding='utf-8'))
    sidecar = path.with_name('runtime-contract.sha256')
    if not sidecar.is_file() or sidecar.read_text(encoding='ascii').strip() != sha256_file(path):
        raise ValueError('Runtime contract checksum mismatch')
    if contract.get('artifact_type') != 'airwatch_a800_runtime_v2' or contract.get('precision') != 'fp32':
        raise ValueError('Unsupported runtime contract/precision')
    expected_input = dict(sample_rate_hz=100000000, windows_per_member=32, window_samples=4096, channels=['in_phase', 'quadrature'], preprocessing_id=PREPROCESSING_ID)
    if contract.get('input') != expected_input or contract.get('primary_score') != 'cosine' or contract.get('direction') != 'higher_is_known':
        raise ValueError('Input/scoring contract changed')
    if set(contract.get('artifacts', {})) != {'model.pt', 'label-map.json', 'preprocessing.json', 'open-set.json', 'environment.json'}:
        raise ValueError('Incomplete runtime payload')
    for relative, expected in contract['artifacts'].items():
        actual = _local(path.parent, relative)
        if not actual.is_file() or actual.stat().st_size != expected['size_bytes'] or sha256_file(actual) != expected['sha256']:
            raise ValueError(f'Runtime artifact checksum mismatch: {relative}')
    source_root = Path(__file__).resolve().parents[2]
    if set(contract.get('code_hashes', {})) != set(CODE_FILES):
        raise ValueError('Incomplete runtime source identity')
    for relative, digest in contract['code_hashes'].items():
        shipped = _local(path.parent / 'source', relative)
        active = _local(source_root, relative)
        if not shipped.is_file() or sha256_file(shipped) != digest or sha256_file(active) != digest:
            raise ValueError(f'Runtime source differs from frozen implementation: {relative}')
    labels = json.loads((path.parent / 'label-map.json').read_text(encoding='utf-8'))
    if labels != {'frysky': 0, 'spektrum_dx4e': 1, 'dji_mini2_rc': 2}:
        raise ValueError('Runtime label mapping changed')
    preprocessing = json.loads((path.parent / 'preprocessing.json').read_text(encoding='utf-8'))
    if preprocessing != expected_input:
        raise ValueError('Runtime preprocessing metadata changed')
    opened = json.loads((path.parent / 'open-set.json').read_text(encoding='utf-8'))
    if opened.get('primary_score') != 'cosine' or opened.get('comparison') != 'score>=threshold' or opened.get('direction') != 'higher_is_known' or set(opened.get('scores', {})) != {'cosine', 'msp', 'energy'}:
        raise ValueError('Runtime open-set protocol changed')
    prototypes = np.asarray(opened['prototypes'], dtype=np.float64)
    if prototypes.shape != (3, 256) or not np.isfinite(prototypes).all() or not np.allclose(np.linalg.norm(prototypes, axis=1), 1., atol=1e-6):
        raise ValueError('Runtime prototypes must be finite normalized class vectors')
    for specification in opened['scores'].values():
        if specification.get('comparison') != 'score>=threshold' or specification.get('direction') != 'higher_is_known' or not np.isfinite(specification.get('threshold', np.nan)):
            raise ValueError('Runtime threshold is invalid')
    model = build_a800_model(contract['architecture'], num_classes=3)
    model.load_state_dict(torch.load(path.parent / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    if sum(p.numel() for p in model.parameters()) != contract.get('parameters'):
        raise ValueError('Runtime architecture parameter count changed')
    model.eval().float().to(torch.device(device))
    contract.update(_model=model, _device=torch.device(device), _prototypes=prototypes, _scores=opened['scores'], _labels=labels, _path=str(path))
    return contract


def predict_member(contract, iq_windows, sample_rate_hz):
    values = np.asarray(iq_windows)
    if values.shape != (32, 2, 4096) or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
        raise ValueError('Runtime member requires 32 finite IQ windows [32,2,4096]')
    if sample_rate_hz != 100000000:
        raise ValueError('Runtime requires 100 MS/s complex IQ; automatic conversion is not registered')
    device = contract['_device']
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    prepared = np.stack([preprocess_iq_window(window) for window in values])
    with torch.inference_mode():
        tensor = torch.from_numpy(prepared).to(device=device, dtype=torch.float32)
        feature = contract['_model'].forward_features(tensor)
        logits = contract['_model'].classifier(feature)
        feature, logits = feature.float().cpu().numpy(), logits.float().cpu().numpy()
    scores = member_scores(logits, feature, contract['_prototypes'])
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    prediction = scores['prediction']
    label = next(name for name, value in contract['_labels'].items() if value == prediction)
    accepted = scores['cosine'] >= contract['_scores']['cosine']['threshold']
    return dict(known_class_candidate=label, label_index=prediction, confidence=scores['msp'],
                decision='known' if accepted else 'unknown', scores={name: scores[name] for name in ('cosine', 'msp', 'energy')},
                probabilities=scores['probabilities'], logits=logits.tolist(), model_version=contract['model_version'],
                input_sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
                window_count=32, elapsed_seconds=elapsed, precision='fp32',
                limitation='Unknown means rejected by known-source model; it is not proof of UAV presence.')
