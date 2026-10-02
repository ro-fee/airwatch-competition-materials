"""Reproducible synthetic checks; no dataset/model accuracy claims.

Run with the dedicated Conda Python. --package-dir permits isolated wheel tests.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import hilbert, chirp


def run(output, package_dir=None):
    if package_dir:
        sys.path.insert(0, str(Path(package_dir).resolve()))
    from PyEMD import EMD
    from importlib.metadata import version
    fs = 1024
    n = 2048
    t = np.arange(n) / fs
    rng = np.random.default_rng(42)
    cases = {
        'tone_64Hz': np.cos(2*np.pi*64*t),
        'two_tones_32_160Hz': np.cos(2*np.pi*160*t)+.5*np.cos(2*np.pi*32*t),
        'chirp_40_to_160Hz': chirp(t, f0=40, t1=t[-1], f1=160),
        'am_96Hz': (1+.4*np.cos(2*np.pi*4*t))*np.cos(2*np.pi*96*t),
        'noisy_tone': np.cos(2*np.pi*64*t)+.15*rng.standard_normal(n),
        'tone_plus_trend': np.cos(2*np.pi*64*t)+.2*t,
        'max_length_8192': np.cos(2*np.pi*64*np.arange(8192)/fs),
    }
    records = []
    for name, x in cases.items():
        started = time.perf_counter()
        emd = EMD(spline_kind='cubic', nbsym=2, MAX_ITERATION=1000)
        parts = emd.emd(x.copy(), max_imf=8)
        imfs, residue = emd.get_imfs_and_residue()
        elapsed = time.perf_counter()-started
        analytic = hilbert(imfs[0])
        frequency = np.gradient(np.unwrap(np.angle(analytic))) * fs/(2*np.pi)
        crop = slice(128,-128)
        error = float(np.max(np.abs(parts.sum(axis=0)-x)))
        record = dict(case=name, samples=len(x), components=len(parts), imfs=len(imfs),
            elapsed_seconds=elapsed, reconstruction_max_abs_error=error,
            first_imf_median_hz=float(np.median(frequency[crop])),
            residue_peak=float(np.max(np.abs(residue))), finite=bool(np.isfinite(parts).all()))
        if name.startswith('chirp'):
            target = 40+120*t/t[-1]
            record['frequency_median_abs_error_hz'] = float(np.median(np.abs(frequency[crop]-target[crop])))
        if name.startswith('two_tones'):
            record['second_imf_median_hz'] = float(np.median(np.gradient(np.unwrap(np.angle(hilbert(imfs[1]))))[crop]*fs/(2*np.pi)))
        assert record['finite'] and error < 1e-8, record
        records.append(record)
    by_name = {r['case']:r for r in records}
    checks = {
        'tone_frequency_within_1Hz': abs(by_name['tone_64Hz']['first_imf_median_hz']-64)<1,
        'two_tones_high_within_2Hz': abs(by_name['two_tones_32_160Hz']['first_imf_median_hz']-160)<2,
        'two_tones_low_within_2Hz': abs(by_name['two_tones_32_160Hz']['second_imf_median_hz']-32)<2,
        'am_frequency_within_2Hz': abs(by_name['am_96Hz']['first_imf_median_hz']-96)<2,
        'chirp_median_error_below_3Hz': by_name['chirp_40_to_160Hz']['frequency_median_abs_error_hz']<3,
    }
    result = dict(python=sys.version, platform=platform.platform(),
        versions={key:version(key) for key in ('EMD-signal','numpy','scipy')},
        protocol=dict(seed=42, fs=fs, max_imf=8, max_iteration=1000, boundary_crop_samples=128,
            note='Single-run synthetic functional checks; timing includes decomposition only, not UI or production benchmark.'),
        cases=records, checks=checks)
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    assert all(checks.values()), checks


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-dir')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.output,args.package_dir)
