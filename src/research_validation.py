"""Evaluate the frozen app model, calibrate intervals, and audit source-data quality.

Run once per frozen model; never use the resulting held-out scores to tune it.
The saved test partition is split by group into calibration and evaluation halves.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .data_preprocessing import SEED, linked_groups
from .data_quality import quality_report
from .evaluate_models import regression_metrics, save_diagnostics
from .predict import ROOT, load_bundle, make_training_presets
from .uncertainty import calibrate_intervals, interval_coverage


def split_calibration_evaluation(groups, seed=SEED + 1):
    groups = np.asarray(groups)
    if groups.ndim != 1 or pd.isna(groups).any() or len(np.unique(groups)) < 4:
        raise ValueError('At least four nonmissing held-out groups are required')
    calibration, evaluation = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(groups, groups=groups))
    assert set(groups[calibration]).isdisjoint(groups[evaluation])
    return calibration, evaluation


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def evaluate_bundle(data_path, splits, bundle, output, alpha=0.1, install=False):
    # Offline-only training imports keep the deployed app lightweight.
    from .train_models import load_saved_data
    from .data_preprocessing import clean_dataset, load_dataset

    if output.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation artifacts: {output}')
    pipeline, metadata = load_bundle(bundle)
    X, y, source, indices, digest = load_saved_data(data_path, splits)
    if digest != metadata['dataset_sha256']:
        raise ValueError('Source dataset does not match the frozen app bundle')
    audit = json.loads((splits/'audit.json').read_text(encoding='utf-8'))
    # Checking the manifest prevents evaluating a bundle against another split.
    if audit.get('split_group_column') != metadata['dataset_audit'].get('split_group_column'):
        raise ValueError('Bundle was trained with a different group policy; retrain before using these splits')
    if {k: len(v) for k, v in indices.items()} != metadata['dataset_audit']['split_rows']:
        raise ValueError('Bundle split sizes do not match saved splits')
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    if audit.get('split_group_column'):
        clean, _, _ = clean_dataset(load_dataset(data_path))
        groups = linked_groups(X, clean[audit['split_group_column']])
    train, validation, test = (indices[name] for name in ('train', 'validation', 'test'))
    for other in (train, validation):
        if not set(groups[test]).isdisjoint(groups[other]):
            raise ValueError('Held-out groups overlap model development data')
    ca, ev = split_calibration_evaluation(groups[test])
    calibration_rows, evaluation_rows = test[ca], test[ev]
    protocol = {
        'model_sha256': metadata['model_sha256'], 'dataset_sha256': digest,
        'selected_candidate': metadata['selection']['selected_candidate'],
        'seed': SEED+1, 'alpha': alpha,
        'software_versions': {'python': platform.python_version(), **{
            package: importlib.metadata.version(package) for package in ['numpy','pandas','scikit-learn','lightgbm','joblib']}},
        'group_policy': audit.get('split_group_policy', 'Identical raw configurations'),
        'calibration_source_rows': source[calibration_rows].tolist(),
        'evaluation_source_rows': source[evaluation_rows].tolist(),
        'notice': 'Frozen model; no refit or hyperparameter selection. The prior Phase 3 model already used this test partition. This is a post-selection internal evaluation, not an independent external benchmark.',
    }
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    output.mkdir(parents=True)
    _write_json(output/'protocol.json', protocol)  # Freeze assignments before prediction.
    predicted_calibration = pipeline.predict(X.iloc[calibration_rows])
    uncertainty = calibrate_intervals(y.iloc[calibration_rows], predicted_calibration, groups[calibration_rows], alpha)
    predicted_evaluation = pipeline.predict(X.iloc[evaluation_rows])
    uncertainty['evaluation'] = interval_coverage(y.iloc[evaluation_rows], predicted_evaluation, groups[evaluation_rows], uncertainty)
    metrics = regression_metrics(y.iloc[evaluation_rows], predicted_evaluation)
    save_diagnostics(y.iloc[evaluation_rows], predicted_evaluation, X.iloc[evaluation_rows], source[evaluation_rows], output)
    assignments = pd.DataFrame({'source_row': source[test], 'group': groups[test].astype(str), 'subset': 'evaluation'})
    assignments.loc[ca, 'subset'] = 'calibration'
    assignments.to_csv(output/'holdout_assignments.csv', index=False)
    engineer = pipeline.named_steps['features']
    quality = {name: quality_report(X.iloc[rows], metadata['training_options'], engineer.vocabulary_)
               for name, rows in [('train', train), ('validation', validation),
                                  ('calibration', calibration_rows), ('evaluation', evaluation_rows)]}
    quality['status'] = 'computed'
    quality['scope'] = 'Separate saved partitions, with calibration and evaluation inside the original test partition.'
    report = {'schema_version': 1, 'model_sha256': metadata['model_sha256'], 'dataset_sha256': digest,
              'software_versions': protocol['software_versions'],
              'selected_candidate': protocol['selected_candidate'],
              'evaluation': {'metrics': metrics, 'rows': len(evaluation_rows),
                             'groups': len(np.unique(groups[evaluation_rows])), 'scope': protocol['notice'],
                             'group_policy': protocol['group_policy'], 'external_benchmark': False},
              'uncertainty': uncertainty, 'data_quality': quality,
              'training_presets': make_training_presets(X.iloc[train], source[train])}
    # Validate all fields before installing; an interrupted run cannot activate intervals.
    _write_json(output/'analytics.json', report)
    if install:
        temporary = bundle/'analytics.json.tmp'
        _write_json(temporary, report)
        temporary.replace(bundle/'analytics.json')
    (output/'COMPLETE').write_text('Frozen model evaluated; separate calibration/evaluation groups; analytics verified.\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--splits', type=Path, default=ROOT/'data/processed/phase1')
    parser.add_argument('--bundle', type=Path, default=ROOT/'models/app_bundle')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/phase6_validation_001')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--install', action='store_true', help='Install matching analytics in the app bundle after successful evaluation')
    args = parser.parse_args()
    try:
        report = evaluate_bundle(args.data, args.splits, args.bundle, args.output, args.alpha, args.install)
        print(json.dumps({'evaluation': report['evaluation'], 'uncertainty': report['uncertainty']}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Research validation failed: {exc}\n')


if __name__ == '__main__':
    main()
