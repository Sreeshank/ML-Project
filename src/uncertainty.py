"""Finite-sample split conformal intervals, using a score per configuration group."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

METHOD = 'configuration_group_max_absolute_residual'


def calibrate_intervals(actual, predicted, groups, alpha=0.1):
    actual, predicted, groups = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float), np.asarray(groups)
    if actual.ndim != 1 or not len(actual) or actual.shape != predicted.shape or actual.shape != groups.shape:
        raise ValueError('Calibration arrays must be nonempty, aligned vectors')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or pd.isna(groups).any():
        raise ValueError('Calibration values and groups must be finite/nonmissing')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    scores = pd.DataFrame({'group': groups, 'score': np.abs(actual - predicted)}).groupby('group').score.max().to_numpy()
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError('Too few calibration groups for a finite interval at this coverage level')
    return {'method': METHOD, 'alpha': alpha, 'nominal_coverage': 1 - alpha,
            'radius_PCE_points': float(np.sort(scores)[rank - 1]),
            'calibration_rows': len(actual), 'calibration_groups': len(scores), 'quantile_rank': rank,
            'notice': 'Nominal coverage assumes exchangeable configuration groups. Similar devices and shared publications can violate this assumption. Intervals are not clipped to physical bounds.'}


def interval_coverage(actual, predicted, groups, calibration):
    actual, predicted, groups = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float), np.asarray(groups)
    if actual.ndim != 1 or len(actual) < 2 or actual.shape != predicted.shape or actual.shape != groups.shape:
        raise ValueError('Evaluation requires at least two aligned observations')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or pd.isna(groups).any():
        raise ValueError('Evaluation values and groups must be finite/nonmissing')
    covered = np.abs(actual - predicted) <= calibration['radius_PCE_points']
    group_covered = pd.DataFrame({'group': groups, 'covered': covered}).groupby('group').covered.all()
    return {'evaluation_rows': len(actual), 'evaluation_groups': len(group_covered),
            'empirical_row_coverage': float(covered.mean()),
            'empirical_group_coverage': float(group_covered.mean()),
            'interval_width_PCE_points': 2 * calibration['radius_PCE_points']}


def prediction_intervals(predictions, analytics, model_sha256):
    """Only expose calibrated intervals paired with this exact evaluated model."""
    if not analytics:
        return None
    if analytics.get('model_sha256') != model_sha256:
        raise ValueError('Uncertainty artifact belongs to a different model')
    calibration = analytics.get('uncertainty')
    if not calibration:
        return None
    radius = calibration.get('radius_PCE_points')
    if calibration.get('method') != METHOD or not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius < 0:
        raise ValueError('Invalid uncertainty calibration artifact')
    evaluation = calibration.get('evaluation', {})
    if evaluation.get('evaluation_groups', 0) < 1:
        raise ValueError('Uncertainty coverage has not been evaluated on a separate subset')
    values = np.asarray(predictions, dtype=float)
    return {'lower_PCE_percent': (values-radius).tolist(), 'upper_PCE_percent': (values+radius).tolist(),
            'nominal_coverage': calibration['nominal_coverage'],
            'empirical_group_coverage': evaluation['empirical_group_coverage'],
            'empirical_row_coverage': evaluation['empirical_row_coverage'],
            'notice': calibration['notice']}
