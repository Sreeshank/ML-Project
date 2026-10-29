"""Conservative, sklearn-compatible engineering that retains all raw inputs.

No unit is inferred for bare numbers; unitless and explicitly unit-bearing
measurements are summarized separately. Chemical tokens are labels, not formulas.
"""
from __future__ import annotations
from collections import Counter
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .data_preprocessing import FEATURE_COLUMNS, LEAKAGE_COLUMNS, TARGET, MISSING_CATEGORY

UNKNOWN = {'', 'unknown', 'nan', 'na', 'n/a', MISSING_CATEGORY.lower()}
ABSENT = {'none', 'undoped', 'no', 'not applicable'}
TOKEN_SPLIT = re.compile(r'\s*(?:;|\||>>)\s*')
NUMBER = re.compile(r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\d]*)$')
TOKEN_FIELDS = {
    'a_site': 'Perovskite_composition_a_ions',
    'b_site': 'Perovskite_composition_b_ions',
    'c_site': 'Perovskite_composition_c_ions',
    'additive': 'Perovskite_additives_compounds',
    'solvent': 'Perovskite_deposition_solvents',
}
MEASURE_FIELDS = {
    'anneal_temperature': ('Perovskite_deposition_thermal_annealing_temperature', ('unitless','C','K')),
    'anneal_time': ('Perovskite_deposition_thermal_annealing_time', ('unitless','s','min','h')),
    'perovskite_thickness': ('Perovskite_thickness', ('unitless','nm','um','mm')),
    'backcontact_thickness': ('Backcontact_thickness_list', ('unitless','nm','um','mm')),
}
UNIT_ALIASES = {'': 'unitless', 'c':'C', '°c':'C', 'degc':'C', 'k':'K',
                's':'s', 'sec':'s', 'seconds':'s', 'min':'min', 'minutes':'min',
                'h':'h', 'hr':'h', 'hours':'h', 'nm':'nm', 'um':'um', 'µm':'um', 'μm':'um', 'mm':'mm'}

def is_missing(value):
    return bool(pd.isna(value)) or str(value).strip().lower() in UNKNOWN

def tokens(value):
    if is_missing(value):
        return [], True
    parts = TOKEN_SPLIT.split(str(value).strip())
    incomplete = any(p.strip().lower() in UNKNOWN for p in parts)
    return [p.strip() for p in parts if p.strip().lower() not in UNKNOWN | ABSENT], incomplete

def layer_count(value):
    """Count pipe-separated layers only; a known absent stack has zero layers."""
    if is_missing(value):
        return np.nan
    parts = [p.strip() for p in str(value).split('|')]
    if any(p.lower() in UNKNOWN for p in parts):
        return np.nan
    return float(sum(p.lower() not in ABSENT for p in parts))

def measurement_summary(value, units):
    bins = {u:[] for u in units}
    unknown = rejected = 0
    if pd.isna(value):
        parts = [MISSING_CATEGORY]
    else:
        parts = TOKEN_SPLIT.split(str(value).strip())
    for part in parts:
        if part.strip().lower() in UNKNOWN:
            unknown += 1
            continue
        match = NUMBER.fullmatch(part.strip())
        if match is None:
            rejected += 1
            continue
        number = float(match[1])
        unit = UNIT_ALIASES.get(match[2].strip().lower())
        # Negative durations/thickness/temperature strings are left raw for review;
        # this conservative parser excludes negative values for every field.
        if not np.isfinite(number) or number < 0 or unit not in bins:
            rejected += 1
            continue
        bins[unit].append(number)
    result = {'unknown_count':unknown, 'unparsed_count':rejected,
              'partial':int(bool(unknown or rejected))}
    for unit, values in bins.items():
        result[f'{unit}_count'] = len(values)
        for name, fn in [('min',np.min),('max',np.max),('mean',np.mean)]:
            result[f'{unit}_{name}'] = float(fn(values)) if values else np.nan
    return result

class PerovskiteFeatureEngineer(TransformerMixin, BaseEstimator):
    """Learn bounded token vocabularies on training inputs only.

    Missing counts mean unknown, not zero. Unseen token counts remain available
    at inference. No target, stoichiometry, unit conversions, or imputation is used.
    """
    def __init__(self, max_tokens=32, min_token_frequency=2):
        self.max_tokens = max_tokens
        self.min_token_frequency = min_token_frequency

    def _validate(self, X):
        if not isinstance(X,pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame')
        if not X.columns.is_unique:
            raise ValueError('Duplicate feature columns')
        missing = set(FEATURE_COLUMNS)-set(X.columns)
        extra = set(X.columns)-set(FEATURE_COLUMNS)
        if missing or extra:
            raise ValueError(f'Expected raw predictor columns only; missing={sorted(missing)}, extra={sorted(extra)}')

    def fit(self, X, y=None):
        self._validate(X)
        if not isinstance(self.max_tokens,int) or self.max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        if not isinstance(self.min_token_frequency,int) or self.min_token_frequency < 1:
            raise ValueError('min_token_frequency must be a positive integer')
        self.feature_names_in_ = np.array(FEATURE_COLUMNS,dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.vocabulary_ = {}
        for prefix, column in TOKEN_FIELDS.items():
            counts = Counter(t for value in X[column] for t in set(tokens(value)[0]))
            ranked = sorted(counts,key=lambda t:(-counts[t], t))
            self.vocabulary_[prefix] = [t for t in ranked if counts[t]>=self.min_token_frequency][:self.max_tokens]
        self.feature_names_out_ = np.array(self.transform(X.iloc[:0]).columns,dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self,'vocabulary_')
        self._validate(X)
        engineered = {}
        for column in FEATURE_COLUMNS:
            engineered[f'xai__{column}__missing'] = X[column].map(is_missing).astype('int8')
        for prefix, column in [('device','Cell_stack_sequence'),('etl','ETL_stack_sequence'),('htl','HTL_stack_sequence')]:
            engineered[f'xai__{prefix}__layer_count'] = X[column].map(layer_count)
        for prefix,column in TOKEN_FIELDS.items():
            parsed = X[column].map(tokens)
            known = parsed.map(lambda p:set(p[0]))
            incomplete = parsed.map(lambda p:p[1])
            engineered[f'xai__{prefix}__incomplete'] = incomplete.astype('int8')
            engineered[f'xai__{prefix}__count'] = pd.Series([np.nan if flag else len(ts) for ts,flag in zip(known,incomplete)],index=X.index)
            engineered[f'xai__{prefix}__known_count'] = known.map(len)
            engineered[f'xai__{prefix}__present'] = pd.Series([1. if ts else (np.nan if flag else 0.) for ts,flag in zip(known,incomplete)],index=X.index)
            vocabulary = set(self.vocabulary_[prefix])
            engineered[f'xai__{prefix}__unseen_count'] = known.map(lambda ts:len(ts-vocabulary))
            for token in self.vocabulary_[prefix]:
                engineered[f'xai__{prefix}__token={token}'] = pd.Series([1. if token in ts else (np.nan if flag else 0.) for ts,flag in zip(known,incomplete)],index=X.index)
        for prefix,(column,units) in MEASURE_FIELDS.items():
            # Explicit columns also ensure a stable schema for empty batches.
            keys = list(measurement_summary(None,units))
            rows = [measurement_summary(value,units) for value in X[column]]
            for key in keys:
                engineered[f'xai__{prefix}__{key}'] = pd.Series([row[key] for row in rows],index=X.index,dtype=float)
        return pd.concat([X.loc[:,FEATURE_COLUMNS].copy(),pd.DataFrame(engineered,index=X.index)],axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self,'feature_names_out_')
        return self.feature_names_out_.copy()


def main():
    """Audit training features using the saved Phase 1 split; do not load test features."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import joblib
    from .data_preprocessing import load_dataset, clean_dataset, make_features
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=root.parent/'Solar Cell data set2.csv')
    parser.add_argument('--splits',type=Path,default=root/'data/processed/phase1')
    parser.add_argument('--output',type=Path,default=root/'data/processed/phase2')
    args = parser.parse_args()
    audit = json.loads((args.splits/'audit.json').read_text())
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != audit['dataset_sha256']:
        raise ValueError('Dataset checksum does not match the saved split')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    clean,_,_ = clean_dataset(load_dataset(args.data))
    X,cats = make_features(clean)
    splits = np.load(args.splits/'split_indices.npz')
    train = X.iloc[splits['train']]
    validation = X.iloc[splits['validation']]
    engineer = PerovskiteFeatureEngineer().fit(train)
    transformed = engineer.transform(train)
    val = engineer.transform(validation)
    assert list(transformed.columns)==list(val.columns)
    numeric = transformed.select_dtypes(include='number')
    assert not np.isinf(numeric.to_numpy()).any()
    args.output.mkdir(parents=True)
    joblib.dump(engineer,args.output/'feature_engineer.joblib')
    report = {'dataset_sha256':digest,'training_rows':len(train),'validation_rows':len(val),
              'raw_features':len(FEATURE_COLUMNS),'total_features':transformed.shape[1],
              'engineered_features':transformed.shape[1]-len(FEATURE_COLUMNS),
              'categorical_features':cats,'vocabulary':engineer.vocabulary_,
              'feature_names':list(transformed.columns),
              'unit_policy':'Bare numbers have unspecified units; explicit units summarized separately. No conversions or stoichiometry inferred.'}
    from .data_quality import quality_report
    options = {c:train[c].dropna().unique().tolist() for c in FEATURE_COLUMNS}
    report['data_quality'] = {'status':'computed',
                              'train':quality_report(train, options, engineer.vocabulary_),
                              'validation':quality_report(validation, options, engineer.vocabulary_)}
    (args.output/'feature_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['training_rows','validation_rows','raw_features','total_features','engineered_features']},indent=2))

if __name__ == '__main__':
    main()
