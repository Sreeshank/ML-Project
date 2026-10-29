"""Phase 1: schema validation, leakage exclusion, and reproducible splits.

Run from the project root with ``python -m src.data_preprocessing``.
Source row IDs are zero-based CSV data-row positions (excluding the header).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = 'JV_default_PCE'
LEAKAGE_COLUMNS = ['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_measured']
FEATURE_COLUMNS = [
    'Cell_stack_sequence', 'Cell_architecture', 'Substrate_stack_sequence',
    'ETL_stack_sequence', 'ETL_deposition_procedure',
    'Perovskite_composition_a_ions', 'Perovskite_composition_b_ions',
    'Perovskite_composition_c_ions', 'Perovskite_composition_short_form',
    'Perovskite_additives_compounds', 'Perovskite_additives_concentrations',
    'Perovskite_thickness', 'Perovskite_deposition_number_of_deposition_steps',
    'Perovskite_deposition_procedure', 'Perovskite_deposition_solvents',
    'Perovskite_deposition_quenching_induced_crystallisation',
    'Perovskite_deposition_thermal_annealing_temperature',
    'Perovskite_deposition_thermal_annealing_time', 'HTL_stack_sequence',
    'HTL_deposition_procedure', 'Backcontact_stack_sequence',
    'Backcontact_thickness_list', 'Backcontact_deposition_procedure',
    'Encapsulation', 'Encapsulation_stack_sequence',
]
EXPECTED_COLUMNS = FEATURE_COLUMNS + LEAKAGE_COLUMNS + [TARGET]
MISSING_CATEGORY = '__MISSING__'

def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Dataset not found: {path.resolve()}')
    # Check the header before pandas can silently rename duplicate columns.
    import csv
    with path.open(newline='', encoding='utf-8-sig') as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError('CSV contains duplicate column names')
    missing = sorted(set(EXPECTED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f'Missing expected columns: {missing}')
    return pd.read_csv(path, low_memory=False)

def clean_dataset(raw: pd.DataFrame):
    missing = sorted(set(EXPECTED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f'Missing expected columns: {missing}')
    # Deduplicate original records before adding row provenance.
    clean = raw.loc[~raw.duplicated()].copy()
    source_rows = np.flatnonzero(~raw.duplicated().to_numpy())
    target = pd.to_numeric(clean[TARGET], errors='coerce')
    invalid = clean[TARGET].notna() & (~np.isfinite(target))
    if invalid.any():
        raise ValueError(f'{int(invalid.sum())} nonmissing PCE values are nonnumeric or infinite')
    keep = target.notna().to_numpy()
    clean[TARGET] = target
    clean = clean.loc[keep].reset_index(drop=True)
    source_rows = source_rows[keep]
    audit = {'raw_rows': len(raw), 'raw_columns':len(raw.columns),
             'exact_duplicates_removed':int(raw.duplicated().sum()),
             'missing_target_removed_after_deduplication':int((~keep).sum()),
             'cleaned_rows': len(clean), 'input_features':len(FEATURE_COLUMNS),
             'excluded_columns': LEAKAGE_COLUMNS,
             'extra_columns_excluded':sorted(set(raw.columns)-set(EXPECTED_COLUMNS)),
             'zero_or_negative_targets_retained':int((clean[TARGET]<=0).sum()),
             'target_min':float(clean[TARGET].min()), 'target_max':float(clean[TARGET].max())}
    if len(clean) == 0:
        raise ValueError('No labeled records remain')
    return clean, source_rows, audit

def make_features(clean: pd.DataFrame):
    """Stateless normalization only; preserve units, delimiters, and Unknown text."""
    X = clean.loc[:, FEATURE_COLUMNS].copy()
    cats = X.select_dtypes(include=['object','string','category']).columns.tolist()
    for c in cats:
        if X[c].eq(MISSING_CATEGORY).any():
            raise ValueError(f'Reserved missing token occurs in {c}')
        X[c] = X[c].fillna(MISSING_CATEGORY).astype(str)
    for c in X.columns.difference(cats):
        X[c] = X[c].astype(float)
        if np.isinf(X[c]).any():
            raise ValueError(f'Infinite numeric input in {c}')
    return X, cats

def split_groups(X: pd.DataFrame, groups=None):
    # Do not use target values, row IDs, or engineered features to define groups.
    if groups is None:
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    else:
        groups = np.asarray(groups)
        if groups.shape != (len(X),) or pd.isna(groups).any():
            raise ValueError('Group IDs must be aligned, nonmissing values')
    if len(np.unique(groups)) < 7:
        raise ValueError('At least seven distinct configurations are required')
    trainval, test = next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SEED).split(X,groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1,test_size=.1765,random_state=SEED).split(X.iloc[trainval],groups=groups[trainval]))
    splits = {'train':trainval[tr], 'validation':trainval[va], 'test':test}
    combined = np.concatenate(list(splits.values()))
    assert len(combined) == len(X) and len(np.unique(combined)) == len(X)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert set(groups[splits[a]]).isdisjoint(groups[splits[b]])
    return splits, groups

def build_preprocessor(X_train: pd.DataFrame, categorical_columns: list[str]):
    """Return an UNFITTED sparse OHE pipeline; fit exclusively on training rows.

    Raw CatBoost inputs instead use make_features; numeric missing values are
    natively supported by CatBoost. No chemistry strings are coerced to numbers.
    """
    numeric = [c for c in X_train.columns if c not in categorical_columns]
    return ColumnTransformer([
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant',fill_value=MISSING_CATEGORY,keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore',sparse_output=True)),
        ]),categorical_columns),
        ('numeric', Pipeline([
            ('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
            ('scale',StandardScaler(with_mean=False)),
        ]),numeric),
    ],sparse_threshold=1.0)

def prepare(data_path: str | Path, output: str | Path, group_column=None):
    raw = load_dataset(data_path)
    clean, source_rows, audit = clean_dataset(raw)
    X, cats = make_features(clean)
    if group_column:
        if group_column in EXPECTED_COLUMNS or group_column not in clean:
            raise ValueError('Group column must be a separate publication/laboratory ID column')
        ids = clean[group_column]
        if ids.isna().any() or ids.astype(str).str.strip().eq('').any():
            raise ValueError('Publication/laboratory group IDs cannot be missing')
        # Connected components keep both repeated configurations and shared
        # publications together, including configurations spanning publications.
        group_ids = linked_groups(X, ids)
    else:
        group_ids = None
    splits, groups = split_groups(X, group_ids)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite artifacts: {output}')
    output.mkdir(parents=True)
    np.savez_compressed(output/'split_indices.npz',**splits)
    manifest = pd.DataFrame({'clean_row':np.arange(len(X)), 'source_row':source_rows,
                             'configuration_group':groups.astype(str)})
    for label, idx in splits.items():
        manifest.loc[idx,'split'] = label
    manifest.to_csv(output/'split_manifest.csv',index=False)
    audit.update({'seed':SEED,'split_group_column':group_column,
                  'split_group_policy':'Linked publication/laboratory and identical configuration groups' if group_column else 'Identical raw configurations',
                  'categorical_features':len(cats),'categorical_columns':cats,
                  'feature_columns':FEATURE_COLUMNS,
                  'split_rows':{k:len(v) for k,v in splits.items()},
                  'split_groups':{k:len(np.unique(groups[v])) for k,v in splits.items()},
                  'dataset_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                  'missing_values_before_normalization':clean[FEATURE_COLUMNS].isna().sum().to_dict(),
                  'versions':{'python':platform.python_version(),'pandas':pd.__version__,
                              'numpy':np.__version__,'scikit-learn':sklearn.__version__}})
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


def linked_groups(X, publication_ids):
    """Union publication IDs connected by an identical normalized device."""
    ids = np.asarray(publication_ids).astype(str)
    if ids.shape != (len(X),) or pd.isna(np.asarray(publication_ids)).any() or any(not v.strip() for v in ids):
        raise ValueError('Publication IDs must be aligned and nonmissing')
    parent = {value: value for value in ids}
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    first = {}
    hashes = pd.util.hash_pandas_object(X, index=False).to_numpy()
    for config, publication in zip(hashes, ids):
        if config in first:
            a, b = find(publication), find(first[config])
            parent[max(a, b)] = min(a, b)
        else:
            first[config] = publication
    return np.array([find(value) for value in ids])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[2]/'Solar Cell data set2.csv')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'data'/'processed'/'phase1')
    parser.add_argument('--group-column', help='Optional publication/laboratory ID column; creates new group-safe splits for a new training run')
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.data,args.output,args.group_column),indent=2))
    except (ValueError,FileNotFoundError,FileExistsError) as exc:
        parser.exit(1,f'Preprocessing failed: {exc}\n')

if __name__ == '__main__':
    main()
