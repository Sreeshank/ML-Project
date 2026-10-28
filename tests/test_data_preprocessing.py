import numpy as np
import pandas as pd
import pytest
from src.data_preprocessing import (FEATURE_COLUMNS, EXPECTED_COLUMNS, LEAKAGE_COLUMNS,
    TARGET, clean_dataset, make_features, split_groups, build_preprocessor, load_dataset)

def sample():
    raw = pd.DataFrame({c:['Unknown']*40 for c in EXPECTED_COLUMNS})
    raw['Cell_stack_sequence'] = [f'layer_{i//2}' for i in range(40)]
    raw['Perovskite_deposition_number_of_deposition_steps'] = 1.0
    raw[TARGET] = np.arange(40,dtype=float)
    return raw

def test_cleaning_and_provenance():
    raw = sample()
    raw.loc[1,TARGET] = np.nan
    raw = pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
    clean, rows, audit = clean_dataset(raw)
    assert len(clean)==39 and len(rows)==39
    assert 1 not in rows and 40 not in rows
    assert audit['exact_duplicates_removed']==1
    assert clean[TARGET].iloc[0]==0  # Do not discard low efficiencies.

def test_schema(tmp_path):
    path=tmp_path/'bad.csv'
    pd.DataFrame({'wrong':[1]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Missing expected'): load_dataset(path)
    with pytest.raises(FileNotFoundError): load_dataset(tmp_path/'absent.csv')

def test_invalid_target():
    raw=sample(); raw.loc[0,TARGET]=np.inf
    with pytest.raises(ValueError,match='infinite'): clean_dataset(raw)

def test_leakage_and_preservation():
    raw=sample(); raw.loc[0,'Perovskite_thickness']='100; 200 nm'
    raw.loc[1,'Perovskite_thickness']=None
    X,cats=make_features(raw)
    assert list(X.columns)==FEATURE_COLUMNS
    assert set(LEAKAGE_COLUMNS+[TARGET]).isdisjoint(X.columns)
    assert X.loc[0,'Perovskite_thickness']=='100; 200 nm'
    assert X.loc[1,'Perovskite_thickness']=='__MISSING__'
    assert X.loc[2,'Perovskite_thickness']=='Unknown'

def test_group_split_repeatable_and_complete():
    X,_=make_features(sample())
    a,g=split_groups(X); b,_=split_groups(X)
    assert len(np.unique(np.concatenate(list(a.values()))))==len(X)
    for k in a: np.testing.assert_array_equal(a[k],b[k])
    for k,l in [('train','validation'),('train','test'),('validation','test')]:
        assert set(g[a[k]]).isdisjoint(g[a[l]])
    assert g[0]==g[1]

def test_training_only_imputation_and_unseen_category():
    train=pd.DataFrame({'material':['A','B'],'numeric':[1.0,3.0]})
    unseen=pd.DataFrame({'material':['new'],'numeric':[np.nan]})
    pre=build_preprocessor(train,['material'])
    pre.fit(train)
    assert pre.named_transformers_['numeric'].named_steps['impute'].statistics_[0]==2
    assert pre.transform(unseen).shape[0]==1
    assert 'new' not in pre.named_transformers_['categorical'].named_steps['encode'].categories_[0]
