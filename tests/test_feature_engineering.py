import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from src.data_preprocessing import FEATURE_COLUMNS, build_preprocessor
from src.feature_engineering import PerovskiteFeatureEngineer, measurement_summary, layer_count

def frame(n=3):
    return pd.DataFrame({c:['Unknown']*n for c in FEATURE_COLUMNS})

def test_layers():
    assert layer_count('SLG | FTO | Perovskite | Au')==4
    assert layer_count('none')==0
    assert np.isnan(layer_count('TiO2 | Unknown'))
    assert np.isnan(layer_count(None))

def test_numeric_units_and_no_guesses():
    x=measurement_summary('100; 200 >> Unknown | 300 nm | 0.5 um',('unitless','nm','um'))
    assert x['unitless_mean']==150 and x['nm_mean']==300 and x['um_mean']==.5
    assert x['unknown_count']==1 and x['partial']==1
    assert measurement_summary('10 nm; 20',('unitless','nm'))['unitless_mean']==20
    assert measurement_summary('30 s; 2 min',('unitless','s','min'))['s_mean']==30
    assert measurement_summary('100 °C; 373 K',('unitless','C','K'))['C_mean']==100
    for bad in ['100-200','~100','100 furlong','-3','1e999']:
        z=measurement_summary(bad,('unitless','nm'))
        assert z['unparsed_count']==1 and np.isnan(z['unitless_mean'])

def test_tokens_missing_and_raw_preservation():
    X=frame(); X['Perovskite_composition_a_ions']=['Cs; FA | FA','Unknown','none']
    X['Perovskite_additives_compounds']=['Undoped',None,'Cl; Unknown']
    X['Perovskite_composition_short_form']=['Cs0.2FA0.8PbI3']*3
    before=X.copy(deep=True)
    fe=PerovskiteFeatureEngineer(min_token_frequency=1).fit(X)
    out=fe.transform(X)
    pd.testing.assert_frame_equal(X,before)
    pd.testing.assert_frame_equal(out[FEATURE_COLUMNS],X)
    assert out.loc[0,'xai__a_site__count']==2
    assert out.loc[0,'xai__a_site__token=Cs']==1
    assert np.isnan(out.loc[1,'xai__a_site__token=Cs'])
    assert out.loc[2,'xai__a_site__count']==0
    assert out.loc[0,'xai__additive__present']==0
    assert np.isnan(out.loc[1,'xai__additive__present'])
    assert out.loc[2,'xai__additive__known_count']==1
    assert np.isnan(out.loc[2,'xai__additive__count'])
    assert out.loc[1,'xai__Perovskite_additives_compounds__missing']==1

def test_training_only_vocab_and_stable_inference(tmp_path):
    X=frame(2); X['Perovskite_deposition_solvents']=['DMF','DMF']
    fe=PerovskiteFeatureEngineer().fit(X)
    unseen=frame(1); unseen.index=[42]; unseen['Perovskite_deposition_solvents']='NewSolvent'
    out=fe.transform(unseen)
    assert 'NewSolvent' not in fe.vocabulary_['solvent']
    assert out.loc[42,'xai__solvent__unseen_count']==1
    assert out.columns.tolist()==fe.get_feature_names_out().tolist()
    assert fe.transform(X.iloc[:0]).columns.tolist()==out.columns.tolist()
    path=tmp_path/'fe.joblib'; joblib.dump(fe,path)
    pd.testing.assert_frame_equal(joblib.load(path).transform(unseen),out)
    assert clone(fe).get_params()==fe.get_params()

def test_leakage_rejected_and_schema_errors():
    X=frame()
    for col in ['JV_default_PCE','JV_default_Voc','source_row']:
        with pytest.raises(ValueError,match='raw predictor'): PerovskiteFeatureEngineer().fit(X.assign(**{col:1}))
    with pytest.raises(ValueError): PerovskiteFeatureEngineer().fit(X.drop(columns=FEATURE_COLUMNS[0]))

def test_engineering_preprocessing_integration():
    X=frame(); X['Perovskite_thickness']=['100 nm','200 nm','Unknown']
    fe=PerovskiteFeatureEngineer().fit(X.iloc[:2])
    tr=fe.transform(X.iloc[:2]); va=fe.transform(X.iloc[2:])
    cats=tr.select_dtypes(include=['object']).columns.tolist()
    pre=build_preprocessor(tr,cats)
    a=pre.fit_transform(tr); b=pre.transform(va)
    assert a.shape[1]==b.shape[1] and np.isfinite(b.data).all()
