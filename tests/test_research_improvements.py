import json
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import FEATURE_COLUMNS, linked_groups, split_groups, prepare
from src.data_quality import quality_report, availability_snapshot
from src.feature_engineering import PerovskiteFeatureEngineer
from src.input_guidance import check_device, explanation_summary
from src.predict import ROOT, load_bundle, predict_records
from src.research_validation import split_calibration_evaluation, evaluate_bundle
from src.uncertainty import calibrate_intervals, interval_coverage, prediction_intervals


def device():
    return json.loads((ROOT/'models/app_bundle/example_input.json').read_text())


def test_consistency_checks_are_conservative_and_specific():
    row=device()
    assert not any(i['code'] in ('architecture_order','composition_mismatch') for i in check_device(row))
    row['Cell_architecture']='pin'
    assert 'architecture_order' in {i['code'] for i in check_device(row)}
    row['Perovskite_composition_b_ions']='Pb'
    assert 'composition_mismatch' in {i['code'] for i in check_device(row)}
    row['Perovskite_composition_short_form']='complex / unknown formula'
    assert 'composition_mismatch' not in {i['code'] for i in check_device(row)}
    row['Encapsulation_stack_sequence']='Epoxy | SLG'
    assert 'encapsulation_conflict' in {i['code'] for i in check_device(row)}
    row['Perovskite_thickness']='~500 nm'
    assert 'unparsed_measurement' in {i['code'] for i in check_device(row)}


def test_summary_uses_signs_values_and_no_causal_claim():
    row=device()
    text=explanation_summary({'Cell_architecture':1.2,'ETL_stack_sequence':-0.5},row)
    assert 'raised' in text and 'lowered' in text and '+1.20' in text and '-0.50' in text
    assert 'Device architecture = nip' in text and 'not evidence' in text


def test_quality_keeps_unknown_categories_distinct_from_missing_and_vocab_limits():
    train=pd.DataFrame([device(),device()])
    train['Perovskite_deposition_solvents']=['DMF','DMSO']
    engineer=PerovskiteFeatureEngineer(max_tokens=1,min_token_frequency=1).fit(train)
    options={c:train[c].unique().tolist() for c in FEATURE_COLUMNS}
    val=train.copy()
    val['Perovskite_thickness']=['100; 200 nm','~500 nm']
    val['Perovskite_deposition_solvents']=['DMSO','Unknown']
    report=quality_report(val,options,engineer.vocabulary_)
    solvents=report['fields']['Perovskite_deposition_solvents']
    assert solvents['missing_rate']==0.5 and solvents['unknown_category_count']==0
    assert solvents['tokens']['rows_outside_bounded_vocabulary']==1
    measurement=report['fields']['Perovskite_thickness']['measurement']
    assert measurement['rows_with_parsed_values']==1
    assert measurement['rows_with_unparsed_values']==1
    assert measurement['rows_with_unspecified_units']==1
    snapshot=availability_snapshot(json.loads((ROOT/'reports/phase1_audit.json').read_text()))
    assert snapshot['fields']['Perovskite_thickness']['parsing_success_rate'] is None


def test_conformal_finite_sample_quantile_and_group_max_scores():
    groups=np.repeat(np.arange(19),2)
    actual=np.repeat(np.arange(1,20),2).astype(float)
    actual[::2]=0
    calibration=calibrate_intervals(actual,np.zeros(38),groups,alpha=0.1)
    assert calibration['quantile_rank']==18
    assert calibration['radius_PCE_points']==18
    coverage=interval_coverage([17,19],[0,0],['a','b'],calibration)
    assert coverage['empirical_row_coverage']==0.5
    assert coverage['empirical_group_coverage']==0.5
    calibration['evaluation']=coverage
    report={'model_sha256':'expected','uncertainty':calibration}
    interval=prediction_intervals([2],report,'expected')
    assert interval['lower_PCE_percent']==[-16]  # No arbitrary clipping.
    with pytest.raises(ValueError,match='different model'):
        prediction_intervals([2],report,'other')
    with pytest.raises(ValueError,match='Too few'):
        calibrate_intervals([0,1],[0,0],[0,1],0.1)


def test_calibration_and_evaluation_never_split_a_configuration():
    groups=np.repeat(np.arange(40),3)
    ca,ev=split_calibration_evaluation(groups)
    assert set(groups[ca]).isdisjoint(groups[ev])
    assert set(ca)|set(ev)==set(range(len(groups)))
    ca2,ev2=split_calibration_evaluation(groups)
    np.testing.assert_array_equal(ca,ca2)
    np.testing.assert_array_equal(ev,ev2)


def test_publication_linked_groups_preserve_both_types_of_separation():
    X=pd.DataFrame({'input':['shared','shared']+[f'config_{i}' for i in range(30)]})
    publications=['paper_a','paper_b']+[f'paper_{i//2}' for i in range(30)]
    groups=linked_groups(X,publications)
    assert groups[0]==groups[1]
    assert groups[2]==groups[3]
    splits,_=split_groups(X,groups)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert set(groups[splits[a]]).isdisjoint(groups[splits[b]])
        assert set(X.iloc[splits[a]]['input']).isdisjoint(X.iloc[splits[b]]['input'])


def test_publication_split_roundtrip(tmp_path):
    from src.data_preprocessing import EXPECTED_COLUMNS
    from src.train_models import load_saved_data
    raw=pd.DataFrame({c:['known']*80 for c in EXPECTED_COLUMNS})
    raw['Cell_stack_sequence']=[f'config_{i}' for i in range(80)]
    raw['paper_id']=[f'paper_{i//2}' for i in range(80)]
    raw['JV_default_PCE']=np.arange(80,dtype=float)
    path=tmp_path/'source.csv'; raw.to_csv(path,index=False)
    report=prepare(path,tmp_path/'splits',group_column='paper_id')
    assert report['split_group_column']=='paper_id'
    load_saved_data(path,tmp_path/'splits')


def test_frozen_validation_end_to_end_with_synthetic_source(tmp_path):
    """Synthetic data tests the protocol only; never published as research results."""
    import hashlib
    import joblib
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline
    from src.data_preprocessing import EXPECTED_COLUMNS
    raw=pd.DataFrame([device() for _ in range(300)])
    raw=raw.replace('__MISSING__',None)
    raw['Cell_stack_sequence']=[f'configuration_{i}' for i in range(300)]
    for c in set(EXPECTED_COLUMNS)-set(FEATURE_COLUMNS):
        raw[c]=np.arange(300,dtype=float)/10
    data=tmp_path/'source.csv'; raw.to_csv(data,index=False)
    splits=tmp_path/'splits'; audit=prepare(data,splits)
    from src.train_models import load_saved_data
    X,y,_,idx,digest=load_saved_data(data,splits)
    train=X.iloc[idx['train']]
    engineer=PerovskiteFeatureEngineer().fit(train)
    pipeline=Pipeline([('features',engineer),('model',DummyRegressor())]).fit(train,y.iloc[idx['train']])
    bundle=tmp_path/'bundle'; bundle.mkdir()
    joblib.dump(pipeline,bundle/'pipeline.joblib')
    metadata={'feature_columns':FEATURE_COLUMNS,'dataset_sha256':digest,
              'model_sha256':hashlib.sha256((bundle/'pipeline.joblib').read_bytes()).hexdigest(),
              'training_options':{c:train[c].dropna().unique().tolist() for c in FEATURE_COLUMNS},
              'selection':{'selected_candidate':'synthetic_test_only'},'dataset_audit':audit}
    (bundle/'metadata.json').write_text(json.dumps(metadata))
    report=evaluate_bundle(data,splits,bundle,tmp_path/'evaluation',install=True)
    loaded,meta=load_bundle(bundle)
    result=predict_records(loaded,meta,device())
    assert result['prediction_interval'] is not None
    assert report['data_quality']['status']=='computed'
    protocol=json.loads((tmp_path/'evaluation/protocol.json').read_text())
    assert set(protocol['calibration_source_rows']).isdisjoint(protocol['evaluation_source_rows'])
    assert (tmp_path/'evaluation/COMPLETE').exists()
