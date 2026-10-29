"""Integration tests require a locally generated, trusted app bundle."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from src.predict import load_bundle

ROOT=Path(__file__).resolve().parents[1]


def test_missing_bundle_is_actionable(monkeypatch,tmp_path):
    monkeypatch.setenv('PEROVSKITE_BUNDLE',str(tmp_path/'missing'))
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
    assert not app.exception and len(app.error)==1
    assert 'Could not load' in app.error[0].value


@pytest.mark.skipif(not (ROOT/'models/app_bundle/metadata.json').exists(),reason='Run python -m src.predict prepare for the integration bundle')
@pytest.mark.parametrize('entrypoint', ['app.py', 'deploy/streamlit_app.py'])
def test_app_predict_and_explain(monkeypatch, entrypoint):
    monkeypatch.setenv('PEROVSKITE_BUNDLE',str(ROOT/'models/app_bundle'))
    app=AppTest.from_file(str(ROOT/entrypoint),default_timeout=120).run()
    from src.data_preprocessing import FEATURE_COLUMNS
    assert not app.exception and all(app.selectbox(key=field).help for field in FEATURE_COLUMNS)
    app.button(key='launch_prediction').click().run()
    assert not app.exception and not app.error

    # Pausing decorative motion must retain the submitted model result.
    saved = app.session_state['prediction_result']
    app.toggle(key='pause_animations').set_value(True).run()
    assert not app.exception and not app.error
    assert app.session_state['prediction_result'] == saved
    result=app.session_state['prediction_result']['result']
    assert any(metric.label=='Estimated PCE' for metric in app.metric)
    assert result['prediction_interval'] is not None if (ROOT/'models/app_bundle/analytics.json').exists() else result['prediction_interval'] is None
    assert abs(result['SHAP_base_PCE_percent'][0]+sum(result['SHAP_grouped_contributions'][0].values())-result['predicted_PCE_percent'][0])<1e-5
    field=app.selectbox(key='Encapsulation')
    field.select(1.0 if field.value==0 else 0.0).run()
    if app.checkbox:
        app.checkbox(key='review_device_checks').check().run()
    app.button(key='launch_prediction').click().run()
    assert not app.exception and not app.error
    assert result['explanation_summary'][0]


@pytest.mark.skipif(not (ROOT/'models/app_bundle/metadata.json').exists(),reason='Requires trusted bundle')
def test_preset_restores_whole_device_and_conflict_requires_review(monkeypatch):
    from src.data_preprocessing import FEATURE_COLUMNS
    monkeypatch.setenv('PEROVSKITE_BUNDLE',str(ROOT/'models/app_bundle'))
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
    _, metadata=load_bundle(ROOT/'models/app_bundle')
    default=metadata['default_input']
    wrong_architecture='pin' if default['Cell_architecture']=='nip' else 'nip'
    app.selectbox(key='Cell_architecture').select(wrong_architecture).run()
    assert app.button(key='launch_prediction').disabled
    app.checkbox(key='review_device_checks').check().run()
    assert not app.button(key='launch_prediction').disabled
    app.selectbox(key='device_preset').select(0).run()
    assert not app.exception
    expected=(metadata.get('analytics',{}).get('training_presets') or metadata.get('training_presets') or [{'input':default}])[0]['input']
    assert all(app.selectbox(key=c).value==expected[c] for c in FEATURE_COLUMNS)


def test_bundle_rejects_mismatched_analytics(tmp_path):
    import json
    import shutil
    if not (ROOT/'models/app_bundle/metadata.json').exists():
        pytest.skip('Requires trusted bundle')
    for name in ['metadata.json','pipeline.joblib']:
        shutil.copyfile(ROOT/'models/app_bundle'/name,tmp_path/name)
    (tmp_path/'analytics.json').write_text(json.dumps({'model_sha256':'wrong','dataset_sha256':'wrong'}))
    with pytest.raises(ValueError,match='Analytics artifact'):
        load_bundle(tmp_path)


def test_bundle_checksum_rejection(tmp_path):
    import json
    from src.data_preprocessing import FEATURE_COLUMNS
    (tmp_path/'pipeline.joblib').write_bytes(b'broken artifact')
    (tmp_path/'metadata.json').write_text(json.dumps({'feature_columns':FEATURE_COLUMNS,'model_sha256':'wrong'}))
    with pytest.raises(ValueError,match='checksum'): load_bundle(tmp_path)
