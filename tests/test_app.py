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
def test_app_predict_and_explain(monkeypatch):
    monkeypatch.setenv('PEROVSKITE_BUNDLE',str(ROOT/'models/app_bundle'))
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
    assert not app.exception and len(app.selectbox)==25
    app.button[0].click().run()
    assert not app.exception and not app.error

    # Pausing decorative motion must retain the submitted model result.
    saved = app.session_state['prediction_result']
    app.toggle(key='pause_animations').set_value(True).run()
    assert not app.exception and not app.error
    assert app.session_state['prediction_result'] == saved
    result=app.session_state['prediction_result']['result']
    assert any(metric.label=='Estimated PCE' for metric in app.metric)
    assert abs(result['SHAP_base_PCE_percent'][0]+sum(result['SHAP_grouped_contributions'][0].values())-result['predicted_PCE_percent'][0])<1e-5
    field=app.selectbox(key='Encapsulation')
    field.select(1.0 if field.value==0 else 0.0).run()
    app.button[0].click().run()
    assert not app.exception and not app.error


def test_bundle_checksum_rejection(tmp_path):
    import json
    from src.data_preprocessing import FEATURE_COLUMNS
    (tmp_path/'pipeline.joblib').write_bytes(b'broken artifact')
    (tmp_path/'metadata.json').write_text(json.dumps({'feature_columns':FEATURE_COLUMNS,'model_sha256':'wrong'}))
    with pytest.raises(ValueError,match='checksum'): load_bundle(tmp_path)
