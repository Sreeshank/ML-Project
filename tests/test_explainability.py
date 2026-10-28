import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.data_preprocessing import FEATURE_COLUMNS,build_preprocessor
from src.feature_engineering import PerovskiteFeatureEngineer
from src.train_models import model_candidates
from src.explainability import explain_prediction,source_feature,raw_permutation

@pytest.mark.parametrize('family',['lightgbm','catboost'])
def test_shap_reconstructs_selected_full_pipeline(family):
    X=pd.DataFrame({c:['known']*48 for c in FEATURE_COLUMNS})
    X['Perovskite_deposition_number_of_deposition_steps']=np.arange(48)%3
    X['Perovskite_thickness']=[f'{100+i} nm' for i in range(48)]
    y=np.arange(48)*.2+3
    engineer=PerovskiteFeatureEngineer().fit(X)
    E=engineer.transform(X); model=model_candidates(1,True)[family]
    if family=='catboost':
        model.fit(E,y,cat_features=E.select_dtypes(include='object').columns.tolist())
        pipe=Pipeline([('features',engineer),('model',model)])
    else:
        pre=build_preprocessor(E,E.select_dtypes(include='object').columns.tolist())
        model.fit(pre.fit_transform(E),y)
        pipe=Pipeline([('features',engineer),('preprocessor',pre),('model',model)])
    explained=explain_prediction(pipe,X.iloc[:3])
    np.testing.assert_allclose(explained['base_values']+explained['values'].sum(axis=1),pipe.predict(X.iloc[:3]),atol=1e-5)
    np.testing.assert_allclose(explained['base_values']+explained['grouped_values'].sum(axis=1),pipe.predict(X.iloc[:3]),atol=1e-5)
    assert explained['grouped_values'].shape==(3,25)
    importance=raw_permutation(pipe,X.iloc[:6],y[:6],repeats=1)
    assert len(importance)==25 and np.isfinite(importance.RMSE_increase_mean).all()


def test_feature_origin_mapping():
    assert source_feature('xai__a_site__token=MA')=='Perovskite_composition_a_ions'
    assert source_feature('missingindicator_xai__anneal_time__unitless_mean')=='Perovskite_deposition_thermal_annealing_time'
    with pytest.raises(ValueError): source_feature('target')
