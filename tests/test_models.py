import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.data_preprocessing import FEATURE_COLUMNS, build_preprocessor
from src.feature_engineering import PerovskiteFeatureEngineer
from src.train_models import model_candidates
from src.evaluate_models import regression_metrics


def test_metrics_known_values():
    result=regression_metrics([1.,2.,3.],[1.,2.,3.])
    assert result=={'MAE':0.,'RMSE':0.,'R2':1.}
    with pytest.raises(ValueError): regression_metrics([1.,2.],[np.nan,1.])


@pytest.mark.parametrize('name',list(model_candidates(1,quick=True)))
def test_algorithm_pipeline_roundtrip_and_unseen_categories(name,tmp_path):
    X=pd.DataFrame({c:['known']*48 for c in FEATURE_COLUMNS})
    X['Perovskite_deposition_number_of_deposition_steps']=np.arange(48)%3
    X['Perovskite_thickness']=[f'{i+100} nm' for i in range(48)]
    y=np.arange(48)*.1+5
    engineer=PerovskiteFeatureEngineer().fit(X)
    engineered=engineer.transform(X)
    cats=engineered.select_dtypes(include='object').columns.tolist()
    model=model_candidates(1,quick=True)[name]
    if name=='catboost':
        model.fit(engineered,y,cat_features=cats)
        pipeline=Pipeline([('features',engineer),('model',model)])
    else:
        pre=build_preprocessor(engineered,cats)
        model.fit(pre.fit_transform(engineered),y)
        pipeline=Pipeline([('features',engineer),('preprocessor',pre),('model',model)])
    unseen=X.iloc[:2].copy()
    unseen['Cell_architecture']='unseen architecture'
    predictions=pipeline.predict(unseen)
    path=tmp_path/f'{name}.joblib'; joblib.dump(pipeline,path)
    assert np.isfinite(predictions).all()
    np.testing.assert_allclose(joblib.load(path).predict(unseen),predictions)


def test_training_loader_rejects_changed_data_and_splits(tmp_path):
    from src.data_preprocessing import EXPECTED_COLUMNS, prepare
    from src.train_models import load_saved_data
    X=pd.DataFrame({c:['known']*40 for c in EXPECTED_COLUMNS})
    X['Cell_stack_sequence']=[f'layer_{i}' for i in range(40)]
    X['JV_default_PCE']=np.arange(40,dtype=float)
    path=tmp_path/'data.csv'; X.to_csv(path,index=False)
    splits=tmp_path/'splits'; prepare(path,splits)
    load_saved_data(path,splits)
    content=path.read_bytes()
    path.write_bytes(content+b'\n')
    with pytest.raises(ValueError,match='checksum'): load_saved_data(path,splits)
    path.write_bytes(content)
    with np.load(splits/'split_indices.npz') as saved:
        indices={key:saved[key].copy() for key in saved.files}
    indices['train']=indices['train'][::-1]
    np.savez_compressed(splits/'split_indices.npz',**indices)
    with pytest.raises(ValueError,match='split differs'): load_saved_data(path,splits)
