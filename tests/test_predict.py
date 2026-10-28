import numpy as np
import pandas as pd
import pytest
from src.data_preprocessing import FEATURE_COLUMNS,MISSING_CATEGORY
from src.predict import normalize_input,predict_records,BOOL_COLUMNS,STEP_COLUMN


def record():
    row={c:'Unknown' for c in FEATURE_COLUMNS}
    row.update({c:False for c in BOOL_COLUMNS}); row[STEP_COLUMN]=1
    return row


def test_normalization_preserves_units_and_missingness():
    row=record(); row['Perovskite_thickness']='100; 200 nm'; row['Perovskite_additives_compounds']=None
    X=normalize_input(row)
    assert X.loc[0,'Perovskite_thickness']=='100; 200 nm'
    assert X.loc[0,'Perovskite_additives_compounds']==MISSING_CATEGORY
    assert X.loc[0,'Encapsulation']==0
    row[STEP_COLUMN]=None
    assert np.isnan(normalize_input(row).loc[0,STEP_COLUMN])
    pd.testing.assert_frame_equal(normalize_input(X),X)


@pytest.mark.parametrize('field,value',[(STEP_COLUMN,-1),(STEP_COLUMN,1.5),(STEP_COLUMN,np.inf),
    ('Encapsulation','False'),('Encapsulation',2),('Perovskite_thickness',100),('Cell_architecture',['nip'])])
def test_invalid_input_rejected(field,value):
    row=record(); row[field]=value
    with pytest.raises(ValueError): normalize_input(row)


def test_input_schema_and_size():
    row=record()
    with pytest.raises(ValueError): normalize_input({**row,'JV_default_PCE':15})
    row.pop('Cell_architecture')
    with pytest.raises(ValueError): normalize_input(row)
    with pytest.raises(ValueError): normalize_input([])
    with pytest.raises(ValueError): normalize_input([record()]*513)


def test_unseen_values_warn_without_rejecting():
    class Pipeline:
        def predict(self,X): return np.full(len(X),12.)
    row=record(); metadata={'training_options':{c:[v] for c,v in row.items()}}
    row['Cell_architecture']='new architecture'
    result=predict_records(Pipeline(),metadata,row)
    assert result['predicted_PCE_percent']==[12.]
    assert any('Cell_architecture' in warning for warning in result['warnings'])
