"""Portable training-derived application bundle and validated PCE inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from .data_preprocessing import FEATURE_COLUMNS, MISSING_CATEGORY
from .input_guidance import check_device, explanation_summary
from .uncertainty import prediction_intervals

ROOT=Path(__file__).resolve().parents[1]
BOOL_COLUMNS=['Encapsulation','Perovskite_deposition_quenching_induced_crystallisation']
STEP_COLUMN='Perovskite_deposition_number_of_deposition_steps'
NUMERIC_COLUMNS=BOOL_COLUMNS+[STEP_COLUMN]


def normalize_input(records):
    """Accept complete raw records, preserving chemical strings and missingness."""
    if isinstance(records,dict): records=[records]
    if isinstance(records,pd.DataFrame):
        if not records.columns.is_unique: raise ValueError('Duplicate feature columns')
        records=records.to_dict(orient='records')
    if not isinstance(records,list) or not records or len(records)>512:
        raise ValueError('Provide 1–512 complete input records')
    normalized=[]
    for record in records:
        if not isinstance(record,dict): raise ValueError('Each input must be an object')
        missing=set(FEATURE_COLUMNS)-set(record); extra=set(record)-set(FEATURE_COLUMNS)
        if missing or extra:
            raise ValueError(f'Input schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
        row={}
        for column in FEATURE_COLUMNS:
            value=record[column]
            if isinstance(value,(list,dict,tuple,set)):
                raise ValueError(f'{column} must be a scalar value')
            if value is None or (not isinstance(value,str) and bool(pd.isna(value))):
                row[column]=np.nan if column in NUMERIC_COLUMNS else MISSING_CATEGORY
            elif column in NUMERIC_COLUMNS:
                if isinstance(value,str) or not isinstance(value,(int,float,bool,np.number)):
                    raise ValueError(f'{column} must be numeric or null')
                number=float(value)
                if not np.isfinite(number): raise ValueError(f'{column} must be finite')
                if column in BOOL_COLUMNS and number not in (0.,1.):
                    raise ValueError(f'{column} must be true/false or 0/1')
                if column==STEP_COLUMN and (number<0 or not number.is_integer()):
                    raise ValueError('Deposition step count must be a nonnegative integer')
                row[column]=number
            else:
                if not isinstance(value,str): raise ValueError(f'{column} must be a string or null')
                row[column]=value  # Never silently strip or reinterpret chemistry/units.
        normalized.append(row)
    return pd.DataFrame(normalized,columns=FEATURE_COLUMNS)


def load_bundle(path=ROOT/'models/app_bundle'):
    """Load only trusted local joblib files; checksum guards accidental mismatch."""
    path=Path(path)
    metadata=json.loads((path/'metadata.json').read_text())
    if metadata.get('feature_columns')!=FEATURE_COLUMNS: raise ValueError('Bundle input schema mismatch')
    actual=hashlib.sha256((path/'pipeline.joblib').read_bytes()).hexdigest()
    if actual!=metadata['model_sha256']: raise ValueError('Bundle model checksum mismatch')
    pipeline=joblib.load(path/'pipeline.joblib')
    if (path/'analytics.json').exists():
        analytics=json.loads((path/'analytics.json').read_text(encoding='utf-8'))
        if analytics.get('model_sha256')!=actual or analytics.get('dataset_sha256')!=metadata.get('dataset_sha256'):
            raise ValueError('Analytics artifact does not match this model and dataset')
        metadata['analytics']=analytics
    return pipeline,metadata


def predict_records(pipeline,metadata,records,explain=False):
    raw=normalize_input(records)
    prediction=np.asarray(pipeline.predict(raw),dtype=float)
    if not np.isfinite(prediction).all(): raise ValueError('Model returned nonfinite predictions')
    warnings=[]
    for column in FEATURE_COLUMNS:
        unseen=set(raw[column].dropna())-set(metadata['training_options'][column])
        if unseen: warnings.append(f'{column}: {len(unseen)} value(s) absent from training data')
    result={'predicted_PCE_percent':prediction.tolist(),'warnings':warnings,
            'notice':'Model estimates, not experimentally verified efficiencies. SHAP describes associations, not causal effects.'}
    result['device_checks']=[check_device(row) for row in raw.to_dict(orient='records')]
    intervals=prediction_intervals(prediction,metadata.get('analytics'),metadata.get('model_sha256'))
    result['prediction_interval']=intervals
    result['uncertainty_status']='evaluated' if intervals else 'unavailable: run research validation with the original source data'
    if explain:
        from .explainability import explain_prediction
        explained=explain_prediction(pipeline,raw)
        result['SHAP_base_PCE_percent']=explained['base_values'].tolist()
        result['SHAP_grouped_contributions']=[dict(zip(FEATURE_COLUMNS,map(float,row))) for row in explained['grouped_values']]
        result['explanation_summary']=[explanation_summary(contrib,row) for contrib,row in zip(result['SHAP_grouped_contributions'],raw.to_dict(orient='records'))]
    return result


def prepare_bundle(data_path,splits,run,phase3,output):
    from .train_models import load_saved_data
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    if not (run/'COMPLETE').exists(): raise ValueError('Phase 4 optimization is incomplete')
    X,y,source,indices,digest=load_saved_data(data_path,splits)
    decision=json.loads((run/'selection.json').read_text())
    if digest!=decision['dataset_sha256']: raise ValueError('Selection source checksum mismatch')
    train=X.iloc[indices['train']]
    options={c:train[c].value_counts(dropna=True).index.tolist() for c in FEATURE_COLUMNS}
    defaults=train.iloc[0].to_dict()  # A real training configuration, not independent column modes.
    output.mkdir(parents=True)
    shutil.copyfile(run/'models/selected_pipeline.joblib',output/'pipeline.joblib')
    phase3_config=json.loads((phase3/'run_config.json').read_text())
    if phase3_config['dataset_sha256']!=digest or phase3_config['quick_smoke_only']:
        raise ValueError('Comparison report must be a full run on the same dataset')
    comparison=pd.read_csv(phase3/'validation_comparison.csv')
    metadata={'feature_columns':FEATURE_COLUMNS,'training_options':options,'default_input':defaults,
              'default_source_row':int(source[indices['train'][0]]),'dataset_sha256':digest,
              'model_sha256':hashlib.sha256((output/'pipeline.joblib').read_bytes()).hexdigest(),
              'selection':decision,'dataset_audit':json.loads((splits/'audit.json').read_text()),
              'phase3_validation_comparison':comparison.to_dict(orient='records'),
              'phase3_test_metrics':json.loads((phase3/'test_metrics.json').read_text()),
              'phase4_validation_comparison':pd.read_csv(run/'validation_comparison.csv').to_dict(orient='records')}
    metadata['training_presets']=make_training_presets(train,source[indices['train']])
    # Some incumbent rows have no duration, so replace pandas NaNs for strict JSON.
    metadata['phase4_validation_comparison']=json.loads(pd.DataFrame(metadata['phase4_validation_comparison']).to_json(orient='records'))
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    (output/'example_input.json').write_text(json.dumps(defaults,indent=2,allow_nan=False)+'\n')
    pipeline,loaded=load_bundle(output)
    result=predict_records(pipeline,loaded,defaults,explain=True)
    assert len(result['predicted_PCE_percent'])==1
    print(f'Bundle saved to {output}; training-derived options and end-to-end inference verified.',flush=True)


def make_training_presets(train, source_rows):
    """Choose whole real training rows, never independent modes or synthetic devices."""
    presets=[]
    for architecture in train['Cell_architecture'].value_counts().head(3).index:
        positions=np.flatnonzero(train['Cell_architecture'].eq(architecture).to_numpy())
        position=int(positions[0])
        presets.append({'name':f'Recorded {architecture} device · source row {int(source_rows[position])}',
                        'source_row':int(source_rows[position]),
                        'input':{c:None if pd.isna(v) else v for c,v in train.iloc[position].to_dict().items()}})
    return presets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare')
    prepare.add_argument('--data',type=Path,default=ROOT.parent/'Solar Cell data set2.csv')
    prepare.add_argument('--splits',type=Path,default=ROOT/'data/processed/phase1')
    prepare.add_argument('--run',type=Path,default=ROOT/'reports/phase4_run_001')
    prepare.add_argument('--phase3',type=Path,default=ROOT/'reports/phase3_run_001')
    prepare.add_argument('--output',type=Path,default=ROOT/'models/app_bundle')
    predict=commands.add_parser('predict')
    predict.add_argument('--bundle',type=Path,default=ROOT/'models/app_bundle')
    predict.add_argument('--input',type=Path,required=True)
    predict.add_argument('--explain',action='store_true')
    args=parser.parse_args()
    try:
        if args.command=='prepare': prepare_bundle(args.data,args.splits,args.run,args.phase3,args.output)
        else:
            pipeline,metadata=load_bundle(args.bundle)
            print(json.dumps(predict_records(pipeline,metadata,json.loads(args.input.read_text()),args.explain),indent=2))
    except (ValueError,FileNotFoundError,FileExistsError) as exc:
        parser.exit(1,f'Prediction setup failed: {exc}\n')

if __name__=='__main__': main()
