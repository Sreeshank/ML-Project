"""CPU comparison on saved configuration splits; select on validation RMSE only."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from .data_preprocessing import (TARGET, SEED, load_dataset, clean_dataset,
                                 make_features, split_groups, build_preprocessor)
from .feature_engineering import PerovskiteFeatureEngineer
from .evaluate_models import regression_metrics, save_diagnostics, save_model_importance


def model_candidates(threads=4, quick=False):
    trees=8 if quick else 120
    boosting=12 if quick else 500
    return {
        'linear_regression':LinearRegression(n_jobs=threads,tol=1e-4),
        'random_forest':RandomForestRegressor(n_estimators=trees,max_depth=16,
            min_samples_leaf=3,max_features=.1,n_jobs=threads,random_state=SEED),
        'xgboost':XGBRegressor(n_estimators=boosting,max_depth=6,learning_rate=.05,
            min_child_weight=5,subsample=.8,colsample_bytree=.8,tree_method='hist',
            max_bin=128,n_jobs=threads,random_state=SEED,objective='reg:squarederror'),
        'lightgbm':LGBMRegressor(n_estimators=boosting,num_leaves=31,learning_rate=.05,
            min_child_samples=20,colsample_bytree=.8,n_jobs=threads,random_state=SEED,
            verbosity=-1,deterministic=True,force_col_wise=True),
        'catboost':CatBoostRegressor(iterations=boosting,depth=6,learning_rate=.05,
            l2_leaf_reg=5,loss_function='RMSE',random_seed=SEED,thread_count=threads,
            verbose=False,allow_writing_files=False),
    }


def load_saved_data(data_path, split_dir):
    audit=json.loads((split_dir/'audit.json').read_text())
    digest=hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != audit['dataset_sha256']:
        raise ValueError('Dataset checksum does not match saved splits')
    clean,source,audit_clean=clean_dataset(load_dataset(data_path))
    X,cats=make_features(clean)
    expected,groups=split_groups(X)
    with np.load(split_dir/'split_indices.npz') as saved:
        indices={k:saved[k].copy() for k in ['train','validation','test']}
    for k in indices:
        if not np.array_equal(expected[k],indices[k]):
            raise ValueError(f'Saved {k} split differs from reproducible Phase 1 split')
    return X,clean[TARGET],source,indices,digest


def train(data_path, split_dir, output, threads=4, quick=False):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not 1<=threads<=4:
        raise ValueError('threads must be between 1 and 4')
    X,y,source,indices,digest=load_saved_data(data_path,split_dir)
    output.mkdir(parents=True)
    models_dir=output/'models'; models_dir.mkdir()
    tr,va,te=(indices[k] for k in ['train','validation','test'])
    # Fit engineer afresh on training rows only; same bounded settings as Phase 2.
    engineer=PerovskiteFeatureEngineer().fit(X.iloc[tr])
    engineered_train=engineer.transform(X.iloc[tr])
    engineered_val=engineer.transform(X.iloc[va])
    cats=engineered_train.select_dtypes(include=['object','string','category']).columns.tolist()
    pre=build_preprocessor(engineered_train,cats)
    sparse_train=pre.fit_transform(engineered_train)
    sparse_val=pre.transform(engineered_val)
    baseline=DummyRegressor().fit(np.zeros((len(tr),1)),y.iloc[tr])
    rows=[dict(model='mean_baseline',**regression_metrics(y.iloc[va],baseline.predict(np.zeros((len(va),1)))),fit_seconds=0)]
    candidates=model_candidates(threads,quick)
    run={'seed':SEED,'quick_smoke_only':quick,'threads':threads,'dataset_sha256':digest,
         'split_rows':{k:len(v) for k,v in indices.items()},
         'feature_count':engineered_train.shape[1],'one_hot_shape':list(sparse_train.shape),
         'selection_rule':'Minimum validation RMSE among the five requested algorithms',
         'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','xgboost','lightgbm','catboost','joblib']},
         'model_parameters':{k:m.get_params() for k,m in candidates.items()}}
    (output/'run_config.json').write_text(json.dumps(run,indent=2,default=str)+'\n')
    with threadpool_limits(limits=threads):
        for name,model in candidates.items():
            start=time.monotonic()
            if name=='catboost':
                model.fit(engineered_train,y.iloc[tr],cat_features=cats,
                    eval_set=(engineered_val,y.iloc[va]),early_stopping_rounds=50)
                predictions=model.predict(engineered_val)
                pipeline=Pipeline([('features',engineer),('model',model)])
            else:
                # Fixed budgets, no test-dependent tuning. All algorithms see the
                # same raw+engineered information, with model-appropriate encoding.
                model.fit(sparse_train,y.iloc[tr])
                predictions=model.predict(sparse_val)
                pipeline=Pipeline([('features',engineer),('preprocessor',pre),('model',model)])
            elapsed=time.monotonic()-start
            row=dict(model=name,**regression_metrics(y.iloc[va],predictions),fit_seconds=elapsed)
            rows.append(row)
            joblib.dump(pipeline,models_dir/f'{name}.joblib',compress=3)
            pd.DataFrame(rows).to_csv(output/'validation_comparison.csv',index=False)
            print(json.dumps(row),flush=True)
    selected=min(rows[1:],key=lambda row:row['RMSE'])['model']
    # Freeze the decision on disk before loading or predicting test inputs.
    decision={'selected_model':selected,'validation_metrics':next(r for r in rows if r['model']==selected),
              'refit':False,'note':'Selected fitted training pipeline; no train+validation refit.'}
    (output/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    pipeline=joblib.load(models_dir/f'{selected}.joblib')
    predictions=pipeline.predict(X.iloc[te])
    results={selected:regression_metrics(y.iloc[te],predictions),
             'mean_baseline':regression_metrics(y.iloc[te],baseline.predict(np.zeros((len(te),1))))}
    (output/'test_metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    save_diagnostics(y.iloc[te],predictions,X.iloc[te],source[te],output)
    save_model_importance(pipeline,output)
    joblib.dump(pipeline,models_dir/'selected_pipeline.joblib',compress=3)
    # Prove the persisted full inference pipeline reproduces predictions.
    restored=joblib.load(models_dir/'selected_pipeline.joblib')
    assert np.allclose(restored.predict(X.iloc[va[:8]]),pipeline.predict(X.iloc[va[:8]]))
    (output/'COMPLETE').write_text('Training, selection, test evaluation, and serialization check succeeded.\n')
    print(json.dumps({'selected_model':selected,'test':results},indent=2),flush=True)
    return results


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=root.parent/'Solar Cell data set2.csv')
    parser.add_argument('--splits',type=Path,default=root/'data/processed/phase1')
    parser.add_argument('--output',type=Path,default=root/'reports/phase3_run_001')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--quick',action='store_true',help='Smoke budgets, not research results')
    args=parser.parse_args()
    train(args.data,args.splits,args.output,args.threads,args.quick)

if __name__=='__main__':
    main()
