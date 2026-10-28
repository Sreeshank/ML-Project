"""Small sequential Optuna search on saved train/validation partitions only."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from .train_models import load_saved_data
from .evaluate_models import regression_metrics


def optimize(data_path, splits, previous, output, trials=3, threads=4):
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    if trials<1 or not 1<=threads<=4: raise ValueError('Positive trials and 1–4 threads required')
    if not (previous/'COMPLETE').exists(): raise ValueError('Phase 3 run must be complete')
    X,y,_,indices,digest=load_saved_data(data_path,splits)
    config=json.loads((previous/'run_config.json').read_text())
    if config['dataset_sha256']!=digest or config['quick_smoke_only']:
        raise ValueError('Require a completed non-smoke Phase 3 run on this dataset')
    output.mkdir(parents=True); (output/'models').mkdir()
    tr,va=indices['train'],indices['validation']
    # Reuse exactly the fitted Phase 3 training transformations, never refit on val.
    incumbents={name:joblib.load(previous/'models'/f'{name}.joblib') for name in ['lightgbm','catboost']}
    engineer=incumbents['lightgbm'].named_steps['features']
    pre=incumbents['lightgbm'].named_steps['preprocessor']
    et=engineer.transform(X.iloc[tr]); ev=engineer.transform(X.iloc[va])
    st=pre.transform(et); sv=pre.transform(ev)
    cats=et.select_dtypes(include=['object','string','category']).columns.tolist()
    rows=[]
    for name,pipeline in incumbents.items():
        row=dict(candidate=f'phase3_{name}',family=name,**regression_metrics(y.iloc[va],pipeline.predict(X.iloc[va])) )
        rows.append(row); joblib.dump(pipeline,output/'models'/f'phase3_{name}.joblib',compress=3)
    studies={}
    with threadpool_limits(limits=threads):
        for family in ['lightgbm','catboost']:
            study=optuna.create_study(study_name=family,direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                storage=f'sqlite:///{(output/"optuna.sqlite3").resolve()}')
            def objective(trial):
                start=time.monotonic()
                if family=='lightgbm':
                    model=LGBMRegressor(n_estimators=700,learning_rate=trial.suggest_float('learning_rate',.025,.09,log=True),
                        num_leaves=trial.suggest_int('num_leaves',20,55),min_child_samples=trial.suggest_int('min_child_samples',15,60),
                        reg_lambda=trial.suggest_float('reg_lambda',.1,10,log=True),colsample_bytree=.8,
                        random_state=42,n_jobs=threads,verbosity=-1,deterministic=True,force_col_wise=True)
                    model.fit(st,y.iloc[tr],eval_set=[(sv,y.iloc[va])],callbacks=[early_stopping(50,verbose=False)])
                    pred=model.predict(sv)
                    pipeline=Pipeline([('features',engineer),('preprocessor',pre),('model',model)])
                else:
                    model=CatBoostRegressor(iterations=700,depth=trial.suggest_int('depth',4,7),
                        learning_rate=trial.suggest_float('learning_rate',.025,.09,log=True),
                        l2_leaf_reg=trial.suggest_float('l2_leaf_reg',2,12,log=True),loss_function='RMSE',
                        random_seed=42,thread_count=threads,verbose=False,allow_writing_files=False)
                    model.fit(et,y.iloc[tr],cat_features=cats,eval_set=(ev,y.iloc[va]),early_stopping_rounds=50)
                    pred=model.predict(ev)
                    pipeline=Pipeline([('features',engineer),('model',model)])
                name=f'{family}_trial_{trial.number}'
                metrics=regression_metrics(y.iloc[va],pred)
                rows.append(dict(candidate=name,family=family,**metrics,fit_seconds=time.monotonic()-start))
                joblib.dump(pipeline,output/'models'/f'{name}.joblib',compress=3)
                trial.set_user_attr('validation_metrics',metrics)
                print(json.dumps(rows[-1]),flush=True)
                return metrics['RMSE']
            study.optimize(objective,n_trials=trials,timeout=360,n_jobs=1)
            study.trials_dataframe().to_csv(output/f'{family}_trials.csv',index=False)
            studies[family]=dict(completed_trials=len(study.trials),best_params=study.best_params)
    import pandas as pd
    pd.DataFrame(rows).sort_values('RMSE').to_csv(output/'validation_comparison.csv',index=False)
    winner=min(rows,key=lambda r:r['RMSE'])
    pipeline=joblib.load(output/'models'/f'{winner["candidate"]}.joblib')
    joblib.dump(pipeline,output/'models/selected_pipeline.joblib',compress=3)
    decision=dict(selected_candidate=winner['candidate'],family=winner['family'],validation_metrics=winner,
                  dataset_sha256=digest,previous_run=str(previous.resolve()),studies=studies,
                  seed=42,threads=threads,requested_trials_per_family=trials,
                  test_evaluated=False,note='No test predictions or metrics generated during Phase 4 tuning.')
    (output/'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    assert np.allclose(pipeline.predict(X.iloc[va[:3]]),joblib.load(output/'models/selected_pipeline.joblib').predict(X.iloc[va[:3]]))
    (output/'COMPLETE').write_text('Validation selection and serialization checks complete; no test evaluation.\n')
    print(json.dumps(decision,indent=2),flush=True)


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=root.parent/'Solar Cell data set2.csv')
    parser.add_argument('--splits',type=Path,default=root/'data/processed/phase1')
    parser.add_argument('--previous',type=Path,default=root/'reports/phase3_run_001')
    parser.add_argument('--output',type=Path,default=root/'reports/phase4_run_001')
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(); optimize(args.data,args.splits,args.previous,args.output,args.trials,args.threads)

if __name__=='__main__': main()
