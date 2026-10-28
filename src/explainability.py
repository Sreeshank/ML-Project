"""SHAP and raw-input permutation explanations of the selected fitted pipeline."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from .data_preprocessing import FEATURE_COLUMNS
from .feature_engineering import TOKEN_FIELDS, MEASURE_FIELDS


def source_feature(name):
    """Attribute engineered contributions to their single original source field."""
    if name in FEATURE_COLUMNS: return name
    if name.startswith('missingindicator_'): return source_feature(name[len('missingindicator_'):])
    if not name.startswith('xai__'): raise ValueError(f'Unknown engineered feature: {name}')
    rest=name[len('xai__'):]
    for column in FEATURE_COLUMNS:
        if rest==column+'__missing': return column
    mapping={**{k:v for k,v in TOKEN_FIELDS.items()},
             **{k:v[0] for k,v in MEASURE_FIELDS.items()},
             'device':'Cell_stack_sequence','etl':'ETL_stack_sequence','htl':'HTL_stack_sequence'}
    for prefix,column in mapping.items():
        if rest.startswith(prefix+'__'): return column
    raise ValueError(f'Unknown feature origin: {name}')


def transformed_origins(pipeline):
    if 'preprocessor' not in pipeline.named_steps:
        return [source_feature(c) for c in pipeline.named_steps['features'].get_feature_names_out()]
    pre=pipeline.named_steps['preprocessor']
    origins=[]
    # Match ColumnTransformer's actual output order, including imputer indicators.
    for name,transformer,columns in pre.transformers_:
        if name=='categorical':
            encoder=transformer.named_steps['encode']
            if encoder.drop is not None: raise ValueError('Dropped OHE levels are not supported')
            for column,categories in zip(columns,encoder.categories_):
                origins.extend([source_feature(column)]*len(categories))
        elif name=='numeric':
            names=transformer.get_feature_names_out(columns)
            origins.extend(source_feature(c) for c in names)
        elif name=='remainder' and transformer=='drop': continue
        else: raise ValueError('Unsupported preprocessing structure')
    return origins


def explain_prediction(pipeline, raw):
    """Exact tree contributions after the saved preprocessing; supports both winners.

    Returns feature-level and raw-source-grouped SHAP explanations. Grouping sums
    contributions of engineered and encoded descendants; it is not a new SHAP
    game over independent raw inputs and does not imply causal effects.
    """
    if len(raw)==0: raise ValueError('At least one row is required')
    engineered=pipeline.named_steps['features'].transform(raw)
    model=pipeline.named_steps['model']
    if 'preprocessor' in pipeline.named_steps:
        pre=pipeline.named_steps['preprocessor']
        transformed=pre.transform(engineered)
        dense=transformed.toarray() if hasattr(transformed,'toarray') else np.asarray(transformed)
        names=pre.get_feature_names_out()
        explanation=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent')(dense,check_additivity=True)
        values=np.asarray(explanation.values)
        base=np.broadcast_to(explanation.base_values,(len(raw),)).copy()
        plotting_data=dense
    else:
        from catboost import Pool
        cats=engineered.select_dtypes(include=['object','string','category']).columns.tolist()
        contributions=model.get_feature_importance(Pool(engineered,cat_features=cats),type='ShapValues')
        values=contributions[:,:-1]; base=contributions[:,-1]
        names=engineered.columns.to_numpy(); plotting_data=engineered.to_numpy()
    prediction=pipeline.predict(raw)
    if not np.allclose(base+values.sum(axis=1),prediction,atol=1e-5):
        raise AssertionError('SHAP contributions do not reconstruct pipeline predictions')
    origins=transformed_origins(pipeline)
    if len(origins)!=values.shape[1]: raise AssertionError('Feature origin mapping is misaligned')
    grouped=np.column_stack([values[:,np.array(origins)==c].sum(axis=1) for c in FEATURE_COLUMNS])
    assert np.allclose(grouped.sum(axis=1)+base,prediction,atol=1e-5)
    return dict(values=values,base_values=base,predictions=prediction,names=np.asarray(names),
                plotting_data=plotting_data,grouped_values=grouped)


def raw_permutation(pipeline,X,y,repeats=3):
    """Permute raw fields, then recompute engineering and model transformations."""
    result=permutation_importance(pipeline,X,y,n_repeats=repeats,random_state=42,
                                 scoring='neg_root_mean_squared_error',n_jobs=1)
    return pd.DataFrame({'feature':X.columns,'RMSE_increase_mean':result.importances_mean,
                         'RMSE_increase_std':result.importances_std}).sort_values('RMSE_increase_mean',ascending=False)


def plot_label(name):
    """Keep scientific plots readable while full names remain in CSV artifacts."""
    import textwrap
    name=str(name).replace('categorical__','').replace('numeric__','')
    name=name.replace('Perovskite_', 'PV ').replace('deposition_', '').replace('xai__','')
    name=name.replace('__',' / ').replace('_',' ')
    return textwrap.shorten(name,width=65,placeholder='…')


def save_shap_plots(result, raw, output):
    values=result['values']; base=result['base_values']
    labels=[plot_label(name) for name in result['names']]
    explanation=shap.Explanation(values=values,base_values=base,data=result['plotting_data'],feature_names=labels)
    shap.plots.bar(explanation,max_display=20,show=False)
    plt.gcf().set_size_inches(14,10)
    plt.tight_layout(); plt.savefig(output/'figures/shap_global.png',dpi=150,bbox_inches='tight'); plt.close('all')
    shap.summary_plot(values,result['plotting_data'],feature_names=labels,max_display=20,show=False,plot_size=(14,10))
    plt.tight_layout(); plt.savefig(output/'figures/shap_summary.png',dpi=150,bbox_inches='tight'); plt.close('all')
    grouped_local=shap.Explanation(values=result['grouped_values'][0],base_values=base[0],
                                  data=raw.iloc[0].to_numpy(),feature_names=[plot_label(c) for c in FEATURE_COLUMNS])
    shap.plots.waterfall(grouped_local,max_display=15,show=False)
    plt.gcf().set_size_inches(15,10)
    plt.savefig(output/'figures/shap_local.png',dpi=150,bbox_inches='tight'); plt.close('all')


def explain_run(data_path,splits,run,output,samples=128):
    from .train_models import load_saved_data
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    if not 2<=samples<=512: raise ValueError('samples must be between 2 and 512 for bounded CPU memory')
    if not (run/'COMPLETE').exists(): raise ValueError('Optimization run is incomplete')
    X,y,source,indices,digest=load_saved_data(data_path,splits)
    decision=json.loads((run/'selection.json').read_text())
    if decision['dataset_sha256']!=digest: raise ValueError('Selection source checksum mismatch')
    pipeline=joblib.load(run/'models/selected_pipeline.joblib')
    rng=np.random.default_rng(42)
    idx=rng.choice(indices['validation'],size=min(samples,len(indices['validation'])),replace=False)
    raw=X.iloc[idx]
    output.mkdir(parents=True); (output/'figures').mkdir()
    with threadpool_limits(limits=4):
        result=explain_prediction(pipeline,raw)
        permutation=raw_permutation(pipeline,raw,y.iloc[idx])
    values=result['values']; names=result['names']; base=result['base_values']
    np.savez_compressed(output/'shap_values.npz',values=values,base_values=base,
                        grouped_values=result['grouped_values'],source_rows=source[idx])
    pd.DataFrame({'feature':names,'mean_absolute_SHAP':np.abs(values).mean(axis=0)}).sort_values(
        'mean_absolute_SHAP',ascending=False).to_csv(output/'shap_global.csv',index=False)
    grouped=pd.DataFrame({'feature':FEATURE_COLUMNS,'mean_absolute_grouped_SHAP':np.abs(result['grouped_values']).mean(axis=0)}).sort_values(
        'mean_absolute_grouped_SHAP',ascending=False)
    grouped.to_csv(output/'shap_grouped_raw.csv',index=False)
    local=pd.DataFrame({'feature':FEATURE_COLUMNS,'raw_value':raw.iloc[0].to_numpy(),
                        'SHAP_contribution':result['grouped_values'][0]})
    local.to_csv(output/'shap_local_raw.csv',index=False)
    permutation.to_csv(output/'permutation_importance.csv',index=False)
    save_shap_plots(result,raw,output)
    fig,ax=plt.subplots(figsize=(11,8)); top=permutation.head(15).iloc[::-1]
    ax.barh(top.feature,top.RMSE_increase_mean,xerr=top.RMSE_increase_std)
    ax.set_xlabel('Increase in validation sample RMSE after raw-input permutation')
    fig.tight_layout(); fig.savefig(output/'figures/permutation_importance.png',dpi=150); plt.close(fig)
    metadata=dict(selected_candidate=decision['selected_candidate'],family=decision['family'],
                  dataset_sha256=digest,samples=len(idx),partition='validation',permutation_repeats=3,
                  local_source_row=int(source[idx[0]]),local_prediction=float(result['predictions'][0]),
                  local_base_value=float(base[0]),local_contribution_sum=float(values[0].sum()),
                  shap_version=shap.__version__,
                  note='Tree-path-dependent model associations. Raw-source grouping sums transformed descendants, not causal effects. Correlated inputs can share credit; permutation may create implausible configurations.')
    (output/'explanation_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'COMPLETE').write_text('SHAP additivity, plots, local explanation, and raw-pipeline permutation complete.\n')
    print(json.dumps(metadata,indent=2),flush=True)


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=root.parent/'Solar Cell data set2.csv')
    p.add_argument('--splits',type=Path,default=root/'data/processed/phase1')
    p.add_argument('--run',type=Path,default=root/'reports/phase4_run_001')
    p.add_argument('--output',type=Path,default=root/'reports/phase4_explanations_001')
    p.add_argument('--samples',type=int,default=128)
    args=p.parse_args(); explain_run(args.data,args.splits,args.run,args.output,args.samples)

if __name__=='__main__': main()
