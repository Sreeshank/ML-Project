"""Metrics, diagnostics, and model-specific importance for fitted pipelines."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError('Predictions must be finite and match target shape')
    return {'MAE':float(mean_absolute_error(actual,predicted)),
            'RMSE':float(np.sqrt(mean_squared_error(actual,predicted))),
            'R2':float(r2_score(actual,predicted))}


def save_diagnostics(actual, predicted, raw_features, source_rows, output):
    """Post-selection test diagnostics; never use these to choose hyperparameters."""
    output = Path(output)
    figures = output/'figures'; figures.mkdir(parents=True,exist_ok=True)
    table = pd.DataFrame({'source_row':np.asarray(source_rows),'actual':np.asarray(actual),
                          'predicted':predicted})
    table['residual'] = table.actual-table.predicted
    table['absolute_error'] = table.residual.abs()
    table.to_csv(output/'test_predictions.csv',index=False)
    table.nlargest(30,'absolute_error').to_csv(output/'largest_test_errors.csv',index=False)
    for column in ['Cell_architecture','Perovskite_composition_short_form']:
        grouped = table.assign(group=raw_features[column].to_numpy()).groupby('group').agg(
            count=('absolute_error','size'), MAE=('absolute_error','mean'),
            mean_residual=('residual','mean'))
        grouped.sort_values('count',ascending=False).to_csv(output/f'error_by_{column}.csv')
    bins = pd.cut(table.actual,[-np.inf,5,10,15,20,np.inf])
    table.groupby(bins,observed=True).agg(count=('absolute_error','size'),
        MAE=('absolute_error','mean'),mean_residual=('residual','mean')).to_csv(output/'error_by_observed_pce.csv')
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(table.actual,table.predicted,s=5,alpha=.25)
    limits=[min(table.actual.min(),table.predicted.min()),max(table.actual.max(),table.predicted.max())]
    ax.plot(limits,limits,'k--'); ax.set(xlabel='Observed PCE (%)',ylabel='Predicted PCE (%)')
    fig.tight_layout(); fig.savefig(figures/'test_parity.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots()
    ax.hist(table.residual,bins=50); ax.set(xlabel='Observed minus predicted PCE (percentage points)',ylabel='Records')
    fig.tight_layout(); fig.savefig(figures/'test_residuals.png',dpi=150); plt.close(fig)


def save_model_importance(pipeline, output):
    model = pipeline.named_steps['model']
    if 'preprocessor' in pipeline.named_steps:
        names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    else:
        names = pipeline.named_steps['features'].get_feature_names_out()
    if hasattr(model,'feature_importances_'):
        values = model.feature_importances_
        kind = 'Native model importance (model-specific scale)'
    elif hasattr(model,'coef_'):
        values = np.abs(model.coef_)
        kind = 'Absolute coefficient (transformed feature scale)'
    else:
        raise ValueError('Selected model has no supported importance attribute')
    importance=pd.DataFrame({'feature':names,'importance':values}).sort_values('importance',ascending=False)
    output=Path(output); importance.to_csv(output/'feature_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,8)); top=importance.head(20).iloc[::-1]
    ax.barh(top.feature,top.importance); ax.set_xlabel(kind)
    fig.tight_layout(); fig.savefig(output/'figures/feature_importance.png',dpi=150); plt.close(fig)
