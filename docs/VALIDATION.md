# Frozen-model validation and uncertainty

The active LightGBM model is kept frozen. This workflow does not refit it or select new hyperparameters. Its model SHA256 and the original dataset SHA256 bind every installed analytics artifact to the exact bundle.

## Reproduce the evaluation

Install the research dependencies and run from the repository root:

```powershell
python -m pip install -r requirements.txt -c deploy/requirements.txt
python -m src.research_validation --data "D:\Vs code\Ai\Solar Cell data set2.csv" --output reports/phase6_validation_001 --install
```

The output directory must not exist. For a deliberate reproduction, choose a new output directory; do not repeatedly evaluate the held-out partition to select models or interval settings. `--install` activates the completed analytics in `models/app_bundle/analytics.json`; omit it to produce reports only. Streamlit invalidates its cached bundle when analytics changes.

1. Verify the original CSV checksum, saved reproducible split, bundle checksum, and grouping policy.
2. Divide the original held-out test partition into calibration and evaluation subsets by configuration group, using seed 43 and a 50% group split. Preserve train and validation partitions.
3. Write `protocol.json` with the frozen model identity and exact source-row assignments **before predicting**.
4. On calibration data, compute the maximum absolute residual for each configuration group. For `n` groups and miscoverage `alpha`, use sorted score rank `ceil((n + 1) * (1 - alpha))`. Refuse a finite interval if this rank exceeds `n`.
5. Measure point-prediction MAE/RMSE/R2 and interval coverage on the separate evaluation subset. Group coverage requires every observation in that evaluation group to lie inside its interval. Row coverage weights each observation equally.
6. Audit missingness, measurements, and training support separately for train, validation, calibration, and evaluation. Select complete recorded presets from training rows only.
7. Save diagnostics, assignments, `analytics.json`, and `COMPLETE`; optionally install analytics after all calculations succeed.

The default `alpha=0.1` was specified before evaluation. The measured 90.53% group coverage is an empirical result, not proof of future performance. The group-max construction is conservative when multiple outcomes share one configuration. Intervals have constant width and can extend below 0% or above 100%; they are not clipped to hide uncertainty.

The prior Phase 3 model already evaluated the original test partition. This is a post-selection internal evaluation of the frozen optimized model, not a fresh external test. Similar configurations may share publications or laboratories across the original split; those IDs are absent from the available CSV. No external-data performance claim is made.

Method references: [split conformal prediction](https://arxiv.org/abs/2107.07511), [group-preserving partitioning](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).

## Stronger publication/laboratory splits for future training

If a future dataset includes a separate publication or laboratory identifier, create a **new** split directory:

```powershell
python -m src.data_preprocessing --data "path\source_with_ids.csv" --group-column publication_id --output data/processed/publication_split_001
```

Rows sharing an ID remain together. Identical configurations also remain together: IDs connected by an identical normalized device are merged into one component. This can produce large groups; at least seven distinct components are required. Missing IDs are rejected. Hashing configurations does not use outcomes.

Use this new split directory for a new feature-engineering, training, optimization, and bundle run as described in [RESEARCH.md](RESEARCH.md). The existing model was trained on the old split; changing its test assignment afterward cannot repair publication leakage. The validation command refuses a bundle with a different grouping policy or split sizes.

## Reading the audit and app checks

- Missingness includes nulls, the reserved missing marker, and semantic Unknown labels. Explicit `None`/`Undoped` chemistry labels indicate absence, not missingness.
- Absent-from-training rates exclude missing values and compare full raw values against the training options.
- Measurement parsing preserves original text. Explicit units and bare numbers are summarized separately, with no conversion or inferred units. Reports distinguish partially parseable rows, unparsed values, and unspecified units.
- Outside-token-vocabulary counts refer to the bounded training vocabulary. A rare training token can be outside this vocabulary, so this is not synonymous with unseen chemistry.
- Presets are complete training records, not combinations of column modes or recommendations for high efficiency.
- Device checks flag explicit layer/order, simple formula/ion-label, encapsulation, and parser inconsistencies. Complex chemistry is left for review; these checks do not establish physical feasibility or causation.
- SHAP summaries report signed predictive associations. Changing an input is not an experimentally established intervention.

The app remains usable with a trusted model-only bundle, but then labels intervals and full quality statistics as unavailable. An analytics file with a mismatched model or dataset checksum is rejected.
