# Solarcell XAI: research methodology and recorded experiments

For the project overview and current quick start, see the [main README](../README.md). The live application is available at [solarcellxai.streamlit.app](https://solarcellxai.streamlit.app/). This document preserves the detailed phase methodology and historical experiment observations; current hosting instructions are in the [deployment guide](DEPLOYMENT.md).

Explainable AI for perovskite solar-cell power conversion efficiency (PCE) prediction, with an interactive crystal explorer and a Streamlit research interface.

All five phases are implemented. The CPU research pipeline, optimization, explanations, prediction wrapper, and Streamlit app are available. Existing workspace notebooks and experiments are preserved separately.


## Launch the completed app

The selected inference bundle is included at `models/app_bundle/` so a fresh clone can make predictions. Other trained models, the original dataset, and processed datasets remain excluded by `.gitignore`. Only load model artifacts from a trusted source. The setup and training sections below describe how to regenerate the pipeline from the source dataset.

From the project directory:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501
```

Open the local URL printed by Streamlit. For SageMaker Studio JupyterLab, run the command in its terminal and use the workspace's supported access/port-forwarding mechanism for port 8501. The app runs on CPU; no GPU or AWS API calls are required.

The interface uses Streamlit 1.66.1 or newer. If upgrading an existing environment, run `python -m pip install -r requirements.txt` before launching.

The opening crystal explorer supports dragging, atom-site inspection, zoom, bond visibility, and reset. Focus the crystal and use arrow keys to rotate, `+`/`-` to zoom, or `Home` to reset. It illustrates an idealized cubic ABX3 cell; it does not simulate the selected device. The prediction studio includes animated result and SHAP displays, a global **Pause animations** switch, and support for the operating system's reduced-motion preference. All visual assets are local and require no third-party CDN.

UI checks: `python -m pytest tests/test_app.py -q` and `node --test tests/crystal_interactions.test.mjs` (the latter uses Node's built-in test runner; Node is not required to run the app).

## Setup

### Deploy on Streamlit Community Cloud

Go to https://share.streamlit.io/ and create an app with:

- Repository: `Sreeshank/ML-Project`
- Branch: `main`
- Main file path: `deploy/streamlit_app.py`
- Advanced settings → Python version: **3.12**

Click **Deploy**. The entrypoint uses `deploy/requirements.txt` for the inference dependencies and the root `packages.txt` for LightGBM's OpenMP library. The saved model is loaded from the repository; the original CSV and a training run are not required. No API keys or secrets are needed for the app itself. Dependency installation and the final public URL are managed by Community Cloud.

### Local research environment

Use Python 3.11 or a compatible SageMaker kernel. Phase 1 was tested on Python 3.12.15, pandas 2.3.3, NumPy 1.26.4, and scikit-learn 1.7.2. No GPU is needed; this stage is suitable for ml.m5.xlarge.

```bash
cd /home/sagemaker-user/perovskite-xai
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.data_preprocessing
```

The default CSV is `../Solar Cell data set2.csv`. To choose another location or create a fresh experiment:

```bash
python -m src.data_preprocessing --data '/path/Solar Cell data set2.csv' --output data/processed/phase1_new
```

Existing output directories are never overwritten. `requirements-phase1-lock.txt` records the exact direct package versions used in this environment. `notebooks/01_data_preparation.ipynb` imports the modules in SageMaker JupyterLab; update its dataset path if needed.

## Phase 1 behavior

All 30 expected columns are validated. Unexpected columns are reported and excluded from predictors. Exact original records are deduplicated before dropping missing labels. Invalid nonmissing or infinite targets cause an error. No efficiency cutoff is invented: zero values and high values remain for later source review.

The 25 allowed inputs exclude PCE, Voc, Jsc, FF, and JV_measured. Material strings, units, delimiters, and literal Unknown entries are preserved. Null categorical inputs receive a reserved token with collision checking. Boolean inputs become numeric 0/1. Numeric missing values remain available for native CatBoost handling or training-only median imputation.

Identical normalized raw input configurations are hashed and grouped before splitting, using seed 42. Groups, rather than rows, are assigned approximately 70/15/15%; row ratios can vary. The group hash does not include labels or provenance. Train/validation/test coverage and group disjointness are asserted. Learned preprocessing is returned unfitted: fit it only on training rows, then transform validation and test rows. One-hot encoding ignores unseen categories and stays sparse; numeric imputation and scaling are learned from training data only.

`data/processed/phase1/` contains `split_indices.npz` (positions in the cleaned dataset), `split_manifest.csv` (clean positions, original zero-based source rows, groups, assignments), and `audit.json` (source SHA256, versions, counts, missingness). The audit is also copied to `reports/phase1_audit.json`. Regenerate inputs using the same source checksum, recorded versions, and preprocessing before applying saved indices; never use them with another CSV.

## Observed Phase 1 results

- Raw: 42,443 rows, 30 columns.
- Removed: 1,364 duplicate records and 597 remaining missing-target records.
- Clean: 40,482 rows, 25 predictors, 22 categorical predictors.
- Saved split: 28,215 train / 6,152 validation / 6,115 test records.

The row split differs from the earlier reference experiment. Boolean representation, categorical missing tokens, input order, and library versions can alter hash ordering and group assignment. These are a new saved split, not a reproduction of the reference metrics. No models were trained and no prediction metrics were computed in this phase. Prior workspace experiments already used this dataset; a new split does not create an independent external benchmark.

## Next stages

Phase 2 implements conservative feature engineering and tests (see below). Phase 3 compares five algorithms on the saved split and evaluates the selected pipeline after validation-based selection. Phase 4 adds budgeted optimization and explanations of the actual selected pipeline. Phase 5 adds Streamlit and final usage documentation. Later stages must not select features or hyperparameters using test results. Explanations will describe predictive associations, not causal material effects.

## Phase 2: conservative feature engineering

```bash
python -m pytest -q
python -m src.feature_engineering
```

The second command verifies the dataset checksum against Phase 1, fits token vocabularies exclusively on saved training rows, transforms train/validation inputs, and saves `data/processed/phase2/feature_engineer.joblib` and `feature_audit.json`. It refuses to overwrite existing artifacts. Use `--output data/processed/phase2_new` for a fresh audit. No model is trained or scored, and test inputs are not transformed.

`PerovskiteFeatureEngineer` retains all 25 raw predictors and adds:

- Device, ETL, and HTL pipe-separated layer counts. Unknown layers yield an unknown count; explicitly absent layers count as zero.
- A/B/C-site, additive, and solvent unique-token counts and indicators. Tokens are separated on semicolons, pipes, or `>>`. Parenthesized chemical labels are preserved. Vocabularies use at most 32 tokens per field, with training-document frequency at least two and deterministic tie ordering. Unseen-token counts handle inference values outside that vocabulary. Counts with unknown components are marked incomplete; known counts are also retained. No chemical stoichiometry is inferred from short-form formulas. Raw short-form composition, additive concentrations, architecture and fabrication information remain available as categorical predictors.
- Minimum, maximum, mean and count of reported annealing temperature/time and perovskite/backcontact thickness values, separated by explicit unit. Supported units: temperature C/K, time s/min/h, thickness nm/um/mm, plus bare unit-unspecified numbers for each field. No unit conversions, inherited units, or physical units for bare numbers are assumed. These are summaries of reported tokens, not total annealing durations or total film thicknesses.
- Missing indicators for every original input, incomplete token flags, and unknown/unparsed measurement counts. Known absences (`none`, `Undoped`) remain distinct from missing information.

Ranges, approximate values, unsupported units, nonfinite numbers, and negative measurement values are excluded from numeric summaries and flagged as unparsed; the original values remain available. Negative temperatures could be valid scientifically, but require source review under this deliberately conservative parser. No full-dataset statistics or labels are used. Numeric missing features are imputed later using only training data.

`notebooks/02_feature_engineering.ipynb` demonstrates loading the saved transformer and transforming only validation inputs. In Phase 3, use the same fitted engineer in the model's saved inference pipeline, select categorical columns from its output, and fit `build_preprocessor` exclusively on engineered training rows for OHE-based models. Original saved configuration groups must remain unchanged. Only load trusted joblib artifacts.

## Phase 3: CPU model comparison

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train_models --threads 4 --output reports/phase3_run_002
```

Each output directory must be new. The first completed run is saved under `reports/phase3_run_001/`. No GPU, cloud API, or external service is used. `requirements-phase3-lock.txt` records the direct dependency versions used for this run. scikit-learn is constrained to 1.7.x because the installed CatBoost 1.2.7 emits a compatibility deprecation warning about estimator tags on newer sklearn; tests verify successful predictions and serialization in the recorded environment. LightGBM may emit a feature-name warning on sparse arrays; column ordering is fixed by the fitted preprocessing pipeline.

The input CSV checksum and saved split indices are verified before training. Feature engineering is fitted on training inputs only. Four models use the same sparse one-hot representation, training-fitted median imputation and numeric scaling; CatBoost receives the same engineered information with native categorical features. The 25 raw fields are retained alongside 252 engineered features.

The budgets are 120 Random Forest trees (depth 16, minimum leaf size 3, 10% feature sampling), 500 boosting rounds for XGBoost/LightGBM/CatBoost, and sparse Linear Regression with tolerance 1e-4. CatBoost uses validation-based early stopping with patience 50. All seeds are 42, models run sequentially, and threads are limited to four. These are modest initial comparison settings rather than exhaustive or equal-runtime optimization. `--quick` reduces tree budgets for smoke checks only; do not describe its results as the research comparison. Optuna is reserved for Phase 4.

The selected algorithm minimizes validation RMSE among the five requested models; a mean baseline is reported separately. All fitted pipelines and parameters are saved. `selection.json` is written before test prediction. No train+validation refit is performed. Only the selected model and mean baseline receive final test metrics. This run has finalized its test evaluation; Phase 4 must tune on train/validation only and avoid using these test diagnostics to guide changes. Prior experiments have already used this dataset, so these results do not establish independent external generalization.

Artifacts include:

- `validation_comparison.csv`, `selection.json`, `test_metrics.json`, `run_config.json`, and a `COMPLETE` marker on success.
- `models/` with five fitted raw-input pipelines plus `selected_pipeline.joblib`. Load only trusted joblib files.
- Test predictions and residuals, largest-error records, and error summaries by architecture, reported composition and observed PCE range. Group results include sample counts; sparse groups require caution.
- `figures/test_parity.png`, `figures/test_residuals.png`, and native `feature_importance.csv` plus visualization. Native importance scales vary by algorithm and are predictive associations, not causal effects. SHAP and permutation explanations follow in Phase 4.

`notebooks/03_model_comparison.ipynb` displays saved results without retraining. For inference, import project modules from the project root, load the pipeline, and pass a DataFrame containing exactly the 25 normalized raw inputs produced by `make_features`; do not supply target or electrical outcome columns. The model pipeline includes engineering and encoding. The Phase 5 prediction module validates complete input records and exposes CLI inference.

### Recorded Phase 3 run

Actual results from `phase3_run_001` (seed 42, saved Phase 1 split):

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| mean_baseline | 4.1763 | 5.0980 | -0.0005 |
| linear_regression | 3.1063 | 4.0920 | 0.3554 |
| random_forest | 3.0061 | 3.9079 | 0.4121 |
| xgboost | 2.9107 | 3.8016 | 0.4437 |
| lightgbm | 2.8578 | 3.7628 | 0.4550 |
| catboost | 2.8546 | 3.7635 | 0.4548 |

LightGBM was selected on validation RMSE. Final test MAE: 2.8324; RMSE: 3.7196 PCE percentage points; R²: 0.4988. LightGBM and CatBoost have almost identical validation RMSE; this single split does not establish a robust ranking. These results are from the new saved split and engineered features, not a reproduction of the earlier reference experiment. All 18 unit tests passed; saved-pipeline predictions were checked after reload. Notebook code and artifact paths were also verified.

## Phase 4: Optuna and explanations

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.optimize_models --trials 3 --threads 4 --output reports/phase4_run_002
python -m src.explainability --run reports/phase4_run_002 --output reports/phase4_explanations_002 --samples 128
```

Use fresh output directories. The initial run uses `phase4_run_001` and `phase4_explanations_001`. The optimization module reuses Phase 3 training-fitted feature engineering and encoding. It runs three seeded Optuna TPE trials per family for the two promising validation models (LightGBM, CatBoost), sequentially with four threads, maximum 700 rounds and validation early-stopping patience 50. A 360-second timeout per family prevents starting more trials after that budget; a trial already in progress can finish beyond the timeout. Three trials are an exploratory budget, not a thorough search or evidence of a global optimum.

The two Phase 3 incumbents remain eligible, so tuning cannot replace them with a model having worse validation RMSE. SQLite stores trial parameters, scores and states; trial CSVs and `validation_comparison.csv` aid review. Each trial's complete inference pipeline is saved. `selection.json` identifies the winner. `models/selected_pipeline.joblib` is the actual winner used for explanations and the later application. No refit on validation rows occurs. Phase 4 generates no test predictions or scores; the earlier Phase 3 test numbers must not be attributed to the optimized model. Validation reuse for tuning creates selection optimism; independent external evaluation remains necessary for a defensible generalization claim.

### Explanation methods and scope

Explanations use 128 seed-42 validation examples, never test examples. LightGBM uses tree-path-dependent Tree SHAP on the exact fitted one-hot/imputation/scaling output. CatBoost uses its native tree SHAP on the exact fitted engineered inputs. Full-pipeline prediction reconstruction is checked numerically for both paths. Tree path statistics derive from training, without a validation-fitted background distribution. SHAP columns align with actual transformed feature names, including imputer indicators. `explain_prediction(pipeline, raw_rows)` is available for later per-input explanations.

Artifacts include global mean absolute SHAP, a summary/beeswarm plot, a local waterfall plot and CSV, contribution arrays and sample source row IDs, and raw-source grouped contributions. Grouped SHAP sums contributions of all engineered/encoded descendants of a raw input; this preserves additivity but does not equal independently recomputing SHAP with raw inputs as players. Raw formulas, missingness and derived indicators can be redundant and share attribution. The local base value plus contributions equals the selected pipeline's PCE prediction.

Permutation importance permutes each of the 25 raw fields and recomputes the entire pipeline, with three repeats on the same 128-row validation sample. Reported values are increases in sample RMSE with repeat standard deviation; these are not confidence intervals. Correlated materials/layers and impossible permuted combinations can distort importance. Negative importance is retained. No inferred causal effects or experimental efficiency claims are made. LIME is optional and is not implemented.

Memory stays bounded by a maximum of 512 explanation samples. For sparse LightGBM inputs, only that explanation batch is converted to dense form. `notebooks/04_optimization_explainability.ipynb` displays saved optimization and explanation results without rerunning either procedure. Phase 4 exact direct package versions are recorded in `requirements-phase4-lock.txt`.

### Recorded Phase 4 run

Six trials completed (three per family). The selected pipeline is `lightgbm_trial_0`, with validation MAE **2.7808**, RMSE **3.6830**, and R² **0.4778**. Its validation RMSE improves on the Phase 3 LightGBM incumbent (3.7628), but it has **no newly computed test metrics**. Phase 3 test results belong to the earlier untuned pipeline.

SHAP and permutation artifacts were generated for 128 validation rows. The local example (CSV data row 19,482, zero-based) predicts PCE 11.1076%; base value 11.9810 plus contributions −0.8734 reconstructs that prediction. This is a model estimate, not experimentally verified efficiency. All 22 tests passed, including SHAP reconstruction for LightGBM and CatBoost, raw-source grouping, full-pipeline permutation, and rejection of modified source or split artifacts. Notebook result paths were verified.

## Phase 5: application and portable inference

The app has a project overview with dataset counts and saved partitions, five-model validation comparison, optimization comparison, and a materials/fabrication prediction form. All 25 dropdowns are populated from training records only. The initial values come from one actual training configuration. It predicts PCE percentage and calculates a SHAP explanation for those submitted inputs using the active optimized pipeline. Grouped SHAP contributions include engineered and encoded descendants and preserve the prediction sum. Results and submitted inputs can be downloaded as JSON. Results remain labeled as the last submission until the user submits again.

The interface clearly states that efficiencies are estimates, not experimental validation. It does not provide an uncalibrated uncertainty interval or imply that independently combined layer/material choices are chemically consistent. Bare recorded measurements retain unspecified units. Original per-field values remain available rather than being converted to invented units. Prior Phase 3 test results are labeled separately from the active optimized model's validation results.

### Create a portable app bundle

```bash
python -m src.predict prepare
```

This command creates `models/app_bundle/` with `pipeline.joblib`, `metadata.json`, and `example_input.json`. Metadata contains source/model hashes, training-derived options, an actual training configuration, dataset audit and correctly attributed comparison results. The command refuses to overwrite an existing bundle; use `--output models/app_bundle_new` for another copy. The dataset, optimization and comparison reports must refer to the same source checksum. Model/pipeline deserialization is only appropriate for trusted local artifacts; checksums detect accidental mismatches, not malicious modifications.

Once prepared, the app and CLI do not need the source CSV or training reports. Transfer the bundle together with this project's `src/`, `app.py` and dependencies. Set `PEROVSKITE_BUNDLE` to another trusted bundle directory if necessary. Keep the recorded compatible package versions when moving artifacts across environments. `requirements-lock.txt` records all final direct dependency versions tested here; Python 3.11 compatibility is targeted but this execution was verified on Python 3.12.15. The environment-specific lock may require adjustment for other Python versions.

### Predict from JSON

```bash
python -m src.predict predict --input models/app_bundle/example_input.json --explain
```

Pass a JSON object for one prediction or a list of up to 512 objects. Each record must contain exactly the 25 raw feature fields, without labels or electrical outcomes. Categorical values must be strings or null; material/fabrication strings are preserved. Boolean columns accept JSON true/false or numeric 0/1, not strings such as `"False"`. Deposition steps must be a nonnegative integer or null. Nulls receive the same normalization used in training. Unseen categories are accepted safely and reported in warnings. Nested values, unknown columns, nonfinite numbers and invalid Boolean values are rejected. Predictions are not silently clipped.

### Reproduce all phases in a fresh checkout

Place `Solar Cell data set2.csv` next to the project directory (or pass `--data` to every data-dependent module). These commands assume generated output directories do not exist:

```bash
python -m pip install -r requirements.txt
python -m src.data_preprocessing
python -m src.feature_engineering
python -m src.train_models --threads 4
python -m src.optimize_models --trials 3 --threads 4
python -m src.explainability --samples 128
python -m src.predict prepare
python -m pytest -q
python -m streamlit run app.py
```

Existing experiments are protected against overwrites. To create additional experiments, use new `--output` directories and pass their paths to downstream `--splits`, `--previous`, `--run`, and `--phase3` arguments as appropriate. Source data, split manifests, and generated models are excluded from Git, with the selected `models/app_bundle/` inference bundle included as an explicit exception. Preserve other experiment artifacts separately alongside the SHA256 audits if reproducibility is required.

### Verification

Unit tests cover data cleaning, leakage exclusion, grouping, raw-value preservation, unit-aware feature parsing, training-only vocabularies, all five model serialization paths, SHAP additivity, raw-source grouping, raw-pipeline permutation, changed source/split rejection, inference validation and model checksum rejection. Streamlit AppTest checks missing-bundle handling, all 25 form inputs, prediction/SHAP output and changed-input resubmission. The app integration test is skipped when no generated bundle is present; generate the bundle to run it.

The optional LIME method was not added. There is no independent external test benchmark, physical-consistency validator, calibrated uncertainty model, or new test score for the optimized model. These limits are documented throughout the app and research reports.

Historical Phase 5 verification: **35 tests passed** with the generated app bundle present. CLI inference and prediction notebook code also executed successfully. Existing LightGBM sparse feature-name and CatBoost sklearn-tag compatibility warnings remain documented; all prediction checks passed under the recorded versions. Subsequent application checks also exercise the cloud entrypoint. Solarcell XAI is now deployed at [solarcellxai.streamlit.app](https://solarcellxai.streamlit.app/).
