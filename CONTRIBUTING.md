# Contributing to Solarcell XAI

For bugs, open an [issue](https://github.com/Sreeshank/ML-Project/issues/new/choose). For changes, create a branch and open a pull request describing the problem, resulting behavior, and validation performed.

## Set up the research environment

Use Python 3.12 and run these commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the research dependencies with deployment pins applied:

```bash
python -m pip install -r requirements.txt -c deploy/requirements.txt
python -m streamlit run app.py
```

For app-only development, install `deploy/requirements.txt`. Training and research tests use the broader root requirements.

## Validate changes

```bash
python -m pytest -q
node --test tests/crystal_interactions.test.mjs
```

The JavaScript tests use Node's built-in test runner; Node 22 is used in CI. Node is not required to run the Streamlit app.

AppTest exercises both local and cloud entrypoints, submitted input preservation, predictions, and SHAP additivity. The included model bundle allows these integration checks to run without the original CSV. Data and model unit tests use synthetic fixtures; a full training reproduction requires the original source dataset.

For interface changes, also inspect the overview, model comparison, and prediction tabs at desktop and narrow window sizes. Exercise crystal dragging, keyboard controls, atom inspection, and motion settings when changing the viewer.

## Research changes

The detailed [research notes](docs/RESEARCH.md) document preprocessing, feature engineering, model selection, and saved experiment results. Describe any changes to those assumptions in the pull request. Keep validation metrics and earlier test metrics attributed to their own pipelines, and keep labels and electrical outcomes outside the feature inputs.

Use fresh output directories when running experiments. The pipeline intentionally refuses to overwrite completed experiment outputs. Only the selected app inference bundle belongs in `models/app_bundle/`; other generated models and datasets remain excluded by `.gitignore`.

## Deployment changes

See [deployment settings and troubleshooting](docs/DEPLOYMENT.md). App deployment uses `deploy/streamlit_app.py`, Python 3.12, and the pinned inference requirements. If replacing the bundled model, update its matching metadata and verify prediction and SHAP output before opening the pull request.
