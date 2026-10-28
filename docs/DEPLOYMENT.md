# Deployment guide

**Live app:** https://solarcellxai.streamlit.app/

Solarcell XAI is hosted on Streamlit Community Cloud and loads its saved inference bundle from this repository.

## Deployment settings

| Setting | Value |
| --- | --- |
| Repository | `Sreeshank/ML-Project` |
| Branch | `main` |
| Main file path | `deploy/streamlit_app.py` |
| Python version | `3.12` |
| Python dependencies | `deploy/requirements.txt` |
| Linux dependencies | `packages.txt` (`libgomp1`) |
| Theme and browser settings | `.streamlit/config.toml` |

The app itself does not require API keys, cloud credentials, or the original training CSV.

## Deploy another instance

1. Fork this repository or connect a repository containing the project files.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
3. Enter the repository, branch, and main file path shown above.
4. In **Advanced settings**, select **Python 3.12**.
5. Choose an available app URL and click **Deploy**.
6. Once the build completes, open **Predict & explain** and launch a prediction. Confirm that the PCE value, SHAP contributions, and JSON download appear.

## Files required for prediction

```text
app.py
assets/
src/
deploy/streamlit_app.py
deploy/requirements.txt
packages.txt
.streamlit/config.toml
models/app_bundle/pipeline.joblib
models/app_bundle/metadata.json
```

`example_input.json` in the bundle provides a complete input for CLI checks. The bundle is included in Git, while original datasets, processed splits, and other trained models are excluded.

The application checks the pipeline's SHA256 against its metadata before loading it. Keep those two files together when replacing a model. Use trusted model artifacts: the checksum detects mismatched files and does not establish the safety of arbitrary serialized Python objects.

## Local check with deployment dependencies

Use a Python 3.12 virtual environment, then run these commands from the repository root:

```bash
python -m pip install -r deploy/requirements.txt
python -m streamlit run deploy/streamlit_app.py
```

For command-line prediction:

```bash
python -m src.predict predict --input models/app_bundle/example_input.json --explain
```

## Updating the hosted app

Commit application changes to the configured `main` branch. Community Cloud monitors that branch; watch the build logs to confirm an update has completed. Changing dependencies triggers an environment rebuild. Verify a prediction after updates to the model, preprocessing, dependencies, or entrypoint.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Could not load the research model` | Confirm the two bundle files exist in the deployed branch and match each other. Check `PEROVSKITE_BUNDLE` if you configured an alternative directory. |
| NumPy installation fails | Use Python 3.12 with the deployment requirements, which pin the saved model's original NumPy version. |
| LightGBM reports a missing `libgomp` library | Confirm the root `packages.txt` file contains `libgomp1`, then rebuild the app environment. |
| Training packages appear missing during a research command | Install root `requirements.txt`; deployment requirements cover inference. |
| An update is not visible | Check the app's repository, branch, entrypoint, and build logs in Community Cloud. |

Official reference: [Streamlit Community Cloud deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
