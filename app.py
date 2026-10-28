"""Run locally with: python -m streamlit run app.py"""
from html import escape
from pathlib import Path
import json
import os

import pandas as pd
import streamlit as st

from src.data_preprocessing import FEATURE_COLUMNS, MISSING_CATEGORY
from src.predict import load_bundle, predict_records, BOOL_COLUMNS
from src.ui_components import crystal_viewer

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Solarcell XAI | Research Lab", page_icon="◈", layout="wide")


@st.cache_resource
def resources(path, model_mtime, metadata_mtime):
    return load_bundle(path)


def label(column):
    return column.replace("Perovskite_", "Perovskite ").replace("_", " ")


def section(number, title, note=""):
    st.markdown(
        f'<div class="lab-section"><span class="section-number">{escape(number)}</span>'
        f'<h2>{escape(title)}</h2><span class="section-note">{escape(note)}</span></div>',
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown('''<div class="lab-masthead">
<div class="lab-brand"><div class="lab-mark" aria-hidden="true">◈</div><div><div class="lab-wordmark">SOLARCELL <span>XAI</span></div><small>SOLAR MATERIALS RESEARCH LAB</small></div></div>
<div class="lab-status"><i aria-hidden="true"></i> RESEARCH CONSOLE</div>
</div>''', unsafe_allow_html=True)
    crystal_viewer(key="crystal_explorer", data={"paused": st.session_state.get("pause_animations", False)})
    st.markdown('<div class="lab-notice">Research use · Predictions are model estimates, not experimentally verified efficiencies.</div>', unsafe_allow_html=True)


def render_overview(pipeline, metadata):
    audit = metadata["dataset_audit"]
    decision = metadata["selection"]
    section("01", "The research at a glance", "DATASET TELEMETRY")
    cols = st.columns(4)
    engineered = len(pipeline.named_steps["features"].get_feature_names_out()) - len(FEATURE_COLUMNS)
    for col, title, value in zip(
        cols,
        ["Raw records", "Labeled records", "Raw predictors", "Engineered features"],
        [audit["raw_rows"], audit["cleaned_rows"], len(FEATURE_COLUMNS), engineered],
    ):
        col.metric(title, f"{value:,}")

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown('''<div class="lab-card">
<div class="card-index">THE APPROACH</div><h3>From composition to contribution.</h3>
<p>A CPU-trained model connects material composition, device layers, and fabrication records with predicted power conversion efficiency. Engineered inputs capture layer counts, chemical tokens, reported measurements, and missingness.</p>
<div class="lab-pipeline"><span>25 raw inputs</span><b>→</b><span>Feature engineering</span><b>→</b><span>PCE + SHAP</span></div>
</div>''', unsafe_allow_html=True)
    with right:
        metrics = decision["validation_metrics"]
        st.markdown(f'''<div class="lab-card">
<div class="card-index">ACTIVE MODEL / VALIDATION SELECTED</div>
<h3 class="lab-model-name">{escape(decision['selected_candidate'])}</h3>
<p>Selected by validation RMSE. The active optimized model has no newly computed test score.</p>
<div class="lab-model-meta"><div><small>VALIDATION RMSE · PCE POINTS</small><strong>{metrics['RMSE']:.3f}</strong></div><div><small>VALIDATION R²</small><strong>{metrics['R2']:.3f}</strong></div></div>
</div>''', unsafe_allow_html=True)

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.subheader("Dataset partitions")
        partition = pd.DataFrame({"Partition": list(audit["split_rows"]), "Records": list(audit["split_rows"].values())})
        st.dataframe(partition, hide_index=True, width="stretch",
                     column_config={"Records": st.column_config.ProgressColumn(
                         "Records", format="%d", min_value=0, max_value=audit["cleaned_rows"])})
        st.caption("Identical input configurations stay in one partition. Publication/laboratory identifiers are unavailable; similar configurations may cross partitions.")
    with right:
        st.subheader("Interpretation limits")
        st.write("SHAP describes how this model uses inputs, not causal material effects. Bare measurement numbers retain unspecified units.")
        st.caption("Dropdown values occur in training data. Independently combined choices may form an untested or physically inconsistent device. Earlier test metrics belong to the Phase 3 pipeline.")


def render_comparison(metadata):
    section("02", "A closer look at performance", "VALIDATION BENCHMARKS")
    decision = metadata["selection"]
    cols = st.columns(3)
    for col, name in zip(cols, ["RMSE", "MAE", "R2"]):
        title = "Validation R²" if name == "R2" else f"Validation {name}"
        col.metric(title, f"{decision['validation_metrics'][name]:.4f}")
    st.caption("MAE and RMSE are in PCE percentage points. These are validation metrics for the active model.")
    st.subheader("Five-algorithm validation comparison")
    comparison = pd.DataFrame(metadata["phase3_validation_comparison"]).sort_values("RMSE")
    plot, table = st.columns([1, 1.4], gap="large")
    with plot:
        st.bar_chart(comparison.set_index("model")[["RMSE"]], horizontal=True,
                     color="#65e9da", height=275, x_label="RMSE · PCE percentage points", y_label="Model")
        st.caption("Lower RMSE indicates a smaller validation error. The mean baseline is shown alongside the five algorithms.")
    with table:
        st.dataframe(comparison, hide_index=True, width="stretch")
    st.subheader("Budgeted optimization · validation only")
    st.dataframe(pd.DataFrame(metadata["phase4_validation_comparison"]).sort_values("RMSE"),
                 hide_index=True, width="stretch")
    st.info(f"Active model: {decision['selected_candidate']} · Validation RMSE: {decision['validation_metrics']['RMSE']:.4f} PCE percentage points")
    with st.expander("Earlier Phase 3 test results"):
        st.caption("These scores belong to the earlier, untuned pipeline. They do not evaluate the active optimized model.")
        st.dataframe(pd.DataFrame(metadata["phase3_test_metrics"]).T, width="stretch")
    st.caption("Repeated validation-based selection can make validation estimates optimistic. No independent external benchmark has been established.")


def render_reactor(complete):
    state = "complete" if complete else "standby"
    title = "PREDICTION COMPLETE" if complete else "READY FOR YOUR EXPERIMENT"
    st.markdown(f'''<div class="reactor {state}">
<div class="reactor-top"><span>◉ INFERENCE CORE</span><span class="reactor-status">{title}</span></div>
<div class="reactor-scene" aria-hidden="true"><div class="reactor-grid"></div><div class="reactor-halo"></div><div class="reactor-ring outer"></div><div class="reactor-ring middle"></div><div class="reactor-ring inner"></div><div class="reactor-orbit one"><i></i></div><div class="reactor-orbit two"><i></i></div><div class="reactor-center">◈<span>PCE</span></div></div>
<div class="reactor-caption">{'ESTIMATE + EXPLANATION GENERATED' if complete else 'CONFIGURE · PREDICT · DISCOVER'}</div>
</div>''', unsafe_allow_html=True)


def render_contributions(contributions):
    """Show signed bars with a shared, symmetric scale and exact text values."""
    top = contributions.loc[contributions.abs().sort_values(ascending=False).head(12).index]
    scale = float(top.abs().max()) if not top.empty else 0.0
    rows = []
    for i, (field, value) in enumerate(top.items()):
        direction = "positive" if value >= 0 else "negative"
        width = 50 * abs(float(value)) / scale if scale else 0
        name = label(field)
        short_name = name.removeprefix("Perovskite ")
        rows.append(
            f'<div class="shap-row" style="--delay:{i * 55}ms">'
            f'<div class="shap-name" title="{escape(name, quote=True)}">{escape(short_name)}</div>'
            f'<div class="shap-track" aria-hidden="true"><div class="shap-bar {direction}" style="width:{width:.4f}%"></div></div>'
            f'<div class="shap-value {direction}">{value:+.3f}</div></div>'
        )
    st.markdown(
        '<div class="shap-panel"><div class="shap-legend"><span>TOP 12 CONTRIBUTIONS · PCE POINTS</span>'
        '<div><b class="negative">− Lowers estimate</b><b class="positive">+ Raises estimate</b></div></div>'
        + ''.join(rows)
        + '<div class="shap-axis">Bars share a symmetric scale around zero. Ranked by absolute contribution.</div></div>',
        unsafe_allow_html=True,
    )


def render_prediction(pipeline, metadata):
    heading, motion_control = st.columns([3.5, 1])
    with heading:
        section("03", "The prediction studio", "MATERIALS → INTELLIGENCE")
    with motion_control:
        st.toggle("Pause animations", key="pause_animations", help="Pause decorative motion throughout the app. You can still rotate the crystal manually.")
    st.markdown('''<div class="studio-intro"><div><span class="studio-kicker">YOUR VIRTUAL SOLAR LAB</span>
<h3>Design a cell. Discover its potential.</h3><p>Move from material choices to a model estimate, then explore the contributions behind it.</p></div><div class="studio-emblem" aria-hidden="true">✧</div></div>
<div class="studio-steps"><div><span>01</span><section><b>Configure</b><small>Materials & fabrication</small></section></div><i></i><div><span>02</span><section><b>Predict</b><small>Estimated solar efficiency</small></section></div><i></i><div><span>03</span><section><b>Understand</b><small>Signed SHAP contributions</small></section></div></div>''', unsafe_allow_html=True)
    options, defaults = metadata["training_options"], metadata["default_input"]
    groups = {
        "Device and layer structure": [c for c in FEATURE_COLUMNS if c.startswith(("Cell_", "Substrate_", "ETL_", "HTL_", "Backcontact_"))],
        "Perovskite composition & additives": [c for c in FEATURE_COLUMNS if c.startswith(("Perovskite_composition_", "Perovskite_additives_"))],
        "Perovskite fabrication": [c for c in FEATURE_COLUMNS if c.startswith("Perovskite_") and not c.startswith(("Perovskite_composition_", "Perovskite_additives_"))],
        "Encapsulation": [c for c in FEATURE_COLUMNS if c.startswith("Encapsulation")],
    }
    group_hints = [
        "Build the device architecture, transport layers, and contacts.",
        "Choose the recorded ions, composition, and additives.",
        "Set deposition, solvents, thickness, and annealing conditions.",
        "Specify whether and how the device was encapsulated.",
    ]
    values = {}
    inputs, output = st.columns([1.75, 1], gap="large")
    with inputs:
        with st.container(key="studio_inputs"):
            with st.form("solar_cell_input"):
                st.markdown('<div class="panel-heading"><span>INPUT CONFIGURATION</span><b>25 parameters / 4 modules</b></div>', unsafe_allow_html=True)
                st.caption("Starts from one real training configuration. Recorded values are preserved; bare thickness, temperature, and time values have unspecified units.")
                for group_index, (title, fields) in enumerate(groups.items()):
                    with st.expander(f"0{group_index + 1} / {title}", expanded=group_index == 0):
                        st.caption(group_hints[group_index])
                        columns = st.columns(2)
                        for i, field in enumerate(fields):
                            choices = options[field]

                            def fmt(value, field=field):
                                if value == MISSING_CATEGORY:
                                    return "Missing / not recorded"
                                if field in BOOL_COLUMNS:
                                    return "Yes" if value == 1 else "No"
                                return str(value)

                            values[field] = columns[i % 2].selectbox(
                                label(field), choices, index=choices.index(defaults[field]),
                                format_func=fmt, key=field,
                            )
                st.markdown('<div class="launch-note"><span>◈</span> Every prediction includes an explanation.</div>', unsafe_allow_html=True)
                submitted = st.form_submit_button("Launch prediction →", type="primary", width="stretch")
            if submitted:
                try:
                    with st.spinner("Computing efficiency and SHAP contributions…"):
                        result = predict_records(pipeline, metadata, values, explain=True)
                    st.session_state["prediction_result"] = {"input": dict(values), "result": result}
                except (ValueError, TypeError, KeyError, AssertionError) as exc:
                    st.session_state.pop("prediction_result", None)
                    st.error(f"Prediction could not be completed: {exc}")
    with output:
        with st.container(key="studio_output"):
            complete = "prediction_result" in st.session_state
            render_reactor(complete)
            if complete:
                payload = st.session_state["prediction_result"]
                result = payload["result"]
                st.metric("Estimated PCE", f"{result['predicted_PCE_percent'][0]:.2f}%")
                st.caption("Last submitted configuration · Submit again after editing inputs. No calibrated uncertainty interval is available.")
                st.download_button("↓ Export prediction + SHAP", json.dumps(payload, indent=2),
                                   file_name="perovskite_prediction.json", mime="application/json", width="stretch")
            else:
                st.markdown('''<div class="standby-copy"><h3>What will your cell reveal?</h3><p>Explore the four modules, then launch your prediction. Your estimated PCE will appear here.</p><div class="standby-value">—<span> ESTIMATED PCE</span></div></div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="engine-info"><span>MODEL IN USE</span><b>{escape(metadata['selection']['selected_candidate'])}</b><small>Validation selected · CPU inference</small></div>''', unsafe_allow_html=True)
        st.markdown('''<div class="studio-tip"><span>↗</span><p><b>Read the why, too.</b> PCE estimates light-to-electricity conversion. SHAP shows which inputs move the estimate above or below the model's base value.</p></div>''', unsafe_allow_html=True)

    if "prediction_result" in st.session_state:
        payload = st.session_state["prediction_result"]
        result = payload["result"]
        for warning in result["warnings"]:
            st.warning(warning)
        contributions = pd.Series(result["SHAP_grouped_contributions"][0], name="Contribution (PCE percentage points)")
        base = result["SHAP_base_PCE_percent"][0]
        positive = float(contributions[contributions > 0].sum())
        negative = float(contributions[contributions < 0].sum())
        section("04", "Inside the prediction", "EXPLAINABLE BY DESIGN")
        st.markdown(f'''<div class="explanation-equation">
<div><small>MODEL BASE</small><strong>{base:.3f}<em>%</em></strong></div><span>+</span>
<div class="positive"><small>POSITIVE CONTRIBUTIONS</small><strong>{positive:.3f}<em> pts</em></strong></div><span>−</span>
<div class="negative"><small>NEGATIVE CONTRIBUTIONS</small><strong>{abs(negative):.3f}<em> pts</em></strong></div><span>=</span>
<div class="equation-result"><small>PREDICTED PCE</small><strong>{result['predicted_PCE_percent'][0]:.3f}<em>%</em></strong></div>
</div>''', unsafe_allow_html=True)
        st.subheader("SHAP explanation for this prediction")
        render_contributions(contributions)
        st.caption("The equation includes all contributions; the bars show the 12 largest by magnitude. Values are rounded for display. Contributions combine each raw field and its engineered/encoded descendants. They describe predictive associations, not causal effects.")
        with st.expander("Full contributions and submitted input"):
            st.dataframe(contributions.to_frame(), width="stretch")
            st.json(payload["input"])


def main():
    st.markdown(f"<style>{(ROOT / 'assets/app.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    if st.session_state.get("pause_animations", False):
        st.markdown("<style>.stApp *, .stApp *::before, .stApp *::after { animation: none !important; transition: none !important; }</style>", unsafe_allow_html=True)
    render_header()
    bundle = Path(os.environ.get("PEROVSKITE_BUNDLE", str(ROOT / "models/app_bundle")))
    try:
        pipeline, metadata = resources(
            str(bundle), (bundle / "pipeline.joblib").stat().st_mtime_ns,
            (bundle / "metadata.json").stat().st_mtime_ns,
        )
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not load the research model: {exc}")
        st.code("python -m src.predict prepare", language="bash")
        st.stop()

    overview, comparison, prediction = st.tabs(["Project overview", "Model comparison", "Predict & explain"])
    with overview:
        render_overview(pipeline, metadata)
    with comparison:
        render_comparison(metadata)
    with prediction:
        render_prediction(pipeline, metadata)
    st.markdown('''<div class="lab-footer"><span>SOLARCELL XAI / MATERIALS TO POSSIBILITIES</span><span>RESEARCH ESTIMATES · PREDICTIVE ASSOCIATIONS</span></div>''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
