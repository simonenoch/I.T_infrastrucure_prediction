from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "models" / "smd_isolation_forest.joblib"

st.set_page_config(page_title="Server Health Anomaly Monitor", page_icon="🖥️", layout="wide")
st.title("IT Infrastructure Anomaly Detection")
st.caption("Student prototype — server telemetry anomaly monitoring")
st.warning("An anomaly is an unusual metric pattern, not proof of a failure. Investigate alerts before taking action.")

if not ARTIFACT_PATH.exists():
    st.error("Model artefact not found. First run all cells in notebooks/IT_Infrastructure_Anomaly_Detection.ipynb to download data, train the model, and save the artefact.")
    st.stop()

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
features = artifact["feature_names"]
medians = pd.Series(artifact["training_medians"]).reindex(features)
threshold = float(artifact["threshold"])

st.subheader("Analyse a telemetry CSV")
st.write("Upload a CSV containing the same 38 anonymised metric columns used in the benchmark. Column names must be metric_01 through metric_38. Extra columns are ignored; missing metric columns are filled with training medians and flagged below.")
upload = st.file_uploader("Upload server metrics CSV", type=["csv"])
if upload is not None:
    try:
        data = pd.read_csv(upload)
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.warning(f"{len(missing)} expected metric columns are missing; they will be filled with training medians. Example: {missing[:5]}")
        X = data.reindex(columns=features)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(medians)
        scores = model.decision_function(X)
        alerts = (scores < threshold).astype(int)
        result = data.copy()
        result["anomaly_score"] = scores
        result["alert"] = alerts
        a, b, c = st.columns(3)
        a.metric("Rows analysed", f"{len(result):,}")
        b.metric("Alerts", f"{int(alerts.sum()):,}")
        c.metric("Alert rate", f"{alerts.mean()*100:.2f}%")
        st.subheader("Results")
        st.dataframe(result.head(500), use_container_width=True)
        st.download_button("Download scored results CSV", result.to_csv(index=False).encode("utf-8"), "server_anomaly_results.csv", "text/csv")
        st.subheader("Anomaly score over rows")
        st.line_chart(pd.DataFrame({"anomaly_score": scores, "threshold": np.repeat(threshold, len(scores))}))
        st.info("Lower Isolation Forest decision scores are more anomalous. Rows below the training-derived threshold are marked as alerts.")
    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")
else:
    st.info("No file uploaded yet. Use a CSV with the 38 metric columns to demonstrate inference.")

st.divider()
st.subheader("About this prototype")
st.markdown("""- **Dataset:** Server Machine Dataset (SMD), machine-1-1.
- **Algorithm:** Isolation Forest trained on the training stream.
- **Threshold:** Derived from training anomaly scores, not selected using test labels.
- **Limitation:** The metric names are anonymised; the app is a demonstration, not a production monitoring system.""")
