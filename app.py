import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="IT Infrastructure Failure Prediction",
    page_icon="🖥️",
    layout="wide",
)

FEATURES = ["cpu_util", "memory_util", "disk_util", "load1", "load5", "load15"]
USAGE_COLUMNS = [
    "timestamp", "machineID", *FEATURES
]
EVENT_COLUMNS = [
    "timestamp", "machineID", "event_type", "event_detail",
    "capacity_cpu", "capacity_memory", "capacity_disk"
]
TARGET = "hardware_failure_next_hour"
HORIZON_SECONDS = 3600
BASE_DIR = Path(__file__).resolve().parent

st.title("🖥️ IT Infrastructure Failure Prediction")
st.caption("A machine_learning system for predicting server hardware failures.")


@st.cache_data(show_spinner=False)
def load_training_data(usage_path: str, event_path: str) -> pd.DataFrame:
    usage = pd.read_csv(usage_path, header=None, names=USAGE_COLUMNS, low_memory=False)
    events = pd.read_csv(event_path, header=None, names=EVENT_COLUMNS, low_memory=False)

    for col in USAGE_COLUMNS:
        usage[col] = pd.to_numeric(usage[col], errors="coerce")
    for col in ["timestamp", "machineID"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")

    events["event_type"] = events["event_type"].astype("string").str.strip().str.lower()
    usage = usage.drop_duplicates().dropna(subset=["timestamp", "machineID"]).copy()
    events = events.drop_duplicates().dropna(subset=["timestamp", "machineID"]).copy()
    usage["timestamp"] = usage["timestamp"].astype("int64")
    usage["machineID"] = usage["machineID"].astype("int64")
    events["timestamp"] = events["timestamp"].astype("int64")
    events["machineID"] = events["machineID"].astype("int64")

    # The supplied trace records the relevant target event as "softerror".
    failure_events = events.loc[
        (events["event_type"] == "softerror") & (events["timestamp"] > 0),
        ["timestamp", "machineID"],
    ].copy()
    if failure_events.empty:
        raise ValueError(
            "No 'softerror' events with positive timestamps were found in server_event.csv."
        )

    event_end_time = int(events["timestamp"].max())
    data = usage.loc[
        (usage["timestamp"] > 0)
        & (usage["timestamp"] + HORIZON_SECONDS <= event_end_time)
    ].copy()

    failure_times = {
        int(machine_id): np.sort(group["timestamp"].to_numpy(dtype=np.int64))
        for machine_id, group in failure_events.groupby("machineID")
    }
    labels = np.zeros(len(data), dtype=np.int8)

    # Label a row 1 when a future event for that same machine occurs within one hour.
    for machine_id, positions in data.groupby("machineID", sort=False).indices.items():
        row_times = data.iloc[positions]["timestamp"].to_numpy(dtype=np.int64)
        event_times = failure_times.get(int(machine_id))
        if event_times is None or len(event_times) == 0:
            continue
        next_event_idx = np.searchsorted(event_times, row_times, side="right")
        has_future_event = next_event_idx < len(event_times)
        positive = np.zeros(len(row_times), dtype=bool)
        positive[has_future_event] = (
            event_times[next_event_idx[has_future_event]]
            <= row_times[has_future_event] + HORIZON_SECONDS
        )
        labels[positions] = positive.astype(np.int8)

    data[TARGET] = labels
    return data.sort_values(["timestamp", "machineID"]).reset_index(drop=True)

@st.cache_resource(show_spinner="Training the Random Forest model on the supplied data…")
def train_model(usage_path: str, event_path: str):
    data = load_training_data(usage_path, event_path)
    if len(data) < 20 or data[TARGET].nunique() < 2:
        raise ValueError("Not enough labeled data or only one target class is present.")

    split_index = int(len(data) * 0.8)
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(train[FEATURES], train[TARGET].astype(int))

    predictions = model.predict(test[FEATURES])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(test[TARGET], predictions)),
        "test_rows": int(len(test)),
        "test_positive_rows": int(test[TARGET].sum()),
        "roc_auc": float(roc_auc_score(test[TARGET], probabilities))
            if test[TARGET].nunique() == 2 else None,
        "average_precision": float(average_precision_score(test[TARGET], probabilities))
            if test[TARGET].nunique() == 2 else None,
        "confusion_matrix": confusion_matrix(test[TARGET], predictions, labels=[0, 1]).tolist(),
        "report": classification_report(
            test[TARGET], predictions,
            labels=[0, 1], target_names=["No recorded event", "Recorded event"],
            output_dict=True, zero_division=0
        ),
        "importance": dict(zip(
            FEATURES,
            model.named_steps["classifier"].feature_importances_.tolist()
        )),
    }
    return model, data, metrics

def make_template() -> bytes:
    example = pd.DataFrame([{
        "cpu_util": 25.0,
        "memory_util": 50.0,
        "disk_util": 45.0,
        "load1": 18.0,
        "load5": 17.5,
        "load15": 16.0,
    }])
    return example.to_csv(index=False).encode("utf-8")

usage_path = BASE_DIR / "server_usage.csv"
event_path = BASE_DIR / "server_event.csv"

if not usage_path.exists() or not event_path.exists():
    st.error(
        "Training data files were not found. Keep server_usage.csv and server_event.csv "
        "in the same GitHub repository folder as app.py."
    )
    st.stop()

try:
    model, labeled_data, metrics = train_model(str(usage_path), str(event_path))
except Exception as exc:
    st.error(f"Could not train the model: {exc}")
    st.info("Check that both supplied CSV files are in the repository and have the expected format.")
    st.stop()

st.success("Model trained successfully from the included dataset.")
c1, c2, c3 = st.columns(3)
c1.metric("Labeled usage rows", f"{len(labeled_data):,}")
c2.metric("Positive examples", f"{int(labeled_data[TARGET].sum()):,}")
c3.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")

with st.expander("Model evaluation (chronological 80/20 split)"):
    st.write(
        "The earliest 80% of time-sorted rows are used for training; the latest 20% are used "
        "for a simple holdout evaluation. This is more appropriate for time-ordered data than "
        "a random split, although it does not remove every possible source of data leakage."
    )
    m1, m2 = st.columns(2)
    m1.metric("Test rows", f"{metrics['test_rows']:,}")
    m2.metric("Positive test rows", f"{metrics['test_positive_rows']:,}")
    m3, m4 = st.columns(2)
    m3.metric("ROC-AUC", "N/A" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}")
    m4.metric(
        "Average precision",
        "N/A" if metrics["average_precision"] is None else f"{metrics['average_precision']:.3f}"
    )
    st.subheader("Classification report")
    report_df = pd.DataFrame(metrics["report"]).T
    st.dataframe(report_df.round(3), use_container_width=True)
    st.subheader("Confusion matrix")
    st.dataframe(
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual: no event", "Actual: event"],
            columns=["Predicted: no event", "Predicted: event"],
        ),
        use_container_width=True,
    )
    st.subheader("Feature importance")
    importance_df = pd.DataFrame(
        {"Feature": list(metrics["importance"].keys()),
         "Importance": list(metrics["importance"].values())}
    ).sort_values("Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

st.divider()
st.header("1. Predict from manually entered values")
st.write("Enter one server's current workload measurements. Values should use the same units and ranges as the training dataset.")

with st.form("manual_prediction"):
    col1, col2, col3 = st.columns(3)
    with col1:
        cpu = st.number_input("CPU utilization (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
        memory = st.number_input("Memory utilization (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.5)
    with col2:
        disk = st.number_input("Disk utilization (%)", min_value=0.0, max_value=100.0, value=45.0, step=0.5)
        load1 = st.number_input("Load average — 1 minute", min_value=0.0, value=18.0, step=0.5)
    with col3:
        load5 = st.number_input("Load average — 5 minutes", min_value=0.0, value=17.5, step=0.5)
        load15 = st.number_input("Load average — 15 minutes", min_value=0.0, value=16.0, step=0.5)
    manual_submit = st.form_submit_button("Predict next-hour event", type="primary")

if manual_submit:
    row = pd.DataFrame([{
        "cpu_util": cpu, "memory_util": memory, "disk_util": disk,
        "load1": load1, "load5": load5, "load15": load15
    }], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])
    predicted_class = int(model.predict(row)[0])
    if predicted_class == 1:
        st.warning(f"Model prediction: possible recorded event within one hour (estimated score: {probability:.1%}).")
    else:
        st.success(f"Model prediction: no recorded event predicted within one hour (estimated score for event: {probability:.1%}).")
    st.caption("The score is a model estimate, not a calibrated guarantee or a diagnosis of server health.")

st.divider()
st.header("2. Predict from a CSV file")
st.write("Upload a CSV containing these six feature columns. Each row will receive a prediction and estimated event score.")
st.code(", ".join(FEATURES), language="text")
st.download_button(
    "Download sample CSV template",
    data=make_template(),
    file_name="server_prediction_template.csv",
    mime="text/csv",
)
uploaded = st.file_uploader("Choose a prediction CSV", type=["csv"])

if uploaded is not None:
    try:
        input_df = pd.read_csv(uploaded)
        missing = [column for column in FEATURES if column not in input_df.columns]
        if missing:
            st.error("Missing required column(s): " + ", ".join(missing))
            st.info("Use the sample template above. Column names must match exactly.")
        elif input_df.empty:
            st.warning("The uploaded CSV has no data rows.")
        else:
            if len(input_df) > 100_000:
                st.error("For this student demo, upload at most 100,000 rows at a time.")
            else:
                numeric_df = input_df[FEATURES].apply(pd.to_numeric, errors="coerce")
                all_missing = numeric_df.isna().all(axis=1)
                if all_missing.any():
                    st.warning(
                        f"{int(all_missing.sum())} row(s) have no usable numeric feature values. "
                        "Those rows will be processed with median imputation."
                    )
                predictions = model.predict(numeric_df)
                probabilities = model.predict_proba(numeric_df)[:, 1]
                output = input_df.copy()
                output["predicted_event_next_hour"] = predictions
                output["event_probability"] = np.round(probabilities, 4)
                output["prediction_label"] = np.where(
                    predictions == 1, "Possible recorded event", "No event predicted"
                )
                st.success(f"Generated predictions for {len(output):,} row(s).")
                st.dataframe(output.head(100), use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="server_failure_predictions.csv",
                    mime="text/csv",
                )
    except Exception as exc:
        st.error(f"Could not read or predict from this CSV: {exc}")

with st.expander("What the CSV columns mean"):
    st.markdown(
        """
        - `cpu_util`: CPU utilization percentage.
        - `memory_util`: memory utilization percentage.
        - `disk_util`: disk utilization percentage.
        - `load1`, `load5`, `load15`: load averages over 1, 5 and 15 minutes.
        - `predicted_event_next_hour`: `1` means the model predicts an event; `0` means it does not.
        - `event_probability`: estimated probability-like model score for class 1.
        - `prediction_label`: readable version of the prediction.

        The uploaded prediction file does **not** need a target column. Do not include the target
        column when asking the model to predict new rows.
        """
    )

st.caption("Academic demonstration only • Random Forest • Median imputation • One-hour prediction horizon")
