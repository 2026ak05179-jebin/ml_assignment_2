import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

MODEL_DIR = Path(__file__).parent / "model"
TARGET = "y"
LABELS = {0: "no", 1: "yes"}
MAROON = LinearSegmentedColormap.from_list("maroon", ["#fbeeee", "#d98c8c", "#a83a3a", "#800000", "#4d0000"])
MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


def prepare_target(series):
    """Accept the target as 0/1 or as no/yes text and return 0/1 ints."""
    if pd.api.types.is_numeric_dtype(series):
        bad = sorted(set(series.dropna().unique()) - {0, 1})
        if bad or series.isna().any():
            raise ValueError(f"Column '{TARGET}' must be 0/1 or no/yes. Found unexpected values: {bad}")
        return series.astype(int)
    mapped = series.astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    if mapped.isna().any():
        raise ValueError(f"Column '{TARGET}' must be 0/1 or no/yes.")
    return mapped.astype(int)


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, pos_label=1),
        "Recall": recall_score(y, pred, pos_label=1),
        "F1": f1_score(y, pred, pos_label=1),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred


st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")
st.title("Bank Marketing – Term Deposit Prediction")
st.caption("UCI Bank Marketing dataset, 4,521 rows. Target is imbalanced (88.5% no / 11.5% yes), "
          "so AUC and MCC matter more here than raw accuracy.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Model", list(MODELS))

if uploaded is None:
    st.info("Upload a CSV in the sidebar to get started.")
    st.stop()

raw = uploaded.getvalue().decode("utf-8-sig")
header = raw.splitlines()[0] if raw else ""
sep = ";" if header.count(";") > header.count(",") else ","  # bank.csv is ;-separated, test_data.csv is ,-separated
df = pd.read_csv(io.StringIO(raw), sep=sep)

if TARGET not in df.columns:
    st.error(f"Uploaded file has no target column named '{TARGET}'.")
    st.stop()

try:
    y = prepare_target(df[TARGET])
except ValueError as e:
    st.error(str(e))
    st.stop()
X = df.drop(columns=TARGET)

pipe = load_model(MODELS[model_name])
missing = [c for c in pipe.feature_names_in_ if c not in X.columns]
if missing:
    st.error(f"Uploaded file is missing required feature columns: {missing}")
    st.stop()
X = X[list(pipe.feature_names_in_)]

if y.nunique() < 2:
    st.error("Target column has only one class — AUC and MCC can't be computed.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features", X.shape[1])
c3.metric("Positive rate (yes)", f"{y.mean():.2%}")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20))

st.header(model_name)
metrics, pred = evaluate(pipe, X, y)
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

st.subheader("Confusion matrix")
cm = confusion_matrix(y, pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap=MAROON, cbar=False, ax=ax,
            annot_kws={"fontsize": 14},
            xticklabels=[LABELS[0], LABELS[1]], yticklabels=[LABELS[0], LABELS[1]])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
plt.close(fig)

st.header("All models on this file")
rows = {name: evaluate(load_model(fn), X, y)[0] for name, fn in MODELS.items()}
comparison = pd.DataFrame.from_dict(rows, orient="index")
comparison.index.name = "Model"
st.dataframe(comparison.round(4).style.highlight_max(
    axis=0, props="background-color: #1b5e20; color: white; font-weight: bold;"))
st.caption("Dark green = best value per metric.")

st.divider()
st.caption("Jebin Johnson · 2026AK05179 — pipelines are pre-trained; this app only loads and predicts.")