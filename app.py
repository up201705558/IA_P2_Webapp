# app.py — Streamlit app (patched)
# - loads pre-trained model artifacts (artifacts/ridge_artifacts.joblib)
# - supports adding clients (appends to data/clientes.csv)
# - single & batch prediction with low-trust flags
# - for batch uploads: show a compact view (cliente_id, ano, mes, horas_estimadas, low_trust_reasons, confidence_level)
#   while download still returns full predictions CSV
# - also shows per-client sum of predicted hours below the compact view

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 4)

# ---------- Constants ----------
ID_COL = "cliente_id"
TARGET = "horas_totais"

NUM_FEATURES = [
    "nr_documentos", "nr_lancamentos", "nr_colaboradores_processados",
    "nr_movimentos_bancarios", "tem_iva_a_entregar", "tem_dmr",
    "tem_modelo_22", "tem_ies", "eh_pico_fiscal",
    "dimensao_colaboradores", "mes",
]

CAT_FEATURES = [
    "forma_juridica", "setor", "regime_iva", "regime_contabilistico",
]

PEAK_MONTHS = [12]  # heuristic for low-trust

ARTIFACT_PATH = Path("artifacts") / "ridge_artifacts.joblib"
DATA_DIR_DEFAULT = "data"

# ---------- Helpers ----------
@st.cache_data
def load_data(data_dir=DATA_DIR_DEFAULT):
    clientes_path = Path(data_dir) / "clientes.csv"
    mensal_path = Path(data_dir) / "mensal_cliente.csv"
    if clientes_path.exists() and mensal_path.exists():
        clientes = pd.read_csv(clientes_path)
        mensal = pd.read_csv(mensal_path)
    else:
        raise FileNotFoundError("Place clientes.csv and mensal_cliente.csv into ./data/")
    df = mensal.merge(clientes, on="cliente_id")
    return clientes, mensal, df

@st.cache_data
def load_artifacts(path=ARTIFACT_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found — run training script to generate artifacts.")
    artifacts = joblib.load(path)
    return artifacts

def prepare_features(df):
    X_raw = df[NUM_FEATURES + CAT_FEATURES + [ID_COL]].copy()
    y = df[TARGET].copy() if TARGET in df.columns else None

    X = pd.get_dummies(X_raw, columns=CAT_FEATURES, drop_first=True, dtype=int)

    # derived features
    X["lancamentos_por_documento"] = np.where(
        X["nr_documentos"] > 0,
        X["nr_lancamentos"] / X["nr_documentos"],
        0.0,
    )
    X["documentos_por_colaborador"] = np.where(
        X["dimensao_colaboradores"] > 0,
        X["nr_documentos"] / X["dimensao_colaboradores"],
        X["nr_documentos"].astype(float),
    )
    return X, y

def align_and_compute(df_raw, model_columns):
    X_full = pd.get_dummies(df_raw.copy(), columns=CAT_FEATURES, drop_first=True, dtype=int)

    # derived features
    if "nr_documentos" in X_full.columns and "nr_lancamentos" in X_full.columns:
        X_full["lancamentos_por_documento"] = np.where(
            X_full["nr_documentos"] > 0,
            X_full["nr_lancamentos"] / X_full["nr_documentos"],
            0.0,
        )
    if "dimensao_colaboradores" in X_full.columns:
        X_full["documentos_por_colaborador"] = np.where(
            X_full["dimensao_colaboradores"] > 0,
            X_full["nr_documentos"] / X_full["dimensao_colaboradores"],
            X_full.get("nr_documentos", 0).astype(float),
        )
    for c in model_columns:
        if c not in X_full.columns:
            X_full[c] = 0
    X_full = X_full[model_columns]
    return X_full

def predict_rows(model, rows_df):
    return model.predict(rows_df)

def mark_low_trust(row, pred):
    reasons = []
    dim = row.get("dimensao_colaboradores", None)
    if pd.notna(dim) and dim >= 6:
        reasons.append("Large client")
    m = int(row.get("mes", -1)) if not pd.isna(row.get("mes", np.nan)) else -1
    if m in PEAK_MONTHS:
        reasons.append("Peak month")
    if pred > 12:
        reasons.append("Large predicted hours")
    return reasons

def compute_confidence(preds, resid_std):
    interval_half = 1.96 * resid_std
    rel_unc = interval_half / np.maximum(np.abs(preds), 1.0)
    conf = 100.0 * (1.0 - rel_unc)
    conf = np.clip(conf, 0.0, 100.0)
    return np.round(conf, 0).astype(int)

# ---------- Streamlit UI ----------
st.title("Horas Totais — Prediction App (pre-trained)")

st.sidebar.header("Paths & controls")
data_dir = st.sidebar.text_input("Data folder", value=DATA_DIR_DEFAULT)
if st.sidebar.button("Reload data"):
    st.cache_data.clear()

# Load data
try:
    clientes, mensal, df = load_data(data_dir)
except Exception as e:
    st.error(str(e))
    st.stop()

# Add client UI (sidebar)
st.sidebar.subheader("Add new client")
with st.sidebar.form("add_client", clear_on_submit=False):
    next_id_default = int(clientes['cliente_id'].max() + 1) if not clientes.empty else 1
    new_id = int(st.number_input("cliente_id", min_value=1, value=next_id_default, step=1))
    new_nome = st.text_input("nome (optional)", value="")
    new_forma = st.text_input("forma_juridica", value=clientes["forma_juridica"].mode()[0] if "forma_juridica" in clientes.columns else "")
    new_setor = st.text_input("setor", value=clientes["setor"].mode()[0] if "setor" in clientes.columns else "")
    new_regime_iva = st.selectbox("regime_iva", options=clientes["regime_iva"].unique().tolist() if "regime_iva" in clientes.columns else ["normal"])
    new_regime_cont = st.selectbox("regime_contabilistico", options=clientes["regime_contabilistico"].unique().tolist() if "regime_contabilistico" in clientes.columns else ["Geral"])
    submitted_client = st.form_submit_button("Add client")

if submitted_client:
    new_row = {
        "cliente_id": new_id,
        "nome": new_nome,
        "forma_juridica": new_forma,
        "setor": new_setor,
        "regime_iva": new_regime_iva,
        "regime_contabilistico": new_regime_cont,
    }
    clientes = pd.concat([clientes, pd.DataFrame([new_row])], ignore_index=True)
    clientes.to_csv(Path(data_dir) / "clientes.csv", index=False)
    st.success(f"Client {new_id} added and clientes.csv saved.")

st.sidebar.write(f"Rows (mensal): {len(mensal):,}")
st.sidebar.write(f"Unique clients: {clientes['cliente_id'].nunique():,}")

# Load artifacts (pre-trained)
try:
    artifacts = load_artifacts(ARTIFACT_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

model = artifacts["model"]
model_columns = artifacts["train_columns"]
resid_std = artifacts.get("resid_std", 0.0)

st.sidebar.subheader("Model info")
st.sidebar.write(artifacts.get("grid_best_params", {}))
st.sidebar.write(f"Model test MAE: {artifacts.get('mae_test', float('nan')):.3f}")

# Coeffs
st.header("1) Model coefficients (top features)")
if hasattr(model, "named_steps") and "reg" in model.named_steps:
    coefs = model.named_steps["reg"].coef_
    feat_names = model_columns
    coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    st.dataframe(coef_df.head(20).loc[:, ["feature", "coef"]].set_index("feature"))
else:
    st.write("Model does not expose linear coefficients to display.")

# Single prediction
st.header("2) Single prediction (manual input)")
with st.form("single"):
    st.write("Fill client attributes (use categorical exact labels as in clientes.csv).")
    col1, col2 = st.columns(2)
    cliente_id = st.number_input("cliente_id (optional)", min_value=0, value=0, step=1)
    nr_documentos = col1.number_input("nr_documentos", min_value=0, value=10)
    nr_lancamentos = col2.number_input("nr_lancamentos", min_value=0, value=15)
    nr_colaboradores_processados = col1.number_input("nr_colaboradores_processados", min_value=0, value=1)
    nr_movimentos_bancarios = col2.number_input("nr_movimentos_bancarios", min_value=0, value=5)
    tem_iva_a_entregar = col1.selectbox("tem_iva_a_entregar", [0, 1], index=1)
    tem_dmr = col2.selectbox("tem_dmr", [0, 1], index=1)
    tem_modelo_22 = col1.selectbox("tem_modelo_22", [0, 1], index=0)
    tem_ies = col2.selectbox("tem_ies", [0, 1], index=0)
    eh_pico_fiscal = col1.selectbox("eh_pico_fiscal", [0, 1], index=0)
    dim_col = col2.number_input("dimensao_colaboradores", min_value=0, value=1)
    mes = st.number_input("mes (1-12)", min_value=1, max_value=12, value=6)
    forma_juridica = st.text_input("forma_juridica (categorical)", value=clientes["forma_juridica"].mode()[0] if "forma_juridica" in clientes.columns else "")
    setor = st.text_input("setor (categorical)", value=clientes["setor"].mode()[0] if "setor" in clientes.columns else "")
    regime_iva = st.selectbox("regime_iva", options=clientes["regime_iva"].unique().tolist() if "regime_iva" in clientes.columns else ["normal"])
    regime_contabilistico = st.selectbox("regime_contabilistico", options=clientes["regime_contabilistico"].unique().tolist() if "regime_contabilistico" in clientes.columns else ["Geral"])
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "nr_documentos": nr_documentos,
        "nr_lancamentos": nr_lancamentos,
        "nr_colaboradores_processados": nr_colaboradores_processados,
        "nr_movimentos_bancarios": nr_movimentos_bancarios,
        "tem_iva_a_entregar": tem_iva_a_entregar,
        "tem_dmr": tem_dmr,
        "tem_modelo_22": tem_modelo_22,
        "tem_ies": tem_ies,
        "eh_pico_fiscal": eh_pico_fiscal,
        "dimensao_colaboradores": dim_col,
        "mes": mes,
        "cliente_id": cliente_id,
        "forma_juridica": forma_juridica,
        "setor": setor,
        "regime_iva": regime_iva,
        "regime_contabilistico": regime_contabilistico,
    }
    row_df = pd.DataFrame([row])
    X_full = align_and_compute(row_df[NUM_FEATURES + CAT_FEATURES + ["cliente_id"]], model_columns)
    pred = predict_rows(model, X_full)
    st.metric("Predicted horas_totais (h)", f"{pred[0]:.2f}")
    lower, upper = pred[0] - 1.96 * resid_std, pred[0] + 1.96 * resid_std
    st.write(f"Approx. 95% interval (based on test residuals): [{lower:.2f}, {upper:.2f}] h")
    reasons = mark_low_trust(row, pred[0])
    if reasons:
        st.warning("Low trust flags: " + ", ".join(reasons))
    else:
        st.success("Prediction confidence: normal")

# Batch prediction
st.header("3) Batch prediction (CSV upload)")
st.write("Upload a CSV with the same columns as mensal_cliente.csv (or a subset that can be merged with clientes.csv).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    batch = pd.read_csv(uploaded)
    if "mes" not in batch.columns:
        st.error("Uploaded CSV must include column 'mes'.")
    else:
        need_merge = any(col not in batch.columns for col in NUM_FEATURES + CAT_FEATURES)
        if need_merge and "cliente_id" in batch.columns:
            batch = batch.merge(clientes, on="cliente_id", how="left")

        missing_cols = [c for c in NUM_FEATURES + CAT_FEATURES + ["cliente_id"] if c not in batch.columns]
        if missing_cols:
            st.warning(f"Missing columns filled with zeros/defaults: {missing_cols}")
            for c in missing_cols:
                batch[c] = 0

        batch_X_raw = batch[NUM_FEATURES + CAT_FEATURES + ["cliente_id"]].copy()
        batch_X = align_and_compute(batch_X_raw, model_columns)
        preds = predict_rows(model, batch_X)
        out = batch.copy().reset_index(drop=True)
        out["y_pred"] = preds
        out["low_trust_reasons"] = out.apply(lambda r: ", ".join(mark_low_trust(r.to_dict(), r["y_pred"])), axis=1)

        # compute confidence level column
        out["confidence_level"] = compute_confidence(out["y_pred"].values, resid_std)

        # Compact display: cliente_id, ano, mes, horas_estimadas, low_trust_reasons, confidence_level
        display = out.copy()
        if "ano" not in display.columns:
            display["ano"] = np.nan
        compact = display[[ "cliente_id", "ano", "mes", "y_pred", "low_trust_reasons", "confidence_level"]].copy()
        compact = compact.rename(columns={"y_pred": "horas_estimadas"})
        st.subheader("Compact predictions (for quick review)")
        st.dataframe(compact.head(200))

        # Per-client sum of predicted hours
        st.subheader("Per-client total estimated hours (in uploaded set)")
        per_client = out.groupby("cliente_id", dropna=False)["y_pred"].sum().reset_index()
        per_client = per_client.rename(columns={"y_pred": "horas_estimadas_total"})
        st.dataframe(per_client)

        # download full predictions (full out)
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download full predictions CSV", data=csv_bytes, file_name="predictions.csv")

st.caption("Compact display shows only relevant columns; Download returns the full predictions CSV.")
