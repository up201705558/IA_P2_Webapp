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

PEAK_MONTHS = [4, 5, 6, 7]  # heuristic for low-trust

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
st.title("Horas Totais — Aplicação de Previsão")
st.caption("Previsão de Horas Mensais por Cliente")

st.sidebar.header("Configuração")
data_dir = st.sidebar.text_input("Pasta dos dados", value=DATA_DIR_DEFAULT)
if st.sidebar.button("Recarregar dados"):
    st.cache_data.clear()

# Load data
try:
    clientes, mensal, df = load_data(data_dir)
except Exception as e:
    st.error(str(e))
    st.stop()

# Add client UI (sidebar)
st.sidebar.subheader("Adicionar novo cliente")
with st.sidebar.form("add_client", clear_on_submit=False):
    next_id_default = int(clientes['cliente_id'].max() + 1) if not clientes.empty else 1
    new_id = int(st.number_input("cliente_id", min_value=1, value=next_id_default, step=1))
    new_nome = st.text_input("nome (opcional)", value="")
    new_forma = st.selectbox("forma_juridica", options=sorted(clientes["forma_juridica"].dropna().unique().tolist()) if "forma_juridica" in clientes.columns else [""])
    new_setor = st.selectbox("setor", options=sorted(clientes["setor"].dropna().unique().tolist()) if "setor" in clientes.columns else [""])
    new_regime_iva = st.selectbox("regime_iva", options=clientes["regime_iva"].unique().tolist() if "regime_iva" in clientes.columns else ["normal"])
    new_regime_cont = st.selectbox("regime_contabilistico", options=clientes["regime_contabilistico"].unique().tolist() if "regime_contabilistico" in clientes.columns else ["Geral"])
    new_dim = int(st.number_input("dimensao_colaboradores", min_value=0, value=1, step=1))
    new_antig = int(st.number_input("antiguidade_meses", min_value=0, value=36, step=1))
    new_avenca = float(st.number_input("avenca_atual_eur (€/mês)", min_value=0.0, value=150.0, step=10.0))
    submitted_client = st.form_submit_button("Adicionar cliente")

if submitted_client:
    new_row = {
        "cliente_id": new_id,
        "forma_juridica": new_forma,
        "setor": new_setor,
        "regime_iva": new_regime_iva,
        "regime_contabilistico": new_regime_cont,
        "dimensao_colaboradores": new_dim,
        "antiguidade_meses": new_antig,
        "avenca_atual_eur": new_avenca,
    }
    if new_nome:
        new_row["nome"] = new_nome
    clientes = pd.concat([clientes, pd.DataFrame([new_row])], ignore_index=True)
    clientes.to_csv(Path(data_dir) / "clientes.csv", index=False)
    st.success(f"Cliente {new_id} adicionado. clientes.csv atualizado.")

st.sidebar.write(f"Linhas (mensal_cliente): {len(mensal):,}")
st.sidebar.write(f"Clientes únicos: {clientes['cliente_id'].nunique():,}")

# Load artifacts (pre-trained)
try:
    artifacts = load_artifacts(ARTIFACT_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

model = artifacts["model"]
model_columns = artifacts["train_columns"]
resid_std = artifacts.get("resid_std", 0.0)

st.sidebar.subheader("Modelo")
st.sidebar.write(f"Algoritmo: Ridge (regressão linear regularizada)")
st.sidebar.write(f"Hiperparâmetros: {artifacts.get('grid_best_params', {})}")
st.sidebar.write(f"MAE em teste: {artifacts.get('mae_test', float('nan')):.3f} h")

# Coeffs
st.sidebar.subheader("Coeficientes do modelo (top 10)")
if hasattr(model, "named_steps") and "reg" in model.named_steps:
    coefs = model.named_steps["reg"].coef_
    coef_df_sb = pd.DataFrame({"feature": model_columns, "coef": coefs})
    coef_df_sb["abs_coef"] = coef_df_sb["coef"].abs()
    coef_df_sb = coef_df_sb.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    st.sidebar.dataframe(
        coef_df_sb.head(10).loc[:, ["feature", "coef"]].set_index("feature").round(3),
        use_container_width=True,
    )
    st.sidebar.caption(
        "Coeficientes em unidades standardizadas — quanto maior o |valor|, "
        "maior a influência da feature no output."
    )
else:
    st.sidebar.write("O modelo não expõe coeficientes lineares.")

# Single prediction
st.header("1) Previsão individual (input manual)")
with st.form("single"):
    st.write("Preencher os atributos do cliente — usar os mesmos rótulos das categorias do dataset.")
    col1, col2 = st.columns(2)
    cliente_id = st.number_input("cliente_id (optional)", min_value=0, value=0, step=1)
    nr_documentos = col1.number_input("nr_documentos", min_value=0, value=10)
    nr_lancamentos = col2.number_input("nr_lancamentos", min_value=0, value=15)
    nr_colaboradores_processados = col1.number_input("nr_colaboradores_processados", min_value=0, value=1)
    nr_movimentos_bancarios = col2.number_input("nr_movimentos_bancarios", min_value=0, value=5)
    tem_iva_a_entregar = col1.selectbox("tem_iva_a_entregar", [0, 1], index=0)
    tem_dmr = col2.selectbox("tem_dmr", [0, 1], index=0)
    tem_modelo_22 = col1.selectbox("tem_modelo_22", [0, 1], index=0)
    tem_ies = col2.selectbox("tem_ies", [0, 1], index=0)
    eh_pico_fiscal = col1.selectbox("eh_pico_fiscal", [0, 1], index=0)
    dim_col = col2.number_input("dimensao_colaboradores", min_value=0, value=1)
    mes = st.number_input("mes (1-12)", min_value=1, max_value=12, value=1)
    forma_juridica = st.selectbox("forma_juridica", options=sorted(clientes["forma_juridica"].dropna().unique().tolist()) if "forma_juridica" in clientes.columns else [""])
    setor = st.selectbox("setor", options=sorted(clientes["setor"].dropna().unique().tolist()) if "setor" in clientes.columns else [""])
    regime_iva = st.selectbox("regime_iva", options=clientes["regime_iva"].unique().tolist() if "regime_iva" in clientes.columns else ["normal"])
    regime_contabilistico = st.selectbox("regime_contabilistico", options=clientes["regime_contabilistico"].unique().tolist() if "regime_contabilistico" in clientes.columns else ["Geral"])
    submitted = st.form_submit_button("Prever")

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
    st.metric("Horas previstas (h)", f"{pred[0]:.2f}")
    lower, upper = pred[0] - 1.96 * resid_std, pred[0] + 1.96 * resid_std
    st.write(f"Intervalo aproximado 95% (com base nos resíduos do teste): [{lower:.2f}, {upper:.2f}] h")
    reasons = mark_low_trust(row, pred[0])
    if reasons:
        st.warning("Atenção — sinais de baixa confiança: " + ", ".join(reasons))
    else:
        st.success("Confiança na previsão: normal")

# Batch prediction
st.header("2) Previsão em lote (upload de CSV)")
st.write("Carregue um CSV com as mesmas colunas que `mensal_cliente.csv` (ou um subconjunto que possa ser fundido com `clientes.csv`).")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    batch = pd.read_csv(uploaded)
    if "mes" not in batch.columns:
        st.error("O CSV carregado tem de incluir a coluna 'mes'.")
    else:
        need_merge = any(col not in batch.columns for col in NUM_FEATURES + CAT_FEATURES)
        if need_merge and "cliente_id" in batch.columns:
            batch = batch.merge(clientes, on="cliente_id", how="left")

        missing_cols = [c for c in NUM_FEATURES + CAT_FEATURES + ["cliente_id"] if c not in batch.columns]
        if missing_cols:
            st.warning(f"Colunas em falta preenchidas com zeros/defaults: {missing_cols}")
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
        st.subheader("Previsões (vista resumida)")
        st.dataframe(compact.head(200))

        # Per-client sum of predicted hours
        st.subheader("Total estimado por cliente (no CSV carregado)")
        per_client = out.groupby("cliente_id", dropna=False)["y_pred"].sum().reset_index()
        per_client = per_client.rename(columns={"y_pred": "horas_estimadas_total"})
        st.dataframe(per_client)

        # download full predictions (full out)
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Descarregar previsões completas (CSV)", data=csv_bytes, file_name="predictions.csv")

st.caption("Vista resumida mostra apenas as colunas relevantes; o download devolve o CSV completo.")

# 3) Prediction vs Current Retainer Comparison (BG2 — potentially underpriced clients)
st.header("3) Comparação Previsão × Avença atual")
st.write(
    "Estima o tempo médio mensal por cliente nos últimos 12 meses, multiplica por um "
    "**custo-hora de referência** e compara com a avença atualmente cobrada. "
    "Clientes onde o valor estimado **supera** a avença em mais do limiar definido "
    "são candidatos a revisão de pricing."
)
 
col_a, col_b, col_c = st.columns(3)
custo_hora = col_a.number_input("Custo-hora de referência (€/h)", min_value=0.0, value=20.0, step=1.0)
threshold_pct = col_b.number_input("Limiar de discrepância (%)", min_value=0.0, value=20.0, step=5.0)
n_meses_hist = col_c.number_input("Meses recentes a considerar", min_value=1, max_value=36, value=12, step=1)
 
if st.button("Calcular comparação"):
    if "avenca_atual_eur" not in clientes.columns:
        st.error("A coluna 'avenca_atual_eur' não existe em clientes.csv — não é possível comparar.")
    else:
        # Use the last n months per client
        df_recent = (
            df.sort_values(["cliente_id", "ano", "mes"], ascending=[True, False, False])
              .groupby("cliente_id", as_index=False)
              .head(int(n_meses_hist))
        )
        # Prepare features and predict
        X_recent_raw = df_recent[NUM_FEATURES + CAT_FEATURES + ["cliente_id"]].copy()
        X_recent = align_and_compute(X_recent_raw, model_columns)
        df_recent = df_recent.copy()
        df_recent["horas_previstas"] = model.predict(X_recent)
 
        # Average predicted hours per client
        comp = (
            df_recent.groupby("cliente_id", as_index=False)
                     .agg(horas_medias_previstas=("horas_previstas", "mean"))
        )
        comp = comp.merge(
            clientes[["cliente_id", "avenca_atual_eur", "dimensao_colaboradores"]],
            on="cliente_id", how="left",
        )
        comp["valor_justo_eur"] = (comp["horas_medias_previstas"] * custo_hora).round(2)
        comp["delta_eur"] = (comp["valor_justo_eur"] - comp["avenca_atual_eur"]).round(2)
        comp["delta_pct"] = (comp["delta_eur"] / comp["avenca_atual_eur"] * 100).round(1)
        comp["horas_medias_previstas"] = comp["horas_medias_previstas"].round(2)
 
        # Categorization
        def status(delta_pct):
            if delta_pct > threshold_pct:
                return "Sub-precificado (a cobrar pouco)"
            if delta_pct < -threshold_pct:
                return "Sobre-precificado (a cobrar muito)"
            return "Equilibrado"
        comp["status"] = comp["delta_pct"].apply(status)
 
        # Sort by largest absolute deviations
        comp = comp.sort_values("delta_pct", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
 
        st.subheader("Resumo geral")
        counts = comp["status"].value_counts().to_dict()
        col1, col2, col3 = st.columns(3)
        col1.metric("Sub-precificados", counts.get("Sub-precificado (a cobrar pouco)", 0))
        col2.metric("Equilibrados",       counts.get("Equilibrado", 0))
        col3.metric("Sobre-precificados", counts.get("Sobre-precificado (a cobrar muito)", 0))
 
        st.subheader("Detalhe por cliente (ordenado por maior desvio)")
        st.dataframe(
            comp[[
                "cliente_id", "horas_medias_previstas", "valor_justo_eur",
                "avenca_atual_eur", "delta_eur", "delta_pct", "status",
            ]]
        )
 
        st.caption(
            "Notas: (1) o cálculo do *valor justo* assume um custo-hora fixo — em produção, "
            "este parâmetro depende da estrutura de custos do gabinete. "
            "(2) A previsão tem maior incerteza para clientes grandes (ver análise em 5.3 do notebook). "
            "(3) Use estas estimativas como **apoio à decisão**, não como verdade absoluta."
        )
 
st.caption("App © Grupo A2_6 — Projeto IART, FEUP, 2025/26")