# train_save.py
import joblib
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_validate
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "data"
OUT_DIR = "artifacts"
Path(OUT_DIR).mkdir(exist_ok=True)

# load
clientes = pd.read_csv(Path(DATA_DIR) / "clientes.csv")
mensal = pd.read_csv(Path(DATA_DIR) / "mensal_cliente.csv")
df = mensal.merge(clientes, on="cliente_id")

# reproduce prepare_features used in app
NUM_FEATURES = [
    "nr_documentos", "nr_lancamentos", "nr_colaboradores_processados",
    "nr_movimentos_bancarios", "tem_iva_a_entregar", "tem_dmr",
    "tem_modelo_22", "tem_ies", "eh_pico_fiscal",
    "dimensao_colaboradores", "mes",
]
CAT_FEATURES = ["forma_juridica", "setor", "regime_iva", "regime_contabilistico"]
TARGET = "horas_totais"
ID_COL = "cliente_id"

X_raw = df[NUM_FEATURES + CAT_FEATURES + [ID_COL]].copy()
y = df[TARGET].copy()
X = pd.get_dummies(X_raw, columns=CAT_FEATURES, drop_first=True, dtype=int)
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

# group-aware split
groups = X[ID_COL].values
splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train = X.iloc[train_idx].drop(columns=[ID_COL]).reset_index(drop=True)
X_test = X.iloc[test_idx].drop(columns=[ID_COL]).reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)
groups_train = X.iloc[train_idx][ID_COL].values

# tune Ridge (small grid)
gkf = GroupKFold(n_splits=5)
pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(random_state=42))])
param_grid = {"reg__alpha": [10, 50, 100, 500, 1000]}
grid = GridSearchCV(pipe, param_grid, cv=gkf, scoring="neg_mean_absolute_error", n_jobs=-1, refit=True)
t0 = time.time()
grid.fit(X_train, y_train, groups=groups_train)
elapsed = time.time() - t0
best = grid.best_estimator_
best.fit(X_train, y_train)

# test residuals and metrics
y_pred_test = best.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = (mean_squared_error(y_test, y_pred_test)) ** 0.5
residuals = (y_test - y_pred_test).reset_index(drop=True)
resid_std = residuals.std()

# save artifacts: model, training-columns, residual std, coef table, metadata
artifacts = {
    "model": best,
    "train_columns": X_train.columns.tolist(),
    "resid_std": float(resid_std),
    "mae_test": float(mae_test),
    "rmse_test": float(rmse_test),
    "grid_best_params": grid.best_params_,
}

joblib.dump(artifacts, Path(OUT_DIR) / "ridge_artifacts.joblib")
print("Saved artifacts to", Path(OUT_DIR) / "ridge_artifacts.joblib")
