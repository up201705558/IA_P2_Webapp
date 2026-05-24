"""
data_generation.py
==================
Synthetic data generator for the project.
Follows the specification described in section 2.3 of the notebook (Data description).

Typical usage:
    from data_generation import generate_dataset
    clientes, mensal = generate_dataset(seed=42, save_to='data/')
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationConfig:
    """Parameters that control dataset generation."""

    n_clientes: int = 60
    n_meses: int = 36          # 3 anos
    mes_fim: tuple[int, int] = (2026, 5)   # (ano, mês) do último mês de dados
    seed: int = 42


# Categorical domains (aligned with the table in section 2.3)
FORMAS_JURIDICAS = ["ENI", "Unipessoal Lda.", "Lda.", "S.A."]
FORMAS_JURIDICAS_PROBS = [0.15, 0.40, 0.40, 0.05]

SETORES = ["comércio", "serviços", "restauração",
           "construção", "indústria", "profissional liberal"]
SETORES_PROBS = [0.25, 0.25, 0.10, 0.15, 0.10, 0.15]

REGIMES_IVA = ["mensal", "trimestral"]
REGIMES_IVA_PROBS = [0.25, 0.75]   # micro-empresas estão maioritariamente em trimestral

REGIMES_CONT = ["organizada", "simplificada"]
REGIMES_CONT_PROBS = [0.80, 0.20]


# ---------------------------------------------------------------------------
# Generation of the `clientes` table
# ---------------------------------------------------------------------------

def _gerar_clientes(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the `clientes` table (one row per client)."""
    n = cfg.n_clientes

    # Size (number of employees) — biased towards 0–5 (micro companies)
    # A truncated lognormal produces the long tail typical of real firms
    dimensao = np.clip(
        np.round(rng.lognormal(mean=0.7, sigma=0.9, size=n)).astype(int),
        0, 30,
    )

    # Seniority with the firm in months (between 36 and 240; long-term relationship)
    antiguidade = rng.integers(low=36, high=241, size=n)

    forma_juridica = rng.choice(FORMAS_JURIDICAS, size=n, p=FORMAS_JURIDICAS_PROBS)
    setor = rng.choice(SETORES, size=n, p=SETORES_PROBS)
    regime_iva = rng.choice(REGIMES_IVA, size=n, p=REGIMES_IVA_PROBS)
    regime_cont = rng.choice(REGIMES_CONT, size=n, p=REGIMES_CONT_PROBS)

    # Historical monthly retainer (50–500€), approximately proportional to size
    # but with noise (reflecting sensitivity-based pricing)
    avenca_base = 80 + 25 * dimensao
    avenca_ruido = rng.normal(loc=0, scale=40, size=n)
    avenca = np.clip(avenca_base + avenca_ruido, 50, 500).round(2)

    # Latent variable: quality of submitted documents (0..1)
    # This does NOT enter the final public dataset — it remains in `clientes`
    # only for the `mensal_cliente` generator. Models rule #5 from section 2.3.
    qualidade_docs_latente = rng.beta(a=4, b=2, size=n)

    return pd.DataFrame({
        "cliente_id": np.arange(1, n + 1),
        "forma_juridica": forma_juridica,
        "setor": setor,
        "regime_iva": regime_iva,
        "regime_contabilistico": regime_cont,
        "dimensao_colaboradores": dimensao,
        "antiguidade_meses": antiguidade,
        "avenca_atual_eur": avenca,
        "_qualidade_docs_latente": qualidade_docs_latente,   # privado (prefixo _)
    })


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------

def _gerar_calendario(cfg: GenerationConfig) -> pd.DataFrame:
    """Generate the sequence of (year, month) that composes the time window."""
    ano_fim, mes_fim = cfg.mes_fim
    # Build chronologically, ending at (ano_fim, mes_fim)
    meses = []
    ano, mes = ano_fim, mes_fim
    for _ in range(cfg.n_meses):
        meses.append((ano, mes))
        mes -= 1
        if mes == 0:
            mes = 12
            ano -= 1
    meses.reverse()
    return pd.DataFrame(meses, columns=["ano", "mes"])


# ---------------------------------------------------------------------------
# Fiscal obligations rules (rules #2 and #3 from section 2.3)
# ---------------------------------------------------------------------------

def _aplicar_obrigacoes_fiscais(
    df: pd.DataFrame,
    clientes: pd.DataFrame,
) -> pd.DataFrame:
    """Apply Portuguese fiscal calendar rules."""
    # Merge to make `regime_iva` available
    df = df.merge(clientes[["cliente_id", "regime_iva"]], on="cliente_id")

    # VAT filing: monthly -> every month; quarterly -> months 2,5,8,11
    iva_trim_meses = {2, 5, 8, 11}
    df["tem_iva_a_entregar"] = (
        (df["regime_iva"] == "mensal")
        | ((df["regime_iva"] == "trimestral") & df["mes"].isin(iva_trim_meses))
    ).astype(int)

    df["tem_modelo_22"] = (df["mes"] == 5).astype(int)   # annual corporate tax filing
    df["tem_ies"] = (df["mes"] == 7).astype(int)         # annual IES filing
    df["tem_irs_anual"] = df["mes"].isin([4, 6]).astype(int)
    df["eh_pico_fiscal"] = df["mes"].isin([4, 5, 7]).astype(int)  # months with fiscal peaks

    df = df.drop(columns=["regime_iva"])
    return df


# ---------------------------------------------------------------------------
# Generation of the `mensal_cliente` table
# ---------------------------------------------------------------------------

def _gerar_mensal(
    cfg: GenerationConfig,
    clientes: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate the `mensal_cliente` table (one row per client × month)."""
    calendario = _gerar_calendario(cfg)

    # Cartesian product clients × months
    df = clientes[["cliente_id"]].merge(calendario, how="cross")

    # Monthly volumes per client, dependent on client size
    # Merge to bring `dimensao_colaboradores` into each row
    df = df.merge(
        clientes[["cliente_id", "dimensao_colaboradores", "regime_contabilistico"]],
        on="cliente_id",
    )

    n_linhas = len(df)

    # nr_documentos: base proportional to size, with monthly variation (Poisson)
    # Clients on simplified accounting have fewer documents
    base_docs = 15 + 8 * df["dimensao_colaboradores"]
    fator_regime = np.where(df["regime_contabilistico"] == "simplificada", 0.5, 1.0)
    media_docs = base_docs * fator_regime
    df["nr_documentos"] = np.clip(
        rng.poisson(lam=media_docs, size=n_linhas), 0, 500
    )

    # nr_lancamentos: proportional to nr_documentos with noise (rule #1)
    df["nr_lancamentos"] = np.clip(
        np.round(df["nr_documentos"] * rng.normal(1.15, 0.10, n_linhas)).astype(int),
        0, 600,
    )

    # nr_colaboradores_processados: close to dimensao_colaboradores, ±1 (rule #1)
    delta = rng.integers(low=-1, high=2, size=n_linhas)
    df["nr_colaboradores_processados"] = np.clip(
        df["dimensao_colaboradores"] + delta, 0, 30
    )

    # nr_movimentos_bancarios: related to nr_documentos (but lower)
    df["nr_movimentos_bancarios"] = np.clip(
        rng.poisson(lam=df["nr_documentos"] * 0.4, size=n_linhas), 0, 200
    )

    # Apply fiscal obligations
    df = _aplicar_obrigacoes_fiscais(df, clientes)

    # tem_dmr depends on processed employees — compute before the target
    df["tem_dmr"] = (df["nr_colaboradores_processados"] > 0).astype(int)

    # Compute the target — see dedicated function
    df["horas_totais"] = _calcular_horas_totais(df, clientes, rng)

    # Reorder columns and remove auxiliary columns
    df = df.drop(columns=["dimensao_colaboradores", "regime_contabilistico"])
    colunas_finais = [
        "cliente_id", "ano", "mes",
        "nr_documentos", "nr_lancamentos",
        "nr_colaboradores_processados", "nr_movimentos_bancarios",
        "tem_iva_a_entregar", "tem_dmr", "tem_modelo_22",
        "tem_ies", "tem_irs_anual", "eh_pico_fiscal",
        "horas_totais",
    ]

    return df[colunas_finais].sort_values(["cliente_id", "ano", "mes"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Target calculation (rules #4 and #5 from section 2.3)
# ---------------------------------------------------------------------------

def _calcular_horas_totais(
    df: pd.DataFrame,
    clientes: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute `horas_totais` as a plausible combination of attributes + noise.

    The relationship is not deterministic (there is Gaussian noise), and the
    client's "document quality" — a latent variable — systematically modulates
    the effort (rule #5).
    """
    # Join the latent quality variable
    df = df.merge(
        clientes[["cliente_id", "_qualidade_docs_latente"]], on="cliente_id"
    )

    # Time components (in hours), additive:
    horas_lancamentos = 0.05 * df["nr_lancamentos"]
    horas_conciliacao = 0.03 * df["nr_movimentos_bancarios"]
    horas_salarios = 0.4 * df["nr_colaboradores_processados"]

    # One-off fiscal obligations add extra time
    horas_obrigacoes = (
        1.5 * df["tem_modelo_22"]
        + 2.0 * df["tem_ies"]
        + 0.8 * df["tem_irs_anual"]
        + 0.6 * df["tem_iva_a_entregar"]
        + 0.3 * df["tem_dmr"]
    )

    # Administrative base time per client
    horas_base = 1.0

    # Sum of components
    horas = horas_base + horas_lancamentos + horas_conciliacao + horas_salarios + horas_obrigacoes

    # Modulation by document quality (rule #5):
    # low quality (0.2) → factor ~1.5; high quality (0.95) → factor ~0.85
    fator_qualidade = 1.8 - df["_qualidade_docs_latente"]
    horas = horas * fator_qualidade

    # Gaussian noise (rule #4)
    ruido = rng.normal(loc=0, scale=0.6, size=len(df))
    horas = horas + ruido

    # Clip to valid range
    horas = np.clip(horas, 0, 40)
    return horas.round(2).to_numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    seed: int = 42,
    save_to: str | None = None,
    cfg: GenerationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the two DataFrames of the synthetic dataset: `clientes` and `mensal_cliente`.

    Parameters
    ----------
    seed : int
        Seed for the random generator (default 42 — reproducible).
    save_to : str | None
        Directory where to save the CSVs. If None, does not persist.
    cfg : GenerationConfig | None
        Alternative configuration. If None, uses defaults (60 clients × 36 months).

    Returns
    -------
    (clientes, mensal_cliente) : tuple[DataFrame, DataFrame]
        The latent column `_qualidade_docs_latente` is removed before returning.
    """
    if cfg is None:
        cfg = GenerationConfig(seed=seed)
    else:
        cfg = GenerationConfig(
            n_clientes=cfg.n_clientes,
            n_meses=cfg.n_meses,
            mes_fim=cfg.mes_fim,
            seed=seed,
        )

    rng = np.random.default_rng(cfg.seed)

    clientes = _gerar_clientes(cfg, rng)
    mensal = _gerar_mensal(cfg, clientes, rng)

    # Remove the latent column before returning/persisting
    clientes_publico = clientes.drop(columns=["_qualidade_docs_latente"])

    if save_to is not None:
        out_dir = Path(save_to)
        out_dir.mkdir(parents=True, exist_ok=True)
        clientes_publico.to_csv(out_dir / "clientes.csv", index=False)
        mensal.to_csv(out_dir / "mensal_cliente.csv", index=False)

    return clientes_publico, mensal


if __name__ == "__main__":
    clientes, mensal = generate_dataset(seed=42, save_to="data/")
    print(f"Clientes:        {len(clientes):>5} linhas")
    print(f"Mensal_cliente:  {len(mensal):>5} linhas")
    print()
    print("Amostra de clientes:")
    print(clientes.head().to_string(index=False))
    print()
    print("Amostra de mensal_cliente:")
    print(mensal.head().to_string(index=False))
    print()
    print("Estatísticas do target (horas_totais):")
    print(mensal["horas_totais"].describe().round(2).to_string())