"""
build_marts.py — Data Marts
=============================
Construction des marts optimisés par usage à partir des tables Gold.

Entrées (Gold) :
  transactions.parquet      → annee | mois | categorie | libelle | montant | type_depense
  savings.parquet           → annee | mois | total_revenus | total_depenses | savings
  forecast_features.parquet → annee | mois | mois_num | total_revenus | total_depenses | savings | saison

Sorties (Marts) :
  mart_dashboard.parquet    → vue mensuelle revenus / dépenses / épargne + dépenses par catégorie
  mart_ai.parquet           → toutes les transactions enrichies pour l'IA conseillère
  mart_forecasting.parquet  → features temporelles pour le modèle prédictif
"""

import pandas as pd
from pathlib import Path

from src.shared.config import GOLD_PATH, MARTS_PATH
from src.shared.utils import ensure_directory_exists


# ─── Chargement Gold ──────────────────────────────────────────────────────────

def load_gold(filename: str) -> pd.DataFrame:
    """Charge une table Gold depuis le dossier Gold."""
    path = Path(GOLD_PATH) / filename
    return pd.read_parquet(path)


# ─── Mart Dashboard ───────────────────────────────────────────────────────────

def build_mart_dashboard(
    df_transactions: pd.DataFrame,
    df_savings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vue mensuelle consolidée pour le dashboard :
      - total revenus, dépenses, épargne par mois
      - dépenses par catégorie et par mois

    Colonnes : annee | mois | categorie | total_depenses_cat |
               total_revenus | total_depenses | savings
    """
    # Dépenses par catégorie et par mois (on exclut revenus et épargne)
    df_cat = (
        df_transactions[df_transactions["type_depense"] == "depense"]
        .groupby(["annee", "mois", "categorie"])["montant"]
        .sum()
        .reset_index()
        .rename(columns={"montant": "total_depenses_cat"})
    )

    # Jointure avec savings pour avoir les totaux mensuels
    df_dashboard = df_cat.merge(df_savings, on=["annee", "mois"], how="left")

    return df_dashboard


# ─── Mart IA Conseillère ──────────────────────────────────────────────────────

def build_mart_ai(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Toutes les transactions enrichies pour l'IA conseillère.
    Ajoute un flag 'est_essentiel' basé sur la catégorie.

    Colonnes : annee | mois | categorie | libelle | montant | type_depense | est_essentiel
    """
    CATEGORIES_ESSENTIELLES = [
        "dépenses essentielles (besoins)",
        "revenus fixes",
        "revenus variables",
        "épargne",
    ]

    df_ai = df_transactions.copy()
    df_ai["est_essentiel"] = df_ai["categorie"].isin(CATEGORIES_ESSENTIELLES)

    return df_ai


# ─── Mart Forecasting ─────────────────────────────────────────────────────────

def build_mart_forecasting(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Features enrichies pour le modèle prédictif.
    Reprend forecast_features tel quel (saison déjà calculée dans Gold).

    Colonnes : annee | mois | mois_num | total_revenus | total_depenses | savings | saison
    """
    return df_features.copy()


# ─── Écriture Marts ───────────────────────────────────────────────────────────

def write_mart(df: pd.DataFrame, name: str) -> None:
    """Écrit un mart en Parquet."""
    ensure_directory_exists(MARTS_PATH)
    output_path = Path(MARTS_PATH) / f"{name}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  💾 {name}.parquet  ({len(df)} lignes)")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_build_marts() -> None:
    """Pipeline complet de construction des Data Marts."""
    print("\n📥 Lecture Gold...")
    df_transactions = load_gold("transactions.parquet")
    df_savings      = load_gold("savings.parquet")
    df_features     = load_gold("forecast_features.parquet")

    print("📊 Mart Dashboard...")
    mart_dashboard = build_mart_dashboard(df_transactions, df_savings)
    write_mart(mart_dashboard, "mart_dashboard")

    print("🤖 Mart IA...")
    mart_ai = build_mart_ai(df_transactions)
    write_mart(mart_ai, "mart_ai")

    print("🔮 Mart Forecasting...")
    mart_forecasting = build_mart_forecasting(df_features)
    write_mart(mart_forecasting, "mart_forecasting")

    print("✅ Data Marts terminés")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_build_marts()