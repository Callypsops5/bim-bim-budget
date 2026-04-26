"""
gold_transform.py — Couche Gold (Data Warehouse)
==================================================
Construction des tables finales à partir de la Silver.

Entrée (Silver) :
  annee | mois | categorie | libelle | montant | type_depense

Sorties (Gold) :
  transactions.parquet      → tous les postes propres et typés
  categories.parquet        → référentiel des catégories uniques
  savings.parquet           → économies mensuelles (revenus - dépenses)
  forecast_features.parquet → features mensuelles pour le modèle ML
"""

import pandas as pd
from pathlib import Path

from src.shared.config import SILVER_PATH, GOLD_PATH
from src.shared.utils import ensure_directory_exists


# ─── Chargement Silver ────────────────────────────────────────────────────────

def load_silver(filename: str) -> pd.DataFrame:
    """Charge un fichier Parquet depuis Silver."""
    path = Path(SILVER_PATH) / filename
    return pd.read_parquet(path)


# ─── Tables Gold ──────────────────────────────────────────────────────────────

def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table principale : tous les postes budgétaires propres.
    Colonnes : annee | mois | categorie | libelle | montant | type_depense
    """
    return df[["annee", "mois", "categorie", "libelle", "montant", "type_depense"]].copy()


def build_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Référentiel des catégories uniques avec leur type.
    Colonnes : categorie | type_depense
    """
    return (
        df[["categorie", "type_depense"]]
        .drop_duplicates()
        .sort_values("categorie")
        .reset_index(drop=True)
    )


def build_savings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Économies mensuelles : revenus - dépenses (épargne exclue des calculs).
    Colonnes : annee | mois | total_revenus | total_depenses | savings
    """
    # On exclut les lignes d'épargne (déjà un résultat, pas une transaction)
    df_ops = df[df["type_depense"] != "epargne"].copy()

    revenus = (
        df_ops[df_ops["type_depense"] == "revenu"]
        .groupby(["annee", "mois"])["montant"]
        .sum()
        .rename("total_revenus")
    )

    depenses = (
        df_ops[df_ops["type_depense"] == "depense"]
        .groupby(["annee", "mois"])["montant"]
        .sum()
        .rename("total_depenses")
    )

    df_savings = (
        pd.concat([revenus, depenses], axis=1)
        .fillna(0)
        .reset_index()
    )

    df_savings["savings"] = df_savings["total_revenus"] - df_savings["total_depenses"]

    return df_savings


def build_forecast_features(df_savings: pd.DataFrame) -> pd.DataFrame:
    """
    Features mensuelles pour le modèle prédictif.
    Ajoute une colonne 'saison' (0=hiver, 1=printemps, 2=été, 3=automne).

    Colonnes : annee | mois_num | total_revenus | total_depenses | savings | saison
    """
    # Mapping mois texte → numéro
    MOIS_MAP = {
        "janvier": 1, "février": 2, "mars": 3, "avril": 4,
        "mai": 5, "juin": 6, "juillet": 7, "août": 8,
        "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
    }

    df = df_savings.copy()
    df["mois_num"] = df["mois"].str.lower().map(MOIS_MAP)
    df["saison"] = ((df["mois_num"] - 1) // 3) % 4  # 0=hiver 1=print 2=été 3=auto

    return df[["annee", "mois", "mois_num", "total_revenus", "total_depenses", "savings", "saison"]]


# ─── Écriture Gold ────────────────────────────────────────────────────────────

def write_gold(df: pd.DataFrame, name: str) -> None:
    """Écrit une table Gold en Parquet."""
    ensure_directory_exists(GOLD_PATH)
    output_path = Path(GOLD_PATH) / f"{name}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  💾 {name}.parquet  ({len(df)} lignes)")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_gold_transform(silver_file: str) -> dict:
    """Pipeline Gold complet à partir d'un fichier Silver."""
    print(f"\n📥 Lecture Silver : {silver_file}")
    df_silver = load_silver(silver_file)

    print("📊 Construction transactions...")
    df_transactions = build_transactions(df_silver)
    write_gold(df_transactions, "transactions")

    print("🏷️  Construction categories...")
    df_categories = build_categories(df_silver)
    write_gold(df_categories, "categories")

    print("💰 Construction savings...")
    df_savings = build_savings(df_silver)
    write_gold(df_savings, "savings")

    print("🔮 Construction forecast_features...")
    df_features = build_forecast_features(df_savings)
    write_gold(df_features, "forecast_features")

    print("✅ Gold terminé")

    return {
        "transactions": df_transactions,
        "categories": df_categories,
        "savings": df_savings,
        "forecast_features": df_features,
    }


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # On peut passer plusieurs Silver et les tables Gold s'accumulent
    run_gold_transform("budget_2025_clean.parquet")
    run_gold_transform("budget_2026_clean.parquet")