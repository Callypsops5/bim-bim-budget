"""
silver_cleaning.py — Couche Silver
=====================================
Nettoyage et normalisation des données Bronze.

Entrée (Bronze) :
  annee | mois | section | libelle | montant

Sortie (Silver) :
  annee | mois | categorie | libelle | montant | type_depense

Transformations :
  - section         → categorie (renommage + normalisation texte)
  - libelle         → nettoyage des espaces et casse
  - montant         → cast float, suppression des NaN
  - type_depense    → "revenu" | "depense" | "epargne" selon la section
"""

import pandas as pd
from pathlib import Path

from src.shared.config import BRONZE_PATH, SILVER_PATH
from src.shared.utils import ensure_directory_exists


# ─── Mapping section → type_depense ──────────────────────────────────────────

def get_type_depense(section: str) -> str:
    """
    Classifie chaque section en type :
      - revenu   : Revenus fixes, Revenus variables
      - epargne  : Épargne
      - depense  : tout le reste (Dépenses essentielles, optionnelles...)
    """
    low = section.strip().lower()
    if "revenu" in low:
        return "revenu"
    if "épargne" in low or "epargne" in low:
        return "epargne"
    return "depense"


# ─── Nettoyage ────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique toutes les transformations Silver sur le DataFrame Bronze.
    """
    # 1. Renommer section → categorie
    df = df.rename(columns={"section": "categorie"})

    # 2. Normaliser categorie : strip + lowercase
    df["categorie"] = df["categorie"].str.strip().str.lower()

    # 3. Normaliser libelle : strip + titre propre
    df["libelle"] = df["libelle"].str.strip()

    # 4. Cast montant en float (sécurité)
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce")

    # 5. Supprimer les lignes sans montant valide
    df = df.dropna(subset=["montant"])

    # 6. Ajouter type_depense
    df["type_depense"] = df["categorie"].apply(get_type_depense)

    # 7. Réordonner les colonnes
    df = df[["annee", "mois", "categorie", "libelle", "montant", "type_depense"]]

    return df


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie que la Silver est conforme avant écriture."""
    required = ["annee", "mois", "categorie", "libelle", "montant", "type_depense"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes Silver manquantes : {missing}")
    if df.empty:
        raise ValueError("Silver vide après nettoyage.")
    return df


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_bronze(filename: str) -> pd.DataFrame:
    """Charge un fichier Parquet depuis Bronze."""
    path = Path(BRONZE_PATH) / filename
    return pd.read_parquet(path)


def write_to_silver(df: pd.DataFrame, filename: str) -> None:
    """Écrit le DataFrame en Parquet dans Silver."""
    ensure_directory_exists(SILVER_PATH)
    output_path = Path(SILVER_PATH) / filename
    df.to_parquet(output_path, index=False)
    print(f"  💾 Écrit : {output_path}  ({len(df)} lignes)")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_silver_cleaning(bronze_file: str, silver_file: str) -> pd.DataFrame:
    """Pipeline Silver complet."""
    print(f"\n📥 Lecture Bronze : {bronze_file}")
    df_bronze = load_bronze(bronze_file)

    print("🧼 Nettoyage et normalisation...")
    df_clean = clean_data(df_bronze)

    print("🔍 Validation Silver...")
    df_clean = validate_silver(df_clean)

    print("📦 Écriture Silver...")
    write_to_silver(df_clean, silver_file)

    print(f"✅ Silver terminé — {len(df_clean)} lignes")
    return df_clean


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_silver_cleaning("budget_2025.parquet", "budget_2025_clean.parquet")
    run_silver_cleaning("budget_2026.parquet", "budget_2026_clean.parquet")