"""
main.py — Point d'entrée du pipeline Finance
==============================================
Lance l'ensemble de la pipeline dans l'ordre :
  1. Bronze  : ingestion des fichiers Excel
  2. Silver  : nettoyage et normalisation
  3. Gold    : construction du Data Warehouse
  4. Marts   : construction des Data Marts
  5. Forecast: prédiction des économies futures

Usage :
  python main.py
"""

from src.finance.bronze_ingestion import run_bronze_ingestion
from src.finance.silver_cleaning  import run_silver_cleaning
from src.finance.gold_transform   import run_gold_transform
from src.finance.build_marts      import run_build_marts
from src.finance.forecasting      import run_forecasting


# ─── Fichiers source ──────────────────────────────────────────────────────────

SOURCES = [
    {
        "excel":  "inputs/Budget 2025.xlsx",
        "bronze": "budget_2025.parquet",
        "silver": "budget_2025_clean.parquet",
    },
    {
        "excel":  "inputs/Budget 2026.xlsx",
        "bronze": "budget_2026.parquet",
        "silver": "budget_2026_clean.parquet",
    },
]


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🚀 PIPELINE FINANCE — DÉMARRAGE")
    print("=" * 55)

    # ── 1. BRONZE ──────────────────────────────────────────
    print("\n╔══ BRONZE — Ingestion Excel ══╗")
    for s in SOURCES:
        run_bronze_ingestion(s["excel"], s["bronze"])

    # ── 2. SILVER ──────────────────────────────────────────
    print("\n╔══ SILVER — Nettoyage ══╗")
    for s in SOURCES:
        run_silver_cleaning(s["bronze"], s["silver"])

    # ── 3. GOLD ────────────────────────────────────────────
    print("\n╔══ GOLD — Data Warehouse ══╗")
    for s in SOURCES:
        run_gold_transform(s["silver"])

    # ── 4. MARTS ───────────────────────────────────────────
    print("\n╔══ MARTS — Data Marts ══╗")
    run_build_marts()

    # ── 5. FORECASTING ─────────────────────────────────────
    print("\n╔══ FORECASTING — Prédictions ══╗")
    run_forecasting(n_months=6)

    print("\n" + "=" * 55)
    print("  ✅ PIPELINE TERMINÉE AVEC SUCCÈS")
    print("=" * 55)


if __name__ == "__main__":
    main()