"""
savings_engine.py — Moteur d'Économies
========================================
Analyse les dépenses et identifie les opportunités d'économies.

Fonctionnalités :
  1. Identifier les postes où on dépense trop (vs moyenne historique)
  2. Suggérer des montants cibles réalistes par poste
  3. Détecter les anomalies (dépenses inhabituellement élevées)

Sortie : dict structuré consommé par le chatbot et le dashboard.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

from src.shared.config import MARTS_PATH, GOLD_PATH
from src.shared.utils import ensure_directory_exists


# ─── Seuils de détection ──────────────────────────────────────────────────────

# Un poste est "trop élevé" si son dernier montant dépasse sa moyenne de X%
SEUIL_TROP_ELEVE_PCT = 0.20        # +20% vs moyenne

# Une dépense est une "anomalie" si elle dépasse moyenne + N * écart-type
SEUIL_ANOMALIE_STD   = 2.0

# La cible suggérée = médiane historique * ce facteur (légèrement en dessous)
FACTEUR_CIBLE        = 0.90


# ─── Structures de données ────────────────────────────────────────────────────

@dataclass
class OpportuniteEconomie:
    libelle:          str
    categorie:        str
    montant_actuel:   float   # dernier montant connu
    montant_moyen:    float   # moyenne historique
    montant_cible:    float   # objectif suggéré
    economie_possible: float  # montant_actuel - montant_cible
    raison:           str     # explication lisible


@dataclass
class Anomalie:
    libelle:        str
    categorie:      str
    montant:        float
    mois:           str
    annee:          int
    montant_moyen:  float
    ecart:          float     # en nombre d'écarts-types
    message:        str


# ─── Chargement ───────────────────────────────────────────────────────────────

def load_transactions() -> pd.DataFrame:
    """Charge toutes les transactions depuis le Mart IA."""
    path = Path(MARTS_PATH) / "mart_ai.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Mart IA introuvable : {path}")
    return pd.read_parquet(path)


# ─── 1. Postes trop élevés ────────────────────────────────────────────────────

def find_overspending(df: pd.DataFrame) -> list[OpportuniteEconomie]:
    """
    Identifie les postes dont le dernier montant dépasse
    significativement la moyenne historique.
    """
    depenses = df[df["type_depense"] == "depense"].copy()
    if depenses.empty:
        return []

    # Trier par date pour identifier le "dernier" mois
    MOIS_MAP = {
        "janvier": 1, "février": 2, "mars": 3, "avril": 4,
        "mai": 5, "juin": 6, "juillet": 7, "août": 8,
        "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
    }
    depenses["mois_num"] = depenses["mois"].str.lower().map(MOIS_MAP).fillna(0)
    depenses = depenses.sort_values(["annee", "mois_num"])

    opportunites = []

    for libelle, grp in depenses.groupby("libelle"):
        if len(grp) < 2:
            continue  # pas assez d'historique pour comparer

        montant_moyen  = grp["montant"].mean()
        montant_median = grp["montant"].median()
        montant_actuel = grp.iloc[-1]["montant"]
        categorie      = grp.iloc[-1]["categorie"]

        if montant_moyen == 0:
            continue

        # Le poste est trop élevé si dernier montant > moyenne + seuil
        if montant_actuel > montant_moyen * (1 + SEUIL_TROP_ELEVE_PCT):
            montant_cible    = round(montant_median * FACTEUR_CIBLE, 2)
            economie_possible = round(montant_actuel - montant_cible, 2)

            if economie_possible <= 0:
                continue

            opportunites.append(OpportuniteEconomie(
                libelle=libelle,
                categorie=categorie,
                montant_actuel=round(montant_actuel, 2),
                montant_moyen=round(montant_moyen, 2),
                montant_cible=montant_cible,
                economie_possible=economie_possible,
                raison=(
                    f"Dernier montant ({montant_actuel:.2f}€) supérieur "
                    f"de {((montant_actuel/montant_moyen - 1)*100):.0f}% "
                    f"à votre moyenne ({montant_moyen:.2f}€)."
                ),
            ))

    # Trier par économie possible décroissante
    opportunites.sort(key=lambda x: x.economie_possible, reverse=True)
    return opportunites


# ─── 2. Suggestions de cibles ─────────────────────────────────────────────────

def suggest_targets(df: pd.DataFrame) -> dict[str, float]:
    """
    Suggère un montant cible mensuel pour chaque poste récurrent
    basé sur la médiane historique * facteur d'optimisation.
    """
    depenses = df[df["type_depense"] == "depense"].copy()
    if depenses.empty:
        return {}

    targets = {}
    for libelle, grp in depenses.groupby("libelle"):
        if len(grp) < 2:
            continue
        median = grp["montant"].median()
        targets[libelle] = round(median * FACTEUR_CIBLE, 2)

    return dict(sorted(targets.items(), key=lambda x: x[1], reverse=True))


# ─── 3. Détection d'anomalies ─────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> list[Anomalie]:
    """
    Détecte les dépenses inhabituellement élevées sur un mois donné
    (montant > moyenne + N * écart-type sur l'historique du poste).
    """
    depenses = df[df["type_depense"] == "depense"].copy()
    if depenses.empty:
        return []

    anomalies = []

    for libelle, grp in depenses.groupby("libelle"):
        if len(grp) < 3:
            continue  # pas assez pour calculer un écart-type fiable

        moyenne = grp["montant"].mean()
        std     = grp["montant"].std()

        if std == 0:
            continue

        seuil = moyenne + SEUIL_ANOMALIE_STD * std

        for _, row in grp.iterrows():
            if row["montant"] > seuil:
                ecart = (row["montant"] - moyenne) / std
                anomalies.append(Anomalie(
                    libelle=libelle,
                    categorie=row["categorie"],
                    montant=round(row["montant"], 2),
                    mois=row["mois"],
                    annee=int(row["annee"]),
                    montant_moyen=round(moyenne, 2),
                    ecart=round(ecart, 2),
                    message=(
                        f"{libelle} : {row['montant']:.2f}€ en {row['mois']} {int(row['annee'])} "
                        f"(moyenne {moyenne:.2f}€, {ecart:.1f}x l'écart-type habituel)."
                    ),
                ))

    anomalies.sort(key=lambda x: x.ecart, reverse=True)
    return anomalies


# ─── 4. Résumé global ─────────────────────────────────────────────────────────

def compute_savings_summary(df: pd.DataFrame) -> dict:
    """
    Calcule le résumé global des opportunités d'économies.
    """
    opportunites = find_overspending(df)
    anomalies    = detect_anomalies(df)
    targets      = suggest_targets(df)

    total_economie_possible = round(
        sum(o.economie_possible for o in opportunites), 2
    )

    return {
        "opportunites":            [asdict(o) for o in opportunites],
        "anomalies":               [asdict(a) for a in anomalies],
        "cibles_suggérées":        targets,
        "total_economie_possible": total_economie_possible,
        "n_postes_trop_eleves":    len(opportunites),
        "n_anomalies":             len(anomalies),
        "message_resume": (
            f"{len(opportunites)} poste(s) à optimiser — "
            f"économie potentielle : {total_economie_possible:.2f}€/mois. "
            f"{len(anomalies)} anomalie(s) détectée(s)."
            if opportunites or anomalies
            else "Aucune anomalie détectée — budget stable. 👍"
        ),
    }


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_savings_engine() -> dict:
    """Point d'entrée principal — retourne le rapport complet."""
    print("\n📊 Chargement des transactions...")
    df = load_transactions()

    print("🔍 Analyse des opportunités d'économies...")
    summary = compute_savings_summary(df)

    print(f"\n{'─'*55}")
    print(f"  RAPPORT SAVINGS ENGINE")
    print(f"{'─'*55}")
    print(f"  {summary['message_resume']}")
    print()

    if summary["opportunites"]:
        print("  💡 Postes à optimiser :")
        for o in summary["opportunites"]:
            print(f"    • {o['libelle']:<40} "
                  f"actuel={o['montant_actuel']:>7.2f}€  "
                  f"cible={o['montant_cible']:>7.2f}€  "
                  f"gain={o['economie_possible']:>6.2f}€")

    if summary["anomalies"]:
        print()
        print("  ⚠️  Anomalies détectées :")
        for a in summary["anomalies"]:
            print(f"    • {a['message']}")

    print(f"{'─'*55}")
    print("✅ Savings Engine terminé")

    return summary


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_savings_engine()