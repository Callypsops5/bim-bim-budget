"""
forecasting.py — Modèle Prédictif des Économies
=================================================
Approche honnête adaptée à peu de données (8 mois).

Stratégie :
  - Moyenne pondérée (mois récents comptent plus)
  - Fourchette de confiance basée sur l'écart-type réel
  - Flag automatique des mois chargés (Mai + détection automatique)
  - Fiabilité affichée et croissante avec l'historique
  - Pas de fausse précision : on affiche une fourchette, pas un chiffre magique

Entrée  : data/silver/*.parquet  (tous les mois disponibles)
Sortie  : data/gold/predictions.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.shared.config import SILVER_PATH, GOLD_PATH
from src.shared.utils import ensure_directory_exists


# ─── Configuration ────────────────────────────────────────────────────────────

# Mois connus comme chargés (numéros) — Mai par défaut
MOIS_CHARGES = {5}

# Seuil : un mois est détecté automatiquement comme "chargé"
# si ses dépenses dépassent la moyenne de plus d'un écart-type
SEUIL_MOIS_CHARGE = 1.0

# Le mois le plus récent a ce poids, le plus ancien a un poids de 1
POIDS_MAX = 3.0

MOIS_MAP = {
    "janvier": 1, "février": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
}

MOIS_NOM = {v: k.capitalize() for k, v in MOIS_MAP.items()}


# ─── Chargement de tout l'historique ─────────────────────────────────────────

def load_all_silver() -> pd.DataFrame:
    """Charge et concatène tous les fichiers Silver disponibles."""
    silver_dir = Path(SILVER_PATH)
    files = sorted(silver_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier Silver trouvé dans {SILVER_PATH}")

    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  📂 {len(files)} fichier(s) Silver chargé(s) — {len(df)} lignes")
    return df


# ─── Calcul des savings historiques ──────────────────────────────────────────

def compute_historical_savings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les savings réels mois par mois à partir du Silver.
    savings = total_revenus - total_depenses  (épargne exclue)
    """
    df_ops = df[df["type_depense"] != "epargne"].copy()

    revenus = (
        df_ops[df_ops["type_depense"] == "revenu"]
        .groupby(["annee", "mois"])["montant"].sum()
        .rename("total_revenus")
    )
    depenses = (
        df_ops[df_ops["type_depense"] == "depense"]
        .groupby(["annee", "mois"])["montant"].sum()
        .rename("total_depenses")
    )

    hist = pd.concat([revenus, depenses], axis=1).fillna(0).reset_index()
    hist["savings"]  = hist["total_revenus"] - hist["total_depenses"]
    hist["mois_num"] = hist["mois"].str.lower().map(MOIS_MAP)
    hist = hist.sort_values(["annee", "mois_num"]).reset_index(drop=True)

    return hist


# ─── Détection automatique des mois chargés ──────────────────────────────────

def detect_mois_charges(hist: pd.DataFrame) -> set:
    """
    Détecte automatiquement les mois dont les dépenses dépassent
    la moyenne + SEUIL * écart-type. Ajoute les mois connus (MOIS_CHARGES).
    """
    moy   = hist["total_depenses"].mean()
    std   = hist["total_depenses"].std()
    seuil = moy + SEUIL_MOIS_CHARGE * std

    detectes = set(hist[hist["total_depenses"] > seuil]["mois_num"].tolist())
    tous = MOIS_CHARGES | detectes

    if tous:
        noms = [MOIS_NOM.get(m, str(m)) for m in sorted(tous)]
        print(f"  ⚠️  Mois chargés détectés : {', '.join(noms)}")

    return tous


# ─── Moyenne pondérée ─────────────────────────────────────────────────────────

def weighted_mean(values: pd.Series) -> float:
    """
    Moyenne pondérée : le dernier élément a le poids POIDS_MAX,
    le premier a le poids 1.
    """
    n = len(values)
    if n == 0:
        return 0.0
    weights = np.linspace(1, POIDS_MAX, n)
    return float(np.average(values, weights=weights))


# ─── Score de fiabilité ───────────────────────────────────────────────────────

def compute_fiabilite(n_mois: int) -> tuple[str, str]:
    """Retourne un label et une explication de fiabilité."""
    if n_mois < 6:
        return "🔴 Très faible", f"Seulement {n_mois} mois — fourchette très large"
    elif n_mois < 12:
        return "🟠 Faible", f"{n_mois} mois — fourchette large, tendance indicative"
    elif n_mois < 24:
        return "🟡 Modérée", f"{n_mois} mois — prédictions raisonnables"
    else:
        return "🟢 Bonne", f"{n_mois} mois — modèle bien calibré"


# ─── Génération des prédictions ───────────────────────────────────────────────

def generate_predictions(
    hist: pd.DataFrame,
    mois_charges: set,
    n_months: int = 6,
) -> pd.DataFrame:
    """
    Génère n_months prédictions à partir du dernier mois connu.

    Pour chaque mois futur :
      - prediction     : moyenne pondérée des savings historiques
      - borne_basse    : prediction - 1 écart-type (x1.5 si mois chargé)
      - borne_haute    : prediction + 1 écart-type (x1.5 si mois chargé)
      - est_mois_charge: True si le mois est dans mois_charges
    """
    n_mois   = len(hist)
    moy_pond = weighted_mean(hist["savings"])
    std_hist = hist["savings"].std()
    fiabilite_label, fiabilite_detail = compute_fiabilite(n_mois)

    last     = hist.iloc[-1]
    annee    = int(last["annee"])
    mois_num = int(last["mois_num"])

    records = []
    for _ in range(n_months):
        mois_num += 1
        if mois_num > 12:
            mois_num = 1
            annee += 1

        est_charge  = mois_num in mois_charges
        facteur     = 1.5 if est_charge else 1.0
        borne_basse = round(moy_pond - facteur * std_hist, 2)
        borne_haute = round(moy_pond + facteur * std_hist, 2)

        records.append({
            "annee":             annee,
            "mois_num":          mois_num,
            "mois":              MOIS_NOM.get(mois_num, str(mois_num)),
            "prediction":        round(moy_pond, 2),
            "borne_basse":       borne_basse,
            "borne_haute":       borne_haute,
            "est_mois_charge":   est_charge,
            "n_mois_historique": n_mois,
            "fiabilite":         fiabilite_label,
            "fiabilite_detail":  fiabilite_detail,
        })

    return pd.DataFrame(records)


# ─── Écriture ─────────────────────────────────────────────────────────────────

def write_predictions(df: pd.DataFrame) -> None:
    ensure_directory_exists(GOLD_PATH)
    output_path = Path(GOLD_PATH) / "predictions.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  💾 predictions.parquet  ({len(df)} lignes)")


# ─── Affichage lisible ────────────────────────────────────────────────────────

def print_predictions(df: pd.DataFrame) -> None:
    print()
    print("┌──────────────────────────────────────────────────────┐")
    print("│       PRÉDICTIONS ÉCONOMIES MENSUELLES               │")
    print("├────────────────┬────────────┬───────────┬────────────┤")
    print("│     Mois       │ Prédiction │  Min      │  Max       │")
    print("├────────────────┼────────────┼───────────┼────────────┤")

    for _, row in df.iterrows():
        flag  = "⚠️ " if row["est_mois_charge"] else "   "
        label = f"{flag}{row['mois']} {int(row['annee'])}"
        pred  = f"{row['prediction']:>7.0f} €"
        basse = f"{row['borne_basse']:>6.0f} €"
        haute = f"{row['borne_haute']:>6.0f} €"
        print(f"│ {label:<14} │  {pred}  │  {basse}  │  {haute}  │")

    fiabilite = df.iloc[0]["fiabilite"]
    detail    = df.iloc[0]["fiabilite_detail"]
    print("├──────────────────────────────────────────────────────┤")
    print(f"│ Fiabilité : {fiabilite:<41}│")
    print(f"│ {detail:<52}│")
    print("│ ⚠️  = mois chargé (charges annuelles détectées)       │")
    print("└──────────────────────────────────────────────────────┘")


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_forecasting(n_months: int = 6) -> pd.DataFrame:
    print("\n📥 Chargement de l'historique complet...")
    df_silver = load_all_silver()

    print("📊 Calcul des savings historiques...")
    hist = compute_historical_savings(df_silver)

    print("🔍 Détection des mois chargés...")
    mois_charges = detect_mois_charges(hist)

    print(f"🔮 Génération des {n_months} prochaines prédictions...")
    predictions = generate_predictions(hist, mois_charges, n_months)

    print("📦 Écriture dans Gold...")
    write_predictions(predictions)

    print_predictions(predictions)

    print("\n✅ Forecasting terminé")
    return predictions


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_forecasting(n_months=6)