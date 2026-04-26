"""
priorities.py — Moteur de Priorités Financières
=================================================
Classe les dépenses par niveau de priorité et recommande
où couper en premier si le budget est déficitaire.

Niveaux de priorité :
  P1 — CRITIQUE   : ne peut pas être réduit (loyer, assurance santé...)
  P2 — IMPORTANT  : difficile à réduire (électricité, transport...)
  P3 — FLEXIBLE   : peut être optimisé (abonnements, courses...)
  P4 — OPTIONNEL  : peut être suspendu si nécessaire (dettes, extras...)

Sortie : dict structuré consommé par le chatbot et le dashboard.
"""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict

from src.shared.config import MARTS_PATH


# ─── Règles de classification ─────────────────────────────────────────────────
# Chaque règle est un tuple (mot-clé dans le libellé, niveau de priorité)
# Les mots-clés sont insensibles à la casse.

REGLES_PRIORITE: list[tuple[str, str]] = [
    # P1 — CRITIQUE
    ("loyer",          "P1"),
    ("maaf santé",     "P1"),
    ("assurance sant", "P1"),
    ("assurance scol", "P1"),

    # P2 — IMPORTANT
    ("electrict",      "P2"),
    ("octopus",        "P2"),
    ("transport",      "P2"),
    ("box internet",   "P2"),
    ("assurance hab",  "P2"),
    ("assurance lcl",  "P2"),

    # P3 — FLEXIBLE
    ("fnac",           "P3"),
    ("bouygues",       "P3"),
    ("produits entre", "P3"),

    # P4 — OPTIONNEL
    ("dette",          "P4"),
    ("ecole",          "P4"),
    ("école",          "P4"),
]

LIBELLES_PRIORITE = {
    "P1": "🔴 Critique — ne pas toucher",
    "P2": "🟠 Important — difficile à réduire",
    "P3": "🟡 Flexible — optimisable",
    "P4": "🟢 Optionnel — peut être suspendu",
}

# Ordre de coupe recommandé (du moins impactant au plus impactant)
ORDRE_COUPE = ["P4", "P3", "P2", "P1"]


# ─── Structure ────────────────────────────────────────────────────────────────

@dataclass
class PostePrioritaire:
    libelle:         str
    categorie:       str
    priorite:        str        # P1 / P2 / P3 / P4
    label_priorite:  str        # texte lisible
    montant_moyen:   float
    est_reductible:  bool       # True si P3 ou P4
    conseil:         str        # recommandation personnalisée


# ─── Chargement ───────────────────────────────────────────────────────────────

def load_transactions() -> pd.DataFrame:
    path = Path(MARTS_PATH) / "mart_ai.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Mart IA introuvable : {path}")
    return pd.read_parquet(path)


# ─── Classification ───────────────────────────────────────────────────────────

def classify_priority(libelle: str) -> str:
    """
    Retourne le niveau de priorité d'un poste selon les règles définies.
    Par défaut P3 si aucune règle ne correspond.
    """
    low = libelle.strip().lower()
    for keyword, priorite in REGLES_PRIORITE:
        if keyword.lower() in low:
            return priorite
    return "P3"  # flexible par défaut


def build_priority_list(df: pd.DataFrame) -> list[PostePrioritaire]:
    """
    Construit la liste des postes classés par priorité
    avec leur montant moyen et leur conseil.
    """
    depenses = df[df["type_depense"] == "depense"].copy()
    if depenses.empty:
        return []

    postes = []
    for libelle, grp in depenses.groupby("libelle"):
        montant_moyen = round(grp["montant"].mean(), 2)
        categorie     = grp.iloc[-1]["categorie"]
        priorite      = classify_priority(libelle)
        est_reductible = priorite in ("P3", "P4")

        # Conseil personnalisé selon priorité et montant
        if priorite == "P1":
            conseil = "Poste incontournable — à préserver en toutes circonstances."
        elif priorite == "P2":
            conseil = (
                f"Difficile à réduire, mais vérifiez si un meilleur tarif existe "
                f"(moy. {montant_moyen:.2f}€/mois)."
            )
        elif priorite == "P3":
            cible = round(montant_moyen * 0.85, 2)
            conseil = (
                f"Optimisable — objectif réaliste : {cible:.2f}€/mois "
                f"(économie : {montant_moyen - cible:.2f}€)."
            )
        else:  # P4
            conseil = (
                f"Peut être suspendu temporairement si nécessaire "
                f"({montant_moyen:.2f}€/mois libérés)."
            )

        postes.append(PostePrioritaire(
            libelle=libelle,
            categorie=categorie,
            priorite=priorite,
            label_priorite=LIBELLES_PRIORITE[priorite],
            montant_moyen=montant_moyen,
            est_reductible=est_reductible,
            conseil=conseil,
        ))

    # Trier : P1 → P4, puis par montant décroissant
    postes.sort(key=lambda x: (x.priorite, -x.montant_moyen))
    return postes


# ─── Plan de coupe si déficitaire ────────────────────────────────────────────

def build_cut_plan(
    postes: list[PostePrioritaire],
    deficit: float,
) -> dict:
    """
    Si le budget est déficitaire, génère un plan de coupe ordonné.
    Commence par P4, puis P3, puis P2 si nécessaire.
    Jamais P1.

    deficit : montant positif à récupérer (ex: 150.0 si savings = -150€)
    """
    if deficit <= 0:
        return {"needed": False, "message": "Budget équilibré — aucune coupe nécessaire. ✅"}

    coupes = []
    total_récupérable = 0.0

    for niveau in ORDRE_COUPE:
        if niveau == "P1":
            continue  # jamais toucher aux critiques

        candidats = [p for p in postes if p.priorite == niveau and p.montant_moyen > 0]
        for poste in sorted(candidats, key=lambda x: -x.montant_moyen):
            reduction = round(poste.montant_moyen * 0.30, 2)  # réduire de 30% max
            if niveau == "P4":
                reduction = poste.montant_moyen  # P4 : suspension totale possible

            coupes.append({
                "libelle":        poste.libelle,
                "priorite":       poste.priorite,
                "montant_moyen":  poste.montant_moyen,
                "reduction":      reduction,
                "action":         "Suspendre" if niveau == "P4" else "Réduire de 30%",
            })
            total_récupérable += reduction

            if total_récupérable >= deficit:
                break

        if total_récupérable >= deficit:
            break

    couvert = total_récupérable >= deficit

    return {
        "needed":             True,
        "deficit":            round(deficit, 2),
        "coupes":             coupes,
        "total_récupérable":  round(total_récupérable, 2),
        "deficit_couvert":    couvert,
        "message": (
            f"Pour combler le déficit de {deficit:.2f}€, "
            f"{'voici un plan qui couvre ' + str(round(total_récupérable, 2)) + '€' if couvert else 'même en optimisant tout le flexible, il manque encore ' + str(round(deficit - total_récupérable, 2)) + '€'}."
        ),
    }


# ─── Rapport complet ──────────────────────────────────────────────────────────

def run_priorities(savings: float | None = None) -> dict:
    """
    Point d'entrée principal.
    savings : solde du dernier mois (négatif = déficit). Si None, lu depuis les données.
    """
    print("\n📊 Chargement des transactions...")
    df = load_transactions()

    # Lire le savings si non fourni
    if savings is None:
        try:
            gold_path = Path(MARTS_PATH).parent / "gold" / "savings.parquet"
            df_sav = pd.read_parquet(gold_path)
            savings = float(df_sav.sort_values(["annee"]).iloc[-1]["savings"])
        except Exception:
            savings = 0.0

    print("🏷️  Classification des postes par priorité...")
    postes = build_priority_list(df)

    print("✂️  Génération du plan de coupe si nécessaire...")
    deficit  = abs(savings) if savings < 0 else 0.0
    cut_plan = build_cut_plan(postes, deficit)

    # Résumé par niveau
    par_niveau = {}
    for niveau in ["P1", "P2", "P3", "P4"]:
        groupe = [p for p in postes if p.priorite == niveau]
        par_niveau[niveau] = {
            "label":        LIBELLES_PRIORITE[niveau],
            "postes":       [asdict(p) for p in groupe],
            "total_mensuel": round(sum(p.montant_moyen for p in groupe), 2),
        }

    # Affichage terminal
    print(f"\n{'─'*60}")
    print(f"  RAPPORT PRIORITÉS  |  Savings dernier mois : {savings:+.2f}€")
    print(f"{'─'*60}")
    for niveau in ["P1", "P2", "P3", "P4"]:
        groupe = par_niveau[niveau]
        if groupe["postes"]:
            print(f"\n  {groupe['label']}  (total: {groupe['total_mensuel']:.2f}€/mois)")
            for p in groupe["postes"]:
                print(f"    • {p['libelle']:<40} {p['montant_moyen']:>8.2f}€")
                print(f"      → {p['conseil']}")

    if cut_plan["needed"]:
        print(f"\n  ✂️  PLAN DE COUPE (déficit : {cut_plan['deficit']:.2f}€)")
        print(f"  {cut_plan['message']}")
        for c in cut_plan["coupes"]:
            print(f"    • [{c['priorite']}] {c['libelle']:<35} {c['action']}  → -{c['reduction']:.2f}€")
    else:
        print(f"\n  ✅ {cut_plan['message']}")

    print(f"{'─'*60}")
    print("✅ Priorities Engine terminé")

    return {
        "savings_dernier_mois": savings,
        "postes":               [asdict(p) for p in postes],
        "par_niveau":           par_niveau,
        "cut_plan":             cut_plan,
    }


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_priorities()