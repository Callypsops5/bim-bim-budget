"""
chatbot_finance.py — IA Conseillère Financière
================================================
Chatbot financier personnel alimenté par Ollama (Mistral).

Fonctionnalités :
  - Répond aux questions sur le budget en FR/EN
  - Analyse les dépenses et donne des conseils
  - Alertes proactives sur les dérives de budget
  - Mémoire de conversation par session
  - Contexte financier injecté automatiquement depuis les Marts

Architecture :
  chatbot_finance.py  ← logique LLM + contexte
  api.py              ← endpoints FastAPI consommés par React
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

import requests

from src.shared.config import MARTS_PATH, GOLD_PATH


# ─── Configuration Ollama ─────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"

# Nombre maximum de messages gardés en mémoire par session
MAX_HISTORY = 20


# ─── Chargement du contexte financier ────────────────────────────────────────

def load_financial_context() -> dict:
    """
    Charge et résume les données financières depuis les Marts et Gold.
    Ce contexte est injecté dans le system prompt du LLM.
    """
    ctx = {}

    try:
        # Dashboard : vue mensuelle
        df_dash = pd.read_parquet(Path(MARTS_PATH) / "mart_dashboard.parquet")
        monthly = (
            df_dash.groupby(["annee", "mois"])
            .agg(
                total_revenus=("total_revenus", "first"),
                total_depenses=("total_depenses", "first"),
                savings=("savings", "first"),
            )
            .reset_index()
        )
        ctx["historique_mensuel"] = monthly.to_dict(orient="records")

        # Derniers mois
        last = monthly.iloc[-1] if not monthly.empty else None
        if last is not None:
            ctx["dernier_mois"] = {
                "mois":           last["mois"],
                "annee":          int(last["annee"]),
                "total_revenus":  round(last["total_revenus"], 2),
                "total_depenses": round(last["total_depenses"], 2),
                "savings":        round(last["savings"], 2),
            }

        # Top dépenses toutes catégories confondues
        df_ai = pd.read_parquet(Path(MARTS_PATH) / "mart_ai.parquet")
        top_depenses = (
            df_ai[df_ai["type_depense"] == "depense"]
            .groupby("libelle")["montant"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .round(2)
            .to_dict()
        )
        ctx["top_depenses_moyennes"] = top_depenses

        # Dépenses par catégorie
        dep_cat = (
            df_ai[df_ai["type_depense"] == "depense"]
            .groupby("categorie")["montant"]
            .sum()
            .round(2)
            .to_dict()
        )
        ctx["depenses_par_categorie"] = dep_cat

        # Statistiques globales
        savings_list = monthly["savings"].tolist()
        ctx["stats"] = {
            "n_mois":         len(monthly),
            "savings_moyen":  round(sum(savings_list) / len(savings_list), 2) if savings_list else 0,
            "savings_min":    round(min(savings_list), 2) if savings_list else 0,
            "savings_max":    round(max(savings_list), 2) if savings_list else 0,
            "mois_negatifs":  int(sum(1 for s in savings_list if s < 0)),
        }

        # Prédictions
        pred_path = Path(GOLD_PATH) / "predictions.parquet"
        if pred_path.exists():
            df_pred = pd.read_parquet(pred_path)
            ctx["predictions"] = df_pred[
                ["mois", "annee", "prediction", "borne_basse", "borne_haute",
                 "est_mois_charge", "fiabilite"]
            ].to_dict(orient="records")

    except Exception as e:
        ctx["erreur_chargement"] = str(e)

    return ctx


# ─── Alertes proactives ───────────────────────────────────────────────────────

def generate_alerts(ctx: dict) -> list[str]:
    """
    Génère des alertes proactives basées sur les données financières.
    Retourne une liste de messages d'alerte.
    """
    alerts = []
    stats = ctx.get("stats", {})
    dernier = ctx.get("dernier_mois", {})

    # Alerte mois négatif
    if dernier.get("savings", 0) < 0:
        alerts.append(
            f"⚠️ Ton dernier mois ({dernier.get('mois')}) est déficitaire : "
            f"{dernier.get('savings')} € de solde."
        )

    # Alerte tendance négative
    if stats.get("mois_negatifs", 0) > 0:
        alerts.append(
            f"📉 Tu as eu {stats['mois_negatifs']} mois déficitaire(s) sur "
            f"{stats['n_mois']} mois d'historique."
        )

    # Alerte épargne faible
    if 0 < stats.get("savings_moyen", 0) < 100:
        alerts.append(
            f"💡 Ton épargne moyenne est de {stats['savings_moyen']} €/mois. "
            f"C'est serré — quelques postes à optimiser ?"
        )

    # Alerte mois chargé à venir
    preds = ctx.get("predictions", [])
    for p in preds:
        if p.get("est_mois_charge") and p.get("borne_basse", 0) < 0:
            alerts.append(
                f"📅 {p['mois']} {p['annee']} est un mois chargé — "
                f"solde prévu entre {p['borne_basse']} € et {p['borne_haute']} €."
            )

    return alerts


# ─── System prompt ────────────────────────────────────────────────────────────

def build_system_prompt(ctx: dict, alerts: list[str]) -> str:
    """
    Construit le system prompt injecté au LLM avec le contexte financier complet.
    """
    alerts_str = "\n".join(f"- {a}" for a in alerts) if alerts else "- Aucune alerte pour le moment."

    prompt = f"""Tu es un conseiller financier personnel intelligent, bienveillant et honnête.
Tu aides l'utilisateur à comprendre et améliorer sa situation financière personnelle.

RÈGLES IMPORTANTES :
- Tu réponds en français ou en anglais selon la langue de l'utilisateur.
- Tu es direct, concis et factuel. Pas de blabla inutile.
- Tu bases TOUJOURS tes réponses sur les données financières réelles fournies ci-dessous.
- Si tu ne sais pas ou si les données sont insuffisantes, tu le dis honnêtement.
- Tu ne inventes jamais de chiffres.
- Tu es encourageant mais réaliste.

DONNÉES FINANCIÈRES RÉELLES (mises à jour automatiquement) :
{json.dumps(ctx, ensure_ascii=False, indent=2)}

ALERTES PROACTIVES DÉTECTÉES :
{alerts_str}

DATE ACTUELLE : {datetime.now().strftime("%B %Y")}

Quand l'utilisateur pose une question sur son budget, utilise ces données pour répondre précisément.
Quand il demande des conseils, base-toi sur ses vraies dépenses pour être pertinent.
Si des alertes sont présentes, mentionne-les naturellement dans la conversation quand c'est pertinent.
"""
    return prompt


# ─── Appel Ollama ─────────────────────────────────────────────────────────────

def call_ollama(
    messages: list[dict],
    system_prompt: str,
) -> str:
    """
    Envoie la conversation à Ollama et retourne la réponse du modèle.
    """
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream":   False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    except requests.exceptions.ConnectionError:
        return (
            "❌ Ollama n'est pas accessible. "
            "Vérifie qu'il tourne avec `ollama serve` et que Mistral est installé (`ollama pull mistral`)."
        )
    except requests.exceptions.Timeout:
        return "⏳ Le modèle met trop de temps à répondre. Réessaie dans quelques secondes."
    except Exception as e:
        return f"❌ Erreur inattendue : {str(e)}"


# ─── Gestionnaire de session ──────────────────────────────────────────────────

class ChatSession:
    """
    Gère une session de conversation avec mémoire.
    Une session = un utilisateur connecté (une instance par session React).
    """

    def __init__(self):
        self.history: list[dict] = []
        self.ctx     = load_financial_context()
        self.alerts  = generate_alerts(self.ctx)
        self.system  = build_system_prompt(self.ctx, self.alerts)

    def chat(self, user_message: str) -> dict:
        """
        Envoie un message et retourne la réponse + les alertes actives.
        """
        # Ajouter le message utilisateur à l'historique
        self.history.append({"role": "user", "content": user_message})

        # Limiter la mémoire
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        # Appel au LLM
        response = call_ollama(self.history, self.system)

        # Ajouter la réponse à l'historique
        self.history.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "alerts":   self.alerts,
            "n_mois":   self.ctx.get("stats", {}).get("n_mois", 0),
        }

    def refresh_context(self):
        """Recharge les données financières (après ajout de nouvelles données)."""
        self.ctx    = load_financial_context()
        self.alerts = generate_alerts(self.ctx)
        self.system = build_system_prompt(self.ctx, self.alerts)

    def reset(self):
        """Réinitialise l'historique de conversation."""
        self.history = []


# ─── Point d'entrée CLI (test rapide) ────────────────────────────────────────

if __name__ == "__main__":
    print("💬 Chatbot Finance — mode terminal (Ctrl+C pour quitter)")
    print("─" * 50)

    session = ChatSession()

    if session.alerts:
        print("\n🔔 Alertes actives :")
        for a in session.alerts:
            print(f"  {a}")
        print()

    while True:
        try:
            user_input = input("Vous : ").strip()
            if not user_input:
                continue
            result = session.chat(user_input)
            print(f"\nIA : {result['response']}\n")
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break