"""
api.py — API FastAPI
======================
Expose le chatbot et les données financières pour le frontend React.

Endpoints :
  GET  /health                → statut de l'API
  GET  /alerts                → alertes proactives
  POST /chat                  → message au chatbot
  DELETE /chat                → réinitialiser la conversation
  GET  /dashboard             → données mensuelles pour les graphiques
  GET  /transactions          → toutes les transactions
  GET  /predictions           → prédictions d'épargne
  POST /refresh               → recharger les données après import Excel

Lancement :
  uvicorn src.finance.api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from pathlib import Path

from src.finance.chatbot_finance import ChatSession, load_financial_context, generate_alerts
from src.shared.config import MARTS_PATH, GOLD_PATH


# ─── App FastAPI ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Finance AI API",
    description="API backend pour le dashboard financier personnel",
    version="1.0.0",
)

# CORS — autorise React (Vite) en dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Session chatbot (une par serveur pour l'instant) ────────────────────────

session = ChatSession()


# ─── Schémas Pydantic ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    alerts:   list[str]
    n_mois:   int


# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_parquet_safe(path: Path) -> list[dict]:
    """Lit un Parquet et retourne une liste de dicts, ou [] si absent."""
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    # Convertir les booléens numpy en bool Python pour JSON
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(bool)
    return df.fillna(0).to_dict(orient="records")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Vérifie que l'API tourne."""
    return {"status": "ok", "model": "mistral", "version": "1.0.0"}


@app.get("/alerts")
def get_alerts():
    """Retourne les alertes proactives basées sur les données actuelles."""
    ctx    = load_financial_context()
    alerts = generate_alerts(ctx)
    return {"alerts": alerts}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Envoie un message au chatbot et retourne sa réponse.
    La session conserve l'historique de conversation.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Le message ne peut pas être vide.")

    result = session.chat(req.message)
    return ChatResponse(**result)


@app.delete("/chat")
def reset_chat():
    """Réinitialise l'historique de conversation."""
    session.reset()
    return {"status": "conversation réinitialisée"}


@app.get("/dashboard")
def get_dashboard():
    """
    Données mensuelles pour les graphiques React :
    revenus, dépenses, épargne par mois + dépenses par catégorie.
    """
    records = read_parquet_safe(Path(MARTS_PATH) / "mart_dashboard.parquet")

    # Grouper par mois pour la vue timeline
    df = pd.DataFrame(records) if records else pd.DataFrame()
    if df.empty:
        return {"monthly": [], "by_category": []}

    monthly = (
        df.groupby(["annee", "mois"])
        .agg(
            total_revenus=("total_revenus", "first"),
            total_depenses=("total_depenses", "first"),
            savings=("savings", "first"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    by_category = (
        df.groupby("categorie")["total_depenses_cat"]
        .sum()
        .reset_index()
        .rename(columns={"total_depenses_cat": "total"})
        .to_dict(orient="records")
    )

    return {"monthly": monthly, "by_category": by_category}


@app.get("/transactions")
def get_transactions(
    annee: int | None = None,
    mois:  str | None = None,
    type_depense: str | None = None,
):
    """
    Retourne toutes les transactions avec filtres optionnels.
    Query params : ?annee=2026&mois=Avril&type_depense=depense
    """
    records = read_parquet_safe(Path(MARTS_PATH) / "mart_ai.parquet")
    df = pd.DataFrame(records) if records else pd.DataFrame()

    if df.empty:
        return {"transactions": [], "total": 0}

    if annee:
        df = df[df["annee"] == annee]
    if mois:
        df = df[df["mois"].str.lower() == mois.lower()]
    if type_depense:
        df = df[df["type_depense"] == type_depense]

    return {
        "transactions": df.to_dict(orient="records"),
        "total":        len(df),
    }


@app.get("/predictions")
def get_predictions():
    """Retourne les prédictions d'épargne pour les prochains mois."""
    records = read_parquet_safe(Path(GOLD_PATH) / "predictions.parquet")
    return {"predictions": records}


@app.get("/stats")
def get_stats():
    """Retourne les statistiques globales du budget."""
    ctx = load_financial_context()
    return {
        "stats":        ctx.get("stats", {}),
        "dernier_mois": ctx.get("dernier_mois", {}),
        "top_depenses": ctx.get("top_depenses_moyennes", {}),
    }


@app.post("/refresh")
def refresh_data():
    """
    Recharge le contexte financier du chatbot après import de nouvelles données.
    À appeler depuis React après upload d'un nouveau fichier Excel.
    """
    session.refresh_context()
    return {"status": "contexte rechargé", "n_mois": session.ctx.get("stats", {}).get("n_mois", 0)}