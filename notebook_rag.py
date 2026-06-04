# -*- coding: utf-8 -*-
"""
Notebook RAG - Phase 7
Fetches top Kaggle notebook strategies and stores in ChromaDB
for the orchestrator to query
"""
import os, chromadb, requests
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
chroma = chromadb.PersistentClient(path=r'C:\Users\abhishek\memory')
rag = chroma.get_or_create_collection("notebook_rag")

# Top strategies from gold medal Kaggle notebooks (manually curated)
# In production this would scrape Kaggle notebooks via API
NOTEBOOK_STRATEGIES = [
    {
        "id": "stellar_tabpfn",
        "content": """TabPFN-3 Stacker for Stellar Classification:
        - Use TabPFN as base model - pretrained transformer for tabular data
        - Key features: color indices (u-g, g-r, r-i, i-z), redshift, spectral_type
        - Stacking with logistic regression meta-learner
        - Balanced accuracy ~0.970+ achievable
        - spectral_type and galaxy_population are extremely predictive
        - redshift alone separates QSO from GALAXY/STAR very well"""
    },
    {
        "id": "stellar_gpu_lr",
        "content": """GPU Logistic Regression Stacker for Stellar:
        - Train 100+ base models with different seeds and subsets
        - Use GPU-accelerated logistic regression for meta-learner
        - Feature engineering: all pairwise color differences
        - Log transform redshift, handle negative values carefully
        - Balanced accuracy 0.968+ with proper stacking"""
    },
    {
        "id": "stellar_features",
        "content": """Best features for Stellar Classification:
        - redshift is the single most important feature (separates QSO)
        - Color indices: u-g, g-r, r-i, i-z, u-r, g-z
        - spectral_type encodes stellar temperature class
        - galaxy_population distinguishes galaxy types
        - Don't use alpha/delta - sky coordinates add noise
        - Log transform redshift: log1p(abs(redshift))
        - Interaction: redshift * (g-r) powerful for QSO detection"""
    },
    {
        "id": "titanic_gold",
        "content": """Titanic gold medal strategies:
        - WomanOrChild feature is single strongest predictor
        - Title extraction from Name (Mr, Mrs, Miss, Master, Rare)
        - FamilySize + IsAlone
        - Cabin deck extraction (A-G + Unknown)
        - Ticket prefix grouping
        - Age imputation by Title+Pclass+Sex median
        - Simple LR often beats complex ensembles on LB due to overfitting
        - RepeatedStratifiedKFold with 10 splits, 5 repeats"""
    },
    {
        "id": "general_tabular",
        "content": """General tabular competition strategies:
        - Always check target leakage first
        - Use OOF predictions for stacking, never train set
        - Optuna with TPE sampler, 50-100 trials minimum
        - LightGBM + XGBoost + CatBoost ensemble almost always wins
        - Pseudo-labelling works best on large test sets (10k+)
        - Threshold tuning: optimize on OOF, apply to test
        - Feature importance: remove bottom 10% features
        - GroupKFold if data has temporal or group structure"""
    }
]

def populate_rag():
    print("[RAG] Populating notebook knowledge base...")
    for nb in NOTEBOOK_STRATEGIES:
        rag.upsert(
            documents=[nb["content"]],
            ids=[nb["id"]],
            metadatas=[{"source": "kaggle_notebook", "id": nb["id"]}]
        )
    print(f"[RAG] Stored {len(NOTEBOOK_STRATEGIES)} notebook strategies")

def query_rag(competition_desc, n=3):
    results = rag.query(query_texts=[competition_desc], n_results=n)
    return "\n\n---\n\n".join(results["documents"][0])

def get_strategy(competition, current_score, history):
    knowledge = query_rag(competition)
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=800,
        messages=[{"role": "user", "content": f"""You are an expert Kaggle agent.

Competition: {competition}
Current best score: {current_score}
History: {history}

Relevant notebook knowledge:
{knowledge}

Give ONE specific, actionable improvement to implement next. Be concrete — name exact features, models, or techniques. Max 5 sentences."""}]
    )
    return response.content[0].text

if __name__ == "__main__":
    populate_rag()
    print("\n[RAG] Testing query...")
    strategy = get_strategy(
        competition="Predicting Stellar Class - classify stars, galaxies, QSOs",
        current_score=0.95655,
        history="v1: 0.95553 basic LGBM, v2: 0.95655 added spectral+galaxy features"
    )
    print(f"\n[Strategy]\n{strategy}")
