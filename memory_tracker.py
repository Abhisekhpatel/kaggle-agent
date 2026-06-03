import os, json, chromadb, mlflow
from datetime import datetime

LOGS_DIR = r'C:\Users\abhishek\logs'
os.makedirs(LOGS_DIR, exist_ok=True)

# ChromaDB vector memory
chroma = chromadb.PersistentClient(path=r'C:\Users\abhishek\memory')
collection = chroma.get_or_create_collection("kaggle_agent")

def save_experiment(run_name, cv_score, lb_score, features, notes):
    # Log to MLflow
    mlflow.set_experiment("titanic-agent")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("cv_score", cv_score)
        mlflow.log_metric("lb_score", lb_score)
        mlflow.log_param("features", str(features))
        mlflow.log_param("notes", notes)
    
    # Save to ChromaDB memory
    collection.add(
        documents=[f"Run {run_name}: CV={cv_score}, LB={lb_score}. Features={features}. Notes={notes}"],
        ids=[run_name],
        metadatas=[{"cv": cv_score, "lb": lb_score, "timestamp": str(datetime.now())}]
    )
    print(f"[Memory] Saved {run_name}")

def get_best_runs(n=5):
    results = collection.get(include=["documents","metadatas"])
    if not results["ids"]:
        return "No runs yet."
    runs = sorted(zip(results["ids"], results["metadatas"], results["documents"]), 
                  key=lambda x: x[1].get("lb", 0), reverse=True)
    return runs[:n]

def recall_memory(query):
    results = collection.query(query_texts=[query], n_results=3)
    return results["documents"]

# Log all existing runs
save_experiment("v1", cv_score=0.8384, lb_score=0.76076, features=["Pclass","Sex","Age","Fare","Title","FamilySize"], notes="First ensemble XGB+LGBM+LR+RF")
save_experiment("v2", cv_score=0.8320, lb_score=0.76315, features=["Pclass","Sex","Age","Fare","FarePerPerson","FamilySize","IsAlone","Title","Embarked"], notes="RepeatedStratifiedKFold, stronger regularisation")
save_experiment("v3", cv_score=0.8309, lb_score=0.77511, features=["WomanOrChild","AgeClass","PclassSex","IsChild"], notes="WomanOrChild + interaction features, best LB so far")

print("\n=== Best runs by LB score ===")
for run_id, meta, doc in get_best_runs():
    print(f"{run_id}: LB={meta['lb']} | {doc[:80]}")

print("\n=== Memory recall: what worked best? ===")
print(recall_memory("best features for Titanic survival prediction"))
