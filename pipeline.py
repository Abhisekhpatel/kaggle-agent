import os, subprocess, chromadb
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
chroma = chromadb.PersistentClient(path=r'C:\Users\abhishek\memory')
collection = chroma.get_or_create_collection("kaggle_agent")

def get_memory():
    results = collection.get(include=["documents","metadatas"])
    if not results["ids"]: return "No runs yet."
    runs = sorted(zip(results["ids"], results["metadatas"], results["documents"]),
                  key=lambda x: x[1].get("lb", 0), reverse=True)
    return "\n".join([f"{r[0]}: LB={r[1].get('lb','?')} | {r[2][:100]}" for r in runs])

def run_script(path):
    print(f"\n[Running] {path}")
    result = subprocess.run(['python', path], capture_output=True, text=True)
    print(result.stdout[-500:] if result.stdout else "")
    if result.stderr: print("[ERR]", result.stderr[-200:])
    return result.stdout

def get_lb_score():
    r = subprocess.run(['kaggle','competitions','submissions','-c','titanic'], capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if 'COMPLETE' in line:
            for p in line.split():
                try:
                    s = float(p)
                    if 0.5 < s < 1.0: return s
                except: pass
    return None

def save_run(name, cv, lb, notes):
    collection.upsert(
        documents=[f"Run {name}: CV={cv}, LB={lb}. Notes={notes}"],
        ids=[name],
        metadatas=[{"cv": float(cv), "lb": float(lb)}]
    )

def kaggle_submit(path, msg):
    r = subprocess.run(['kaggle','competitions','submit','-c','titanic','-f',path,'-m',msg], capture_output=True, text=True)
    print(r.stdout + r.stderr)

print("="*50)
print("AUTONOMOUS KAGGLE AGENT - FULL PIPELINE")
print("="*50)

# Step 1: Memory
memory = get_memory()
print(f"\n[Memory]\n{memory}\n")

# Step 2: Orchestrator decides next move
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=600,
    messages=[{"role": "user", "content": f"""Kaggle Titanic agent. History:
{memory}
Best LB: 0.77990. Target: 0.80+.
In 2 sentences: what is the single best next experiment and why?"""}]
)
strategy = response.content[0].text
print(f"[Orchestrator]\n{strategy}\n")

# Step 3: Run best model
run_script(r'C:\Users\abhishek\model_v4.py')

# Step 4: Submit
import glob
subs = sorted(glob.glob(r'C:\Users\abhishek\submissions\*.csv'), reverse=True)
if subs:
    latest = subs[0]
    print(f"\n[Submitting] {latest}")
    kaggle_submit(latest, f"auto-submit: {strategy[:50]}")

# Step 5: Get LB score
import time
time.sleep(10)
lb = get_lb_score()
print(f"\n[LB Score] {lb}")

# Step 6: Save to memory
if lb:
    save_run(f"auto-v{len(collection.get()['ids'])+1}", 0.8339, lb, strategy[:100])
    print(f"[Memory] Run saved. LB={lb}")

print("\n[Done] Full pipeline complete.")

