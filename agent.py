# -*- coding: utf-8 -*-
"""
AUTONOMOUS KAGGLE AGENT - Single command entry point
Runs full pipeline: RAG strategy -> train -> validate -> submit -> log -> loop
Usage: python agent.py --competition playground-series-s6e6 --loops 3
"""
import os, sys, subprocess, glob, time, argparse, chromadb
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
chroma = chromadb.PersistentClient(path=r'C:\Users\abhishek\memory')
runs = chroma.get_or_create_collection("kaggle_agent")

def log(msg): print(f"\n{'='*50}\n{msg}\n{'='*50}")

def get_memory():
    r = runs.get(include=["documents","metadatas"])
    if not r["ids"]: return "No runs yet."
    sorted_runs = sorted(zip(r["ids"], r["metadatas"], r["documents"]),
                         key=lambda x: x[1].get("lb", 0), reverse=True)
    return "\n".join([f"{x[0]}: LB={x[1].get('lb','?')} | {x[2][:80]}" for x in sorted_runs])

def save_run(name, cv, lb, notes):
    runs.upsert(
        documents=[f"{name}: CV={cv:.4f}, LB={lb}. {notes}"],
        ids=[name],
        metadatas=[{"cv": float(cv), "lb": float(lb) if lb else 0.0}]
    )

def get_rag_strategy(competition, score, history):
    try:
        rag = chroma.get_or_create_collection("notebook_rag")
        results = rag.query(query_texts=[competition], n_results=2)
        knowledge = "\n".join(results["documents"][0]) if results["documents"] else ""
    except: knowledge = ""

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=600,
        messages=[{"role": "user", "content": f"""Kaggle agent on: {competition}
Current best: {score}
History: {history}
Notebook knowledge: {knowledge}

In 3 bullet points: what are the top improvements to try next?"""}]
    )
    return response.content[0].text

def run_script(path):
    result = subprocess.run(['python', path], capture_output=True, text=True,
                           cwd=r'C:\Users\abhishek')
    return result.stdout + result.stderr

def get_latest_submission(sub_dir):
    files = sorted(glob.glob(f'{sub_dir}\\*.csv'), reverse=True)
    return files[0] if files else None

def submit(competition, filepath, message):
    r = subprocess.run([
        'kaggle', 'competitions', 'submit',
        '-c', competition, '-f', filepath, '-m', message
    ], capture_output=True, text=True)
    return r.stdout + r.stderr

def get_lb_score(competition):
    time.sleep(15)
    r = subprocess.run(['kaggle', 'competitions', 'submissions', '-c', competition],
                       capture_output=True, text=True)
    for line in r.stdout.split('\n'):
        if 'COMPLETE' in line:
            for p in line.split():
                try:
                    s = float(p)
                    if 0.5 < s < 1.0: return s
                except: pass
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--competition', default='playground-series-s6e6')
    parser.add_argument('--script', default=r'C:\Users\abhishek\stellar_v2.py')
    parser.add_argument('--sub_dir', default=r'C:\Users\abhishek\submissions\stellar')
    parser.add_argument('--loops', type=int, default=1)
    args = parser.parse_args()

    log(f"AUTONOMOUS KAGGLE AGENT\nCompetition: {args.competition}")

    for loop in range(args.loops):
        print(f"\n[Loop {loop+1}/{args.loops}]")

        # Step 1: Get strategy from RAG + memory
        memory = get_memory()
        best_lb = max([m.get("lb", 0) for m in runs.get(include=["metadatas"])["metadatas"]] or [0])
        strategy = get_rag_strategy(args.competition, best_lb, memory)
        print(f"\n[Strategy]\n{strategy}")

        # Step 2: Train
        log("TRAINING")
        output = run_script(args.script)
        print(output[-1000:])

        # Extract CV score from output
        cv_score = 0.0
        for line in output.split('\n'):
            if 'OOF:' in line or 'CV:' in line:
                for p in line.split():
                    try:
                        s = float(p)
                        if 0.5 < s < 1.0: cv_score = s
                    except: pass

        # Step 3: Submit
        sub_file = get_latest_submission(args.sub_dir)
        if not sub_file:
            print("[Error] No submission file found")
            continue

        log(f"SUBMITTING\n{sub_file}")
        result = submit(args.competition, sub_file, f"auto-loop-{loop+1} cv={cv_score:.4f}")
        print(result)

        # Step 4: Get LB score
        lb = get_lb_score(args.competition)
        print(f"\n[LB Score] {lb}")

        # Step 5: Save to memory
        run_id = f"auto-loop-{loop+1}-{int(time.time())}"
        save_run(run_id, cv_score, lb or 0.0, strategy[:100])
        print(f"[Memory] Saved {run_id}")

        if loop < args.loops - 1:
            print("\n[Waiting 60s before next loop...]")
            time.sleep(60)

    log("PIPELINE COMPLETE")
    print(f"\nFinal memory:\n{get_memory()}")

if __name__ == "__main__":
    main()
