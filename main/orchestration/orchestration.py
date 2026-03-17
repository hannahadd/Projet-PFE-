import subprocess
import os
import shutil
import re
import time
import torch
from pathlib import Path
import sys

# --- CONFIGURATION ---
PYTHON_VENV = "/home/pfe/Documents/PFE/.venv312/bin/python"
BASE_DIR = Path("/home/pfe/Documents/PFE")
EXPAND_DIR = BASE_DIR / "main/expandmodule"
INTEREST_DIR = EXPAND_DIR / "interest"
TODO_DIR = EXPAND_DIR / "todo"
DB_URL = "postgresql://postgres:postgres@localhost:5432/pfe_news"

def clean_gpu_memory(aggressive=False):
    """Nettoie la VRAM et tue Ollama si nécessaire pour le Reranker."""
    print("\n🧹 Nettoyage de la mémoire GPU...")
    if aggressive:
        print("🔴 Libération forcée de la VRAM (Arrêt Ollama)...")
        subprocess.run("ollama stop qwen3.5:9b-q4_K_M", shell=True, capture_output=True)
        subprocess.run("pkill -9 ollama-runner", shell=True, capture_output=True)
        time.sleep(3) 

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(1)

def run_command(cmd_list, capture=False):
    """Exécute une commande et gère les erreurs."""
    print(f"\n🚀 Exécution : {' '.join(cmd_list)}")
    try:
        if capture:
            result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
            return result
        else:
            subprocess.run(cmd_list, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERREUR : {e.cmd}")
        if capture: print(f"Sortie erreur : {e.stderr}")
        raise

def traiter_pipeline_complet(topic_raw):
    # 1. Normalisation immédiate pour éviter les doublons de casse
    topic = topic_raw.lower().strip()
    topic_filename = topic.replace(" ", "_")
    
    try:
        print(f"\n=============================================")
        print(f" DÉBUT DU PIPELINE POUR : {topic.upper()}")
        print(f"=============================================\n")

        # --- ÉTAPE 2 : Expansion ---
        print("\n--- 2. Expansion (Génération JSON) ---")
        TODO_DIR.mkdir(parents=True, exist_ok=True)
        run_command([
            PYTHON_VENV, str(EXPAND_DIR / "generate_expansions.py"),
            "--model", "qwen3.5:9b-q4_K_M",
            "--count", "10", "--min-words", "1", "--max-words", "2",
            "--out-dir", str(TODO_DIR), "--topic", topic
        ])
        
        # Déplacement immédiat du fichier
        topic_file = TODO_DIR / f"{topic_filename}.json"
        INTEREST_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = INTEREST_DIR / f"{topic_filename}.json"
        if topic_file.exists():
            shutil.move(str(topic_file), str(dest_path))
            print(f"✅ Fichier déplacé : {dest_path}")

        # --- ÉTAPE 3 : Retrieval ---
        clean_gpu_memory(aggressive=True) 
        print("\n--- 3. Retrieval ---")
        res_reco = run_command([
            PYTHON_VENV, str(BASE_DIR / "main/news_reco.py"),
            "--db-url", DB_URL, "--topk", "1300",
            "--max-expansions-per-interest", "10",
            "--dense-per-anchor", "600", "--dense-per-expansion", "250",
            "--bm25-title-k", "150", "--bm25-body-k", "300",
            "--rrf-k", "60", "--candidate-cap", "2500",
            "--interest", topic
        ], capture=True)
        
        # Regex robuste pour l'ID Retrieval
        rid_match = re.search(r"(?:ID Retrieval|retrieval_run_id)\s*[:=]\s*(\d+)", res_reco.stdout)
        if not rid_match: raise ValueError("Retrieval ID non trouvé.")
        rid = rid_match.group(1)
        print(f"-> ID Retrieval : {rid}")

        # --- ÉTAPE 4 : Déduplication ---
        print("\n--- 4. Déduplication ---")
        res_dedup = run_command([
            PYTHON_VENV, str(BASE_DIR / "main/depuplication.py"),
            "--db-url", DB_URL, "--interest", topic
        ], capture=True)
        
        # Récupération de l'ID spécifique à la déduplication
        did_match = re.search(r"dedup_run_id=(\d+)", res_dedup.stdout)
        if not did_match: raise ValueError("Dedup ID non trouvé.")
        did = did_match.group(1)
        print(f"-> ID Déduplication : {did}")

        # --- ÉTAPE 5 : Reranker ---
        clean_gpu_memory(aggressive=True) 
        print(f"\n--- 5. Reranker (Batch-size 1) ---")
        res_rerank = run_command([
            PYTHON_VENV, str(BASE_DIR / "main/reranker.py"),
            "--db-url", DB_URL, "--table", "dedup_hits",
            "--run-id", did, # On utilise l'ID de déduplication ici !
            "--topn", "10", "--diversity-scan-k", "80",
            "--pairwise-threshold", "0.62", "--smaxtreshold", "0.70",
            "--smintreshold", "0.35", "--pairwise-batch-size", "1",
            "--hydrate"
        ], capture=True)
        
        rrid_match = re.search(r"rerank_run_id=(\d+)", res_rerank.stdout)
        if not rrid_match: raise ValueError("Rerank ID non trouvé.")
        rrid = rrid_match.group(1)
        print(f"-> ID Rerank : {rrid}")

        # --- ÉTAPE 6 : Writing ---
        clean_gpu_memory() 
        print(f"\n--- 6. Writing ---")
        run_command([
            PYTHON_VENV, str(BASE_DIR / "main/writing.py"),
            "--db-url", DB_URL, "--model", "qwen3.5:9b-q4_K_M",
            "--interest-batch-size", "10", "--top_n", "10",
            "--rerank-run-id", rrid
        ])

        print(f"\n✅ PIPELINE TERMINÉ AVEC SUCCÈS POUR : {topic.upper()}")

    except Exception as e:
        print(f"\n!!! ERREUR FATALE : {e} !!!")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "video games"
    traiter_pipeline_complet(target)