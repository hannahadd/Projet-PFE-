import subprocess
import os
import shutil
import re
from pathlib import Path

# --- CHEMIN DE TON ENVIRONNEMENT VIRTUEL ---
PYTHON_VENV = "/home/pfe/.venv/bin/python"

# Chemins
BASE_DIR = Path("/home/pfe/Documents/PFE")
EXPAND_DIR = BASE_DIR / "main/expandmodule"
INTEREST_DIR = EXPAND_DIR / "interest"
TODO_DIR = EXPAND_DIR / "todo"
DB_URL = "postgresql://postgres:postgres@localhost:5432/pfe_news"

def lancer_generation_interet(topic):
    """Génère le fichier JSON pour le topic."""
    topic_clean = topic.lower()
    script_path = EXPAND_DIR / "generate_expansions.py"
    
    TODO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Utilisation de PYTHON_VENV ici
    cmd = [
        PYTHON_VENV, str(script_path),
        "--model", "qwen3.5:9b-q4_K_M",
        "--count", "10",
        "--min-words", "1",
        "--max-words", "2",
        "--out-dir", str(TODO_DIR),
        "--topic", topic
    ]
    print(f"--- Génération de {topic_clean} ---")
    subprocess.run(cmd, check=True)
    return TODO_DIR / f"{topic_clean}.json"

def traiter_pipeline(topic):
    topic_clean = topic.lower()
    
    try:
        # 1. Génération
        topic_file = lancer_generation_interet(topic)
        
        # 2. Déplacement
        INTEREST_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = INTEREST_DIR / f"{topic_clean}.json"
        
        if topic_file.exists():
            shutil.move(str(topic_file), str(dest_path))
        else:
            raise FileNotFoundError(f"Fichier {topic_file} non trouvé.")

        # 3. Retrieval (Utilisation de PYTHON_VENV)
        print(f"--- Retrieval : {topic_clean} ---")
        res_reco = subprocess.run([
            PYTHON_VENV, str(BASE_DIR / "main/news_reco.py"),
            "--db-url", DB_URL,
            "--interest", topic_clean,
            "--topk", "200"
        ], capture_output=True, text=True, check=True)
        
        retrieval_match = re.search(r"retrieval_run_id=(\d+)", res_reco.stdout)
        if not retrieval_match: raise ValueError("Retrieval ID non trouvé")
        retrieval_id = retrieval_match.group(1)
        
        # 4. Reranker (Utilisation de PYTHON_VENV)
        print(f"--- Reranker : ID {retrieval_id} ---")
        res_rerank = subprocess.run([
            PYTHON_VENV, str(BASE_DIR / "main/reranker.py"),
            "--db-url", DB_URL,
            "--retrieval-run-id", retrieval_id,
            "--topn", "10", "--hydrate"
        ], capture_output=True, text=True, check=True)
        
        rerank_match = re.search(r"rerank_run_id=(\d+)", rerank_res.stdout) if 'rerank_res' in locals() else re.search(r"rerank_run_id=(\d+)", res_rerank.stdout)
        if not rerank_match: raise ValueError("Rerank ID non trouvé")
        rerank_id = rerank_match.group(1)
        
        # 5. Writing (Utilisation de PYTHON_VENV)
        print(f"--- Writing : ID {rerank_id} ---")
        subprocess.run([
            PYTHON_VENV, str(BASE_DIR / "main/writing.py"),
            "--db-url", DB_URL,
            "--rerank-run-id", rerank_id,
            "--model", "qwen3.5:9b-q4_K_M",
            "--interest-batch-size", "10", "--top_n", "10"
        ], check=True)

        print(f"--- Succès : Pipeline terminé pour {topic_clean} ---")

    except subprocess.CalledProcessError as e:
        print(f"\n!!! ERREUR : {e.stderr} !!!")
    except Exception as e:
        print(f"\n!!! ERREUR GÉNÉRALE : {e} !!!")