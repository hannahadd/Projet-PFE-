# PFE News Pipeline

Pipeline de recommandation d’actualités, combinant retrieval hybride (dense + lexical), déduplication, reranking LLM et génération de résumés/titres, avec traçabilité complète en PostgreSQL.

## Table des matières
- [Vue d’ensemble](#vue-densemble)
- [Architecture de la solution](#architecture-de-la-solution)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Exécution complète du pipeline](#exécution-complète-du-pipeline)
- [Commandes d’évaluation et debug](#commandes-dévaluation-et-debug)
- [Visualisation des résultats](#visualisation-des-résultats)
- [Paramètres importants](#paramètres-importants)
- [Dépannage](#dépannage)
- [Annexe — Détail Retrieval & Reranking](#annexe--détail-retrieval--reranking)

---

## Vue d’ensemble

Le projet prend des articles de news bruts (CCNews, GDELT), les normalise et les stocke dans PostgreSQL, puis exécute les étapes suivantes :

1. Retrieval hybride (Qdrant dense + BM25)
2. Déduplication des résultats retrieval
3. Reranking par modèle Qwen
4. Génération de titre + résumé via Ollama
5. Consultation des résultats dans une interface Flask

Chaque étape écrit ses sorties en base via des tables de runs (`retrieval_runs`, `dedup_runs`, `rerank_runs`, `writing_runs`), ce qui permet de rejouer, comparer et auditer les exécutions.

---

## Architecture de la solution

```mermaid
flowchart TD
    A[Sources: CCNews / GDELT / CSV] --> B[Normalisation_dataset.py]
    B --> C[(PostgreSQL: articles)]

    D[expandmodule/generate_expansions.py] --> E[interest/*.json]

    C --> F[news_reco.py]
    E --> F
    F --> G[(Qdrant: news_dense)]
    F --> H[(PostgreSQL: retrieval_runs + retrieval_hits)]

    H --> I[depuplication.py]
    I --> J[(PostgreSQL: dedup_runs + dedup_hits)]

    J --> K[reranker.py]
    H --> K
    K --> L[(PostgreSQL: rerank_runs + rerank_hits)]

    L --> M[writing.py]
    M --> N[(PostgreSQL: writing_runs + article_summaries)]

    N --> O[Flask viewer]
```

### Composants principaux

- `main/db.py`  
  Couche d’accès PostgreSQL (schéma, insert/upsert, lecture par étape).

- `main/news_reco_core.py` + `main/news_reco.py`  
  Retrieval multi-intérêts avec fusion hybride et scoring final.

- `main/depuplication.py`  
  Déduplication des hits retrieval (exact + near-duplicate).

- `main/reranker_core.py` + `main/reranker.py`  
  Reranking LLM (Qwen) + post-traitement pairwise pour diversité.

- `main/writing_core.py` + `main/writing.py`  
  Génération de titres/résumés via Ollama avec parsing robuste et retries.

- `main/orchestration/orchestration.py`  
  Script d’orchestration de bout en bout.

---

## Structure du projet

```text
main/
  db.py
  news_reco_core.py
  news_reco.py
  depuplication.py
  reranker_core.py
  reranker.py
  writing_core.py
  writing.py
  orchestration/
    orchestration.py
  expandmodule/
    generate_expansions.py
    interest/
  ingestiontable/
    Normalisation_dataset.py
    export_dataset.py
    eval.py
    ccnews/
      ccnewsdownload.py
      parse_ccnews_day.py
```

---

## Prérequis

- Linux (recommandé)
- Python 3.12
- Docker (pour PostgreSQL et Qdrant)
- GPU NVIDIA recommandé pour rerank/writing
- Ollama installé et modèle disponible (`qwen3.5:9b-q4_K_M`)

Services attendus :
- PostgreSQL sur `localhost:5432`
- Qdrant sur `localhost:6333`

---

## Installation

### 1) Activer l’environnement Python

```bash
source .venv312/bin/activate
```

### 2) Démarrer les conteneurs

```bash
docker start pfe-qdrant
docker start pfe-postgres
docker ps

si vous ne les avez pas :

docker run -d --name pfe-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=pfe_news \
  -p 5432:5432 postgres:16

docker run -d --name pfe-qdrant \
  -p 6333:6333 \
  qdrant/qdrant
```

### 3) Vérifier Ollama

```bash
ollama list
```

Assurez-vous que `qwen3.5:9b-q4_K_M` est présent.

---

## Démarrage rapide

Exécution pipeline standard pour un intérêt (exemple `finance`) :

```bash
# 1) Normalisation + chargement en DB
python main/ingestiontable/Normalisation_dataset.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --in ccnews_warc_by_day/20260211json \
  --in main/ingestiontable/dataset_top20.csv \
  --chunk-size 1000 \
  --batch-size 1000

# 2) Retrieval (avec --reindex si c'est la première fois, le reindex dure 2h sur RTX 5070 TI)
python main/news_reco.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --topk 1300 \
  --max-expansions-per-interest 10 \
  --dense-per-anchor 600 \
  --dense-per-expansion 250 \
  --bm25-title-k 150 \
  --bm25-body-k 300 \
  --rrf-k 60 \
  --candidate-cap 2500 \
  --min-sim 0.0 \
  --min-bm25 0.0 \
  --interest finance

# 3) Déduplication
/home/pfe/Documents/PFE/.venv312/bin/python main/depuplication.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --interest "finance"

# 4) Reranker (mettre le bon run-id dedup)
python main/reranker.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table dedup_hits \
  --run-id 4 \
  --topn 10 \
  --diversity-scan-k 80 \
  --pairwise-threshold 0.62 \
  --smaxtreshold 0.70 \
  --smintreshold 0.35 \
  --pairwise-batch-size 2 \
  --hydrate

# 5) Writing (mettre le bon rerank-run-id)
python main/writing.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --model qwen3.5:9b-q4_K_M \
  --interest-batch-size 10 \
  --offset 0 \
  --top_n 10 \
  --rerank-run-id 23
```

---

## Exécution complète du pipeline

## A. Récupération des articles

### A.1 CCNews

```bash
python main/ingestiontable/ccnews/ccnewsdownload.py --date 20260211
python main/ingestiontable/ccnews/parse_ccnews_day.py --date 20260211 --skip-existing --workers 8
```

### A.2 GDELT Top 20

```bash
python main/ingestiontable/export_dataset.py --start_date 20260305 --end_date 20260305
```

Note : dans ce script, les arguments sont `--start_date` et `--end_date`.

## B. Génération d’expansions d’intérêts

```bash
/home/pfe/Documents/PFE/.venv312/bin/python main/expandmodule/generate_expansions.py \
  --model qwen3.5:9b-q4_K_M \
  --count 10 \
  --min-words 1 \
  --max-words 2 \
  --out-dir main/expandmodule/interest \
  --topic "war and international conflict" \
  --topic "SpaceX" \
  --topic "Apple" \
  --topic "AI and LLMs" \
  --topic "french politics"
```

## C. Normalisation + chargement base

```bash
python main/ingestiontable/Normalisation_dataset.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --in ccnews_warc_by_day/20260211json \
  --in main/ingestiontable/dataset_top20.csv \
  --chunk-size 1000 \
  --batch-size 1000
```

## D. Retrieval

```bash
python main/news_reco.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --topk 1300 \
  --max-expansions-per-interest 10 \
  --dense-per-anchor 600 \
  --dense-per-expansion 250 \
  --bm25-title-k 150 \
  --bm25-body-k 300 \
  --rrf-k 60 \
  --candidate-cap 2500 \
  --min-sim 0.0 \
  --min-bm25 0.0 \
  --interest finance
```

Important : éviter `--reindex` en routine de run (très coûteux sur gros corpus).

## E. Déduplication

```bash
# Tous les intérêts disponibles (latest run par intérêt)
/home/pfe/Documents/PFE/.venv312/bin/python main/depuplication.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news

# Un intérêt précis
/home/pfe/Documents/PFE/.venv312/bin/python main/depuplication.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --interest "finance"
```

## F. Reranker

```bash
python main/reranker.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table dedup_hits \
  --run-id 4 \
  --topn 10 \
  --diversity-scan-k 80 \
  --pairwise-threshold 0.62 \
  --smaxtreshold 0.70 \
  --smintreshold 0.35 \
  --pairwise-batch-size 2 \
  --hydrate
```

## G. Writing

```bash
python main/writing.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --model qwen3.5:9b-q4_K_M \
  --interest-batch-size 10 \
  --offset 0 \
  --top_n 10 \
  --rerank-run-id 23
```

---

## Commandes d’évaluation et debug

## Upsert du jeu d’évaluation dans `articles`

```bash
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json \
  --table retrieval_hits \
  --run-id 59 \
  --upsert-articles
```

## Évaluation retrieval / dedup / rerank

```bash
# dedup
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table dedup_hits \
  --run-id 2 \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json

# retrieval
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table retrieval_hits \
  --run-id 59 \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json \
  --out main/ingestiontable/evalarticles/report_retrieval_run59.json

# rerank
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table rerank_hits \
  --run-id 9 \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json \
  --out main/ingestiontable/evalarticles/report_rerank_run1.json
```

## Pairwise manuel entre deux articles

```bash
/home/pfe/Documents/PFE/.venv312/bin/python main/reranker.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table dedup_hits \
  --run-id 2 \
  --manual-interest Apple \
  --id1 bcc53846-4c91-5d8a-a125-bc260afaabca \
  --id2 d48750e3-33f6-95c8-8723-6eb68fc7d905 \
  --pairwise-threshold 0.62 \
  --smaxtreshold 0.70 \
  --smintreshold 0.35 \
  --pairwise-batch-size 1 \
  --hydrate
```

---

## Visualisation des résultats

## Viewer simple (résultats writing les plus récents)

```bash
/home/pfe/Documents/PFE/.venv312/bin/python /home/pfe/Documents/PFE/main/fastcheckrerank/save.py
```

Puis ouvrir :

- http://127.0.0.1:8088/show-data

## Viewer complet (sélection d’intérêts + déclenchement pipeline)

```bash
/home/pfe/Documents/PFE/.venv312/bin/python /home/pfe/Documents/PFE/main/front/app.py
```

---

## Paramètres importants

- Base PostgreSQL : `postgresql://postgres:postgres@localhost:5432/pfe_news`
- Qdrant : `http://localhost:6333`
- Modèle writing : `qwen3.5:9b-q4_K_M`
- Table reranker source recommandée : `dedup_hits`

Conseils:
- Commencer par des runs ciblés (1 intérêt) pour valider la chaîne.
- Conserver les `run-id` à chaque étape pour enchaîner correctement.

---

## Dépannage

- Vérifier conteneurs :

```bash
docker ps
```

- Vérifier activation environnement :

```bash
source .venv312/bin/activate
```

- Si une étape ne retourne pas de `run_id`, relancer avec un intérêt unique et vérifier les logs de la commande.
- Si l’interface web n’affiche rien, vérifier que `article_summaries` contient des lignes du dernier `writing_run_id`.

---

## Licence et usage

Projet académique / PFE. Adapter les paramètres de modèles et de base selon votre environnement avant déploiement en production.

---

## Annexe — Détail Retrieval & Reranking

Cette annexe explique **comment fonctionnent** les étapes Retrieval et Reranking, et **à quoi servent les paramètres** que vous passez dans les commandes.

## 1) Retrieval : principe et rôle

Le retrieval (`main/news_reco.py`, cœur dans `main/news_reco_core.py`) sert à trouver un grand pool d’articles pertinents pour un intérêt donné.

Il combine :

- **Dense retrieval (Qdrant)** : similarité sémantique via embeddings.
- **Lexical retrieval (BM25)** : matching mots-clés sur titre/description/corps.

Le système construit plusieurs requêtes par intérêt :

- une requête **anchor** (l’intérêt principal),
- des requêtes **expansions** (termes proches générés),
- éventuellement des **tags**.

Ensuite, il fusionne les candidats dense+BM25, calcule des features de score et produit le top-K final.

## 2) Pipeline interne du Retrieval

Pour chaque intérêt :

1. Génération des `query_specs` (anchor + expansions + tags)
2. Recherche dense Qdrant (`dense_per_anchor`, `dense_per_expansion`)
3. Recherche BM25 (`bm25_title_k`, `bm25_body_k`)
4. Fusion intermédiaire par **RRF pondéré** (`rrf_k`)
5. Cap du nombre de candidats (`candidate_cap`)
6. Rescoring final (features dense + lexical + rrf)
7. Filtrage (`min_sim`, `min_bm25`) puis coupe finale (`topk`)

## 3) Paramètres Retrieval (commande `news_reco.py`)

Commande type :

```bash
python main/news_reco.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --topk 1300 \
  --max-expansions-per-interest 10 \
  --dense-per-anchor 600 \
  --dense-per-expansion 250 \
  --bm25-title-k 150 \
  --bm25-body-k 300 \
  --rrf-k 60 \
  --candidate-cap 2500 \
  --min-sim 0.0 \
  --min-bm25 0.0 \
  --interest finance
```

### Paramètres de connexion / source

- `--db-url` : connexion PostgreSQL (lecture `articles`, écriture des runs/hits retrieval).
- `--qdrant-url` (optionnel) : endpoint Qdrant (défaut `http://localhost:6333`).
- `--collection` (optionnel) : collection dense, par défaut `news_dense`.

### Paramètres de périmètre

- `--interest` (répétable) : intérêt(s) utilisateur.
- `--tags` : tags additionnels séparés par virgules.
- `--lang` : filtre langue (ex `en,fr`).
- `--days` : fenêtre temporelle Qdrant (articles récents uniquement si > 0).
- `--aggregate` : fusionner plusieurs intérêts dans un seul feed.

### Paramètres de volume candidats

- `--topk` : taille finale retournée par intérêt (ou globale en `--aggregate`).
- `--dense-per-anchor` : nb de hits dense pour la requête anchor.
- `--dense-per-expansion` : nb de hits dense pour chaque expansion.
- `--bm25-title-k` : nb de hits BM25 sur titre+description.
- `--bm25-body-k` : nb de hits BM25 sur corps/canonical_text.
- `--candidate-cap` : plafond de candidats après fusion initiale.

### Paramètres de fusion/scoring

- `--rrf-k` : constante du Reciprocal Rank Fusion (plus grand = fusion plus “plate”).
- `--min-sim` : seuil minimal dense pour garder un candidat.
- `--min-bm25` : seuil minimal lexical pour garder un candidat.
- `--dense-only` : désactive BM25 (dense uniquement).

### Paramètres expansions / indexation

- `--max-expansions-per-interest` : limite le nombre d’expansions exploitées.
- `--reindex` : force la réindexation dense complète (coût élevé).
- `--resume-index` : indexation incrémentale (ajoute les manquants).

## 4) Comment régler Retrieval selon l’objectif

- **Max rappel** : augmenter `dense-per-*`, `bm25-*`, `candidate-cap`.
- **Max précision** : augmenter `min-sim` / `min-bm25`, réduire `topk`.
- **Run rapide** : baisser `dense-per-*`, `bm25-*`, `candidate-cap`.
- **Intérêts larges/ambigus** : garder des expansions élevées + BM25 body élevé.

## 5) Reranking : principe et rôle

Le reranker (`main/reranker.py`, cœur dans `main/reranker_core.py`) prend les hits retrieval (ou dedup) et réordonne les articles avec un modèle LLM de type `Qwen`.

Il fait deux choses :

1. **Score de pertinence intérêt↔article** (probabilité “yes” vs “no”).
2. **Diversification post-rerank** avec un juge pairwise (évite les doublons de story dans le top final).

## 6) Pipeline interne du Reranking

1. Charger un run source (`retrieval_hits` ou `dedup_hits`)
2. Construire un prompt par article pour l’intérêt
3. Obtenir un `rerank_score` par hit
4. Trier décroissant par score
5. Scanner un pool (`diversity-scan-k`)
6. Comparer pairwise avec les items déjà conservés
7. Garder uniquement les items “nouvelle story” jusqu’à `topn`

## 7) Paramètres Reranker (commande `reranker.py`)

Commande type :

```bash
python main/reranker.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table dedup_hits \
  --run-id 4 \
  --topn 10 \
  --diversity-scan-k 80 \
  --pairwise-threshold 0.62 \
  --smaxtreshold 0.70 \
  --smintreshold 0.35 \
  --pairwise-batch-size 2 \
  --hydrate
```

### Paramètres source

- `--table` : source des hits (`retrieval_hits` ou `dedup_hits`).
- `--run-id` : identifiant du run source.
- `--db-url` : connexion PostgreSQL.

### Paramètres modèle de rerank

- `--model` : modèle HuggingFace du reranker (`Qwen/Qwen3-Reranker-4B` par défaut).
- `--max-length` : longueur max tokenisée du prompt article.
- `--batch-size` : batch pour scoring initial (intérêt↔article).
- `--instruction` : instruction système personnalisée du juge de pertinence.

### Paramètres diversité pairwise

- `--topn` : taille finale après diversification.
- `--diversity-scan-k` : taille du pool reranké analysé pour la diversité.
- `--pairwise-threshold` : seuil principal de similarité story↔story.
- `--smaxtreshold` : seuil secondaire sur score max bidirectionnel.
- `--smintreshold` : seuil secondaire sur score min bidirectionnel.
- `--pairwise-batch-size` : batch size pour les requêtes pairwise.
- `--pairwise-denoise` / `--no-pairwise-denoise` : active/désactive nettoyage du texte avant pairwise.

### Paramètres enrichissement/debug

- `--hydrate` : ajoute `full_article` depuis `articles` dans `rerank_hits`.
- `--id1` / `--id2` + `--manual-interest` : mode debug manuel pairwise sur 2 articles.

## 8) Intuition sur les seuils pairwise

- `pairwise-threshold` haut (ex. 0.70+) : plus strict, garde plus de variantes.
- `pairwise-threshold` bas (ex. 0.55) : plus agressif, supprime davantage d’articles proches.
- couple `smaxtreshold` + `smintreshold` : filet de sécurité bidirectionnel utile si un sens A→B score différemment de B→A.

Configuration actuelle équilibrée :

- `--pairwise-threshold 0.62`
- `--smaxtreshold 0.70`
- `--smintreshold 0.35`

## 9) Lecture rapide des outputs en base

- `retrieval_hits.score` : score retrieval final (hybride).
- `rerank_hits.rerank_score` : score de pertinence LLM.
- `rerank_hits.rank` : rang final après rerank + diversification.
- `dense_rank` : rang initial avant rerank.

## 10) Recettes prêtes à l’emploi

### A. Maximiser la pertinence (moins de bruit)

- Retrieval : augmenter `min-sim` (ex 0.10–0.20), `min-bm25` (ex 0.05–0.10), réduire `topk`.
- Rerank : garder `topn` bas (8–12), `pairwise-threshold` à 0.62–0.70.

### B. Maximiser la couverture (plus de rappel)

- Retrieval : baisser `min-sim/min-bm25` à 0, augmenter `candidate-cap` et `dense-per-*`.
- Rerank : augmenter `diversity-scan-k` pour laisser plus d’options au filtre pairwise.

### C. Réduire le temps de run

- Retrieval : réduire `dense-per-anchor`, `dense-per-expansion`, `bm25-*`, `candidate-cap`.
- Rerank : réduire `diversity-scan-k`, `max-length`, et limiter `batch-size` selon VRAM.
