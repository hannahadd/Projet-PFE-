``` 
yohan@neon:/home/pfe/Documents/PFE$ uv venv .venv312 --python 3.12
Using CPython 3.12.12
Creating virtual environment at: .venv312
Activate with: source .venv312/bin/activate
yohan@neon:/home/pfe/Documents/PFE$ source .venv312/bin/activate
(.venv312) yohan@neon:/home/pfe/Documents/PFE$ python -V
Python 3.12.12
(.venv312) yohan@neon:/home/pfe/Documents/PFE$ python -m ensurepip --upgrade
Looking in links: /tmp/tmpy8hnq8rl
Processing /tmp/tmpy8hnq8rl/pip-25.0.1-py3-none-any.whl
Installing collected packages: pip
Successfully installed pip-25.0.1
(.venv312) yohan@neon:/home/pfe/Documents/PFE$ python -m pip install --upgrade pip setuptools wheel

```


  166  python src/news_reco.py
  167  python src/news_reco.py --reindex
  168  python src/news_reco.py 
  169  python src/news_reco.py --dense-only
  170* 
  171  python src/news_reco.py --reindex --dense-only
  172* 
  173  python src/news_reco.py --reindex --days 100
  174  source /home/pfe/Documents/PFE/.venv312/bin/activate
  175  python src/news_reco.py --reindex --days 100
  176  python src/news_reco2.py --reindex --days 100
  177  python src/news_reco2.py  --days 100 -topK 30
  178  python src/news_reco2.py  --days 100 -topk 30
  179  python src/news_reco2.py  --days 100 --topk 30
  180  python src/news_reco2.py  --days 100 --topk 30 --out src/qwencandidates.json
  181  python src/reranker.py --in src/qwencandidates.json src/qwenrerank.json --max-length 768 --batch-size 1 --topn 10
  182  python src/reranker.py --in src/qwencandidates.json --out src/qwenrerank.json --max-length 768 --batch-size 1 --topn 10
  183  python src/reranker.py --in src/qwencandidates.json --out src/qwenrerank.json --max-length 768 --batch-size 1 --topn 10 --hydrate --dataset src/merged_dataset.json
  184  python ingestionTest/Normalisation_dataset.py
  185  python src/news_reco.py --reindex  --days 100 -topK 30
  186  python src/news_reco.py --reindex  --days 100 --dense-only --topK 30
  187  python src/news_reco.py --reindex  --days 100 --dense-only --topk 30
  188  python src/news_reco.py  --days 100 --dense-only --topk 30 --out src/bgecandidate.json
  189  python src/reranker.py --in src/bgecandidate.json --out src/qwenrerankbge.json --max-length 768 --batch-size 1 --topn 10 --hydrate --dataset src/merged_dataset.json
  190  git add .
  191  git add commit -m "test"
  192  git commit -m "test"
  193  git push
  194  python src/writing.py  --input src/qwenrerankbge.json --model "qwen3.5:9b-q4_K_M" --num_ctx 4096
  195  python src/writing.py  --input src/qwenrerankbge.json --model "qwen3.5:9b-q4_K_M" --debug



sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama

----------------------------------------------------------------------

docker pull qdrant/qdrant:latest

docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest


--------------------------------------------------


# Tes intérêts explicites (toujours 20 chacun => 100)
python src/news_reco.py --dense-only --days 100 --topk 20 \
  --interest "Guerres et conflits internationaux" \
  --interest "IA et les LLMs" \
  --interest "Politique Française" \
  --interest "SpaceX" \
  --interest "Apple"



  python src/news_reco.py --dense-only --days 100 --topk 20

  ython src/news_reco.py --max-articles 200 --reindex --dense-only

  --aggregate



Dans news_reco.py, j’ai:
mis tau_hours par défaut à 336.0 (14 jours)
ajusté le score en --dense-only à: 0.50 * sim + 0.45 * recency + 0.05 * quality
ajouté un flag CLI --tau-hours pour le régler

  ----------------------------------------------------


  “Base de données sérieuse” mais garder du JSON
L’idée propre, c’est: JSON dans la DB (champ JSONB) ou dans un stockage objet, mais pas des fichiers JSON versionnés dans Git.

Option A (recommandée) : PostgreSQL + JSONB + (Qdrant ou pgvector)

PostgreSQL devient la “source of truth” pour les articles.
Table articles avec colonnes normalisées + une colonne raw JSONB pour garder le JSON original.
Index full-text (BM25-like) via tsvector (ou Elastic/OpenSearch si tu scales fort).
Embeddings:
soit Qdrant (tu gardes ton setup actuel) avec article_id comme clé de jointure,
soit pgvector pour tout faire dans Postgres.
Schéma minimal (id stable comme tu fais déjà) :

articles(id TEXT PRIMARY KEY, url TEXT UNIQUE, domain TEXT, published_at TIMESTAMPTZ, lang TEXT, title TEXT, description TEXT, content TEXT, raw JSONB, created_at TIMESTAMPTZ)
Index: UNIQUE(url), index full-text sur to_tsvector(...), éventuellement GIN sur raw.
Option B (local/dev) : SQLite + FTS5 + JSON

Super pour prototyper: un fichier .db (ignoré par git) + FTS5 pour la recherche texte.
Tu gardes le JSON “raw” dans une colonne texte/JSON.
Option C (gros raw) : Postgres + stockage objet (S3/MinIO)

Tu stockes les gros documents bruts (WARC/HTML/JSON compressé) dans MinIO.
Postgres stocke seulement: métadonnées + pointeur (s3_key) + article_id.
C’est le meilleur compromis “pas de fichiers JSON dans le repo” + reproductibilité.
Si tu veux, je peux te préparer un docker-compose.yml (Postgres + Qdrant + MinIO) et un petit script d’ingestion ingest_to_postgres.py qui prend ton merged_dataset.json/jsonl et remplit la table (avec id stable + raw JSONB).









------------------------------------


# 2) Lancer PostgreSQL (Docker)
docker run -d --name pfe-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=pfe_news \
  -p 5432:5432 postgres:16

# 3) Lancer Qdrant (Docker)
docker run -d --name pfe-qdrant \
  -p 6333:6333 \
  qdrant/qdrant

# 4) (Optionnel pour writing) Lancer Ollama + modèle
ollama serve
ollama pull qwen3.5:9b-q4_K_M

# 5) Normaliser et charger les articles en DB
python main/ingestiontable/Normalisation_dataset.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news

python main/ingestiontable/Normalisation_dataset.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --in ccnews_warc_by_day/20260211json   --in main/ingestiontable/dataset_top20.csv --chunk-size 1000 --batch-size 1000


# 6) Retrieval -> PostgreSQL (récupère retrieval_run_id affiché)
python main/news_reco.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --interest "IA et les LLMs" --interest "SpaceX" --topk 20

python main/news_reco.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --dense-only --reindex --topk 200

python main/news_reco.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --dense-only --resume-index --topk 200

# 7) Rerank -> PostgreSQL (remplacer XXX par retrieval_run_id)
python main/reranker.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --retrieval-run-id XXX --topn 10 --hydrate

python main/reranker.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --topn 10 --hydrate --retrieval-run-id X

# 8) Writing -> PostgreSQL par batch de 10 intérêts (remplacer YYY par rerank_run_id)
python main/writing.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --rerank-run-id YYY \
  --model qwen3.5:9b-instruct-q4_K_M \
  --interest-batch-size 10 --offset 0 --top_n 10

python main/writing.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --model qwen3.5:9b-q4_K_M \
  --interest-batch-size 10 --offset 0 --top_n 10 \
  --rerank-run-id 2


# 9) Batch suivant (10 intérêts suivants)
python main/writing.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --rerank-run-id YYY \
  --model qwen3.5:9b-q4_K_M \
  --interest-batch-size 10 --offset 10 --top_n 10




------------------------------

# 1) Évaluer un run retrieval
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --run-id 1 \
  --table retrieval_hits \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json \
  --out main/ingestiontable/evalarticles/report_retrieval_run1.json

/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table retrieval_hits \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json \
  --out main/ingestiontable/evalarticles/report_retrieval_run1.json \
  --run-id 1 

# 2) Évaluer un run rerank
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --run-id 1 \
  --table rerank_hits \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json \
  --out main/ingestiontable/evalarticles/report_rerank_run1.json

  # 2) Évaluer un run rerank
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table rerank_hits \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json \
  --out main/ingestiontable/evalarticles/report_rerank_run1.json \
  --run-id 1 

# 3) Ajouter les articles d’éval dans la table articles + évaluer
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --run-id 1 \
  --table rerank_hits \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json \
  --upsert-articles \
  --out main/ingestiontable/evalarticles/report_rerank_run1.json

# juste upsert article test

/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --eval-file main/ingestiontable/evalarticles/evalarticles.json   --upsert-articles 
Upserted eval articles into articles table: 10



# #lance le site web, ensuite si on est sur un pc distant, il faut taper directement l'IP de la http://100.86.221.84:8080/ ou bien http://localhost:8000
python3 /home/pfe/Documents/PFE/main/fastcheckrerank/app.py 

/home/pfe/Documents/PFE/.venv312/bin/python /home/pfe/Documents/PFE/main/fastcheckrerank/save.py

http://127.0.0.1:8088/show-data


---------------------------------------------------------

/home/pfe/Documents/PFE/.venv312/bin/python main/expandmodule/generate_expansions.py   --model qwen3.5:9b-q4_K_M   --count 10 --min-words 1 --max-words 2   --out-dir main/expandmodule/interest   --topic "war and international conflict"   --topic "SpaceX"   --topic "Apple"   --topic "AI and LLMs"   --topic "french politics"



------------------------------------------------------------


  yohan@neon:~/Downloads$ docker ps
CONTAINER ID   IMAGE           COMMAND                  CREATED          STATUS          PORTS                                         NAMES
e01555ed3c10   qdrant/qdrant   "./entrypoint.sh"        38 minutes ago   Up 38 minutes   0.0.0.0:6333->6333/tcp, [::]:6333->6333/tcp   pfe-qdrant
9b89bbbd5dd8   postgres:16     "docker-entrypoint.s…"   39 minutes ago   Up 39 minutes   0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp   pfe-postgres
yohan@neon:~/Downloads$ 



yohan@neon:~/Desktop/note$ docker ps -a
CONTAINER ID   IMAGE                  COMMAND                  CREATED        STATUS                       PORTS                              NAMES
e01555ed3c10   qdrant/qdrant          "./entrypoint.sh"        14 hours ago   Exited (255) 2 minutes ago   0.0.0.0:6333->6333/tcp, 6334/tcp   pfe-qdrant
9b89bbbd5dd8   postgres:16            "docker-entrypoint.s…"   14 hours ago   Exited (255) 2 minutes ago   0.0.0.0:5432->5432/tcp             pfe-postgres
a0038f668175   qdrant/qdrant:latest   "./entrypoint.sh"        39 hours ago   Exited (143) 14 hours ago                                       qdrant
03ff90fce65e   hello-world            "/hello"                 39 hours ago   Exited (0) 39 hours ago                                         objective_galois
yohan@neon:~/Desktop/note$ docker start pfe-qdrant
pfe-qdrant
yohan@neon:~/Desktop/note$ docker start pfe-postgres
pfe-postgres
yohan@neon:~/Desktop/note$ docker ps
CONTAINER ID   IMAGE           COMMAND                  CREATED        STATUS          PORTS                                         NAMES
e01555ed3c10   qdrant/qdrant   "./entrypoint.sh"        14 hours ago   Up 14 seconds   0.0.0.0:6333->6333/tcp, [::]:6333->6333/tcp   pfe-qdrant
9b89bbbd5dd8   postgres:16     "docker-entrypoint.s…"   14 hours ago   Up 5 seconds    0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp   pfe-postgres
yohan@neon:~/Desktop/note$ 

# Python (Website)

Pour lancer le site web  : source .venv/bin/activate (car dépendances python qui n'était pas possible de prendre sans l'env virtuel)

(.venv) pfe@neon:~$ python3 Documents/PFE/main/front/app.py #lance le site web, ensuite si on est sur un pc distant, il faut taper directement l'IP de la machine dans le navigateur : http://100.86.221.84:8080/ ou bien http://localhost:8000 pour yohan (j'ai pas testé mais ça devrait marcher)

----------------------------------------------------

# new retrieval call
(.venv312) yohan@neon:/home/pfe/Documents/PFE$ python main/news_reco.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --topk 1300   --max-expansions-per-interest 10   --dense-per-anchor 600   --dense-per-expansion 250   --bm25-title-k 150   --bm25-body-k 300   --rrf-k 60   --candidate-cap 2500   --min-sim 0.0   --min-bm25 0.0

# new dedup

/home/pfe/Documents/PFE/.venv312/bin/python main/depuplication.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news

/home/pfe/Documents/PFE/.venv312/bin/python main/depuplication.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --interest "AI and LLMs"

# new rerank

/home/pfe/Documents/PFE/.venv312/bin/python main/reranker.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table dedup_hits \
  --run-id DEDUP_RUN_ID \
  --topn 10 --hydrate

# eval dedup :
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --table dedup_hits \
  --run-id 1 \
  --eval-file main/ingestiontable/evalarticles/evalarticles.json