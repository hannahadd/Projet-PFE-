New command 


# recuperation des articles (deux sources de données):
python main/ingestiontable/ccnews/ccnewsdownload.py  --date 20260211
python main/ingestiontable/ccnews/parse_ccnews_day.py  --date 20260211 --skip-existing  --workers 8
+
python main/ingestiontable/export_dataset.py --start-date 20260305 --end_date 20260308

# Normaliser et charger les articles en DB
python main/ingestiontable/Normalisation_dataset.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --in ccnews_warc_by_day/20260211json   --in main/ingestiontable/dataset_top20.csv --chunk-size 1000 --batch-size 1000

# new retrieval call
python main/news_reco.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --topk 1300   --max-expansions-per-interest 10   --dense-per-anchor 600   --dense-per-expansion 250   --bm25-title-k 150   --bm25-body-k 300   --rrf-k 60   --candidate-cap 2500   --min-sim 0.0   --min-bm25 0.0 --interest finance

(n'y touche pas et surtout pas de flag --reindex, ça dure 2h30)


# new dedup ???

/home/pfe/Documents/PFE/.venv312/bin/python main/depuplication.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news

/home/pfe/Documents/PFE/.venv312/bin/python main/depuplication.py \
  --db-url postgresql://postgres:postgres@localhost:5432/pfe_news \
  --interest "finance"


# new rerank with diversity
python main/reranker.py --db-url postgresql://postgres:postgres@localhost:5432/pfe_news --table dedup_hits --run-id 4 --topn 10 --diversity-scan-k 80 --pairwise-threshold 0.62 --smaxtreshold 0.70 --smintreshold 0.35 --pairwise-batch-size 2 --hydrate

# writing
python main/writing.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --model qwen3.5:9b-q4_K_M   --interest-batch-size 10 --offset 0 --top_n 10   --rerank-run-id 23


---------------------------------------------------

side command and debugging

# env

source .venv312/bin/activate


# juste upsert article test
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --eval-file main/ingestiontable/evalarticles/evalarticles.json   --upsert-articles 


# juste check le diversity score entre deux articles
pair spaceX merger
 --id1 119c743c-307c-f4ec-105b-cf80fe5a65a7 --id2 bcbce7d7-4fb0-337c-4ba0-edcbac6e1277 

apple iphone 17e
/home/pfe/Documents/PFE/.venv312/bin/python main/reranker.py --db-url postgresql://postgres:postgres@localhost:5432/pfe_news --table dedup_hits --run-id 2 --manual-interest Apple --id1 bcc53846-4c91-5d8a-a125-bc260afaabca --id2 d48750e3-33f6-95c8-8723-6eb68fc7d905 --pairwise-threshold 0.62 --smaxtreshold 0.70 --smintreshold 0.35 --pairwise-batch-size 1 --hydrate


# eval des differentes tables

/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --table dedup_hits   --run-id 2   --eval-file main/ingestiontable/evalarticles/evalarticles.json


/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --table retrieval_hits   --eval-file main/ingestiontable/evalarticles/evalarticles.json   --out main/ingestiontable/evalarticles/report_retrieval_run59.json   --run-id 59

  
/home/pfe/Documents/PFE/.venv312/bin/python main/ingestiontable/eval.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --table rerank_hits   --eval-file main/ingestiontable/evalarticles/evalarticles.json   --out main/ingestiontable/evalarticles/report_rerank_run1.json   --run-id 9 


# verfier writing

/home/pfe/Documents/PFE/.venv312/bin/python /home/pfe/Documents/PFE/main/fastcheckrerank/save.py
http://127.0.0.1:8088/show-data


# expand module

/home/pfe/Documents/PFE/.venv312/bin/python main/expandmodule/generate_expansions.py   --model qwen3.5:9b-q4_K_M   --count 10 --min-words 1 --max-words 2   --out-dir main/expandmodule/interest   --topic "war and international conflict"   --topic "SpaceX"   --topic "Apple"   --topic "AI and LLMs"   --topic "french politics"


# start docker 

docker start pfe-qdrant
docker start pfe-postgres

docker ps
CONTAINER ID   IMAGE           COMMAND                  CREATED        STATUS          PORTS                                         NAMES
e01555ed3c10   qdrant/qdrant   "./entrypoint.sh"        14 hours ago   Up 14 seconds   0.0.0.0:6333->6333/tcp, [::]:6333->6333/tcp   pfe-qdrant
9b89bbbd5dd8   postgres:16     "docker-entrypoint.s…"   14 hours ago   Up 5 seconds    0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp   pfe-postgres


# app

/home/pfe/Documents/PFE/.venv312/bin/python /home/pfe/Documents/PFE/main/front/app.py