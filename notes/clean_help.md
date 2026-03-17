New command 


# 5) recuperation des articles :
python main/ingestiontable/ccnews/ccnewsdownload.py  --date 20260211
python main/ingestiontable/ccnews/parse_ccnews_day.py  --date 20260211 --skip-existing  --workers 8
+
ton code a toi a remplir 

python main/ingestiontable/export_dataset.py --start-date 20260305 --end_date 20260308

# 5) Normaliser et charger les articles en DB
python main/ingestiontable/Normalisation_dataset.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --in ccnews_warc_by_day/20260211json   --in main/ingestiontable/dataset_top20.csv --chunk-size 1000 --batch-size 1000

# new retrieval call
python main/news_reco.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --topk 1300   --max-expansions-per-interest 10   --dense-per-anchor 600   --dense-per-expansion 250   --bm25-title-k 150   --bm25-body-k 300   --rrf-k 60   --candidate-cap 2500   --min-sim 0.0   --min-bm25 0.0 --interest finance

(n'y touche pas et surtout pas de flag --reindex, ça dure 2h30)


# new dedup (le truc a modifier) ???

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

side commaand and debugging

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





----------------------------------
# probleme a regler :

## faire l'orchestrateur, attention, clean la memoire gpu avant de lancer chaque scripts pour eviter ça (se declenche de maniere random):


Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Traceback (most recent call last):
  File "/home/pfe/Documents/PFE/main/reranker.py", line 647, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/pfe/Documents/PFE/main/reranker.py", line 530, in main
    reranker = core.QwenReranker(
               ^^^^^^^^^^^^^^^^^^
  File "/home/pfe/Documents/PFE/main/reranker_core.py", line 103, in __init__
    self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pfe/Documents/PFE/.venv312/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 374, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pfe/Documents/PFE/.venv312/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4137, in from_pretrained
    loading_info, disk_offload_index = cls._load_pretrained_model(model, state_dict, checkpoint_files, load_config)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pfe/Documents/PFE/.venv312/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4216, in _load_pretrained_model
    caching_allocator_warmup(model, expanded_device_map, load_config.hf_quantizer)
  File "/home/pfe/Documents/PFE/.venv312/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4810, in caching_allocator_warmup
    _ = torch.empty(int(byte_count // 2), dtype=torch.float16, device=device, requires_grad=False)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.49 GiB. GPU 0 has a total capacity of 15.47 GiB of which 6.71 GiB is free. Process 28800 has 15.53 MiB memory in use. Process 54776 has 7.75 GiB memory in use. Including non-PyTorch memory, this process has 222.00 MiB memory in use. Of the allocated memory 0 bytes is allocated by PyTorch, and 0 bytes is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)




## "dans writing, regler les fails (second pass), et ne pas les afficher (les fails ont la colonne "notes" non nul dans articles_summaries )

(.venv312) yohan@neon:/home/pfe/Documents/PFE$ python main/writing.py   --db-url postgresql://postgres:postgres@localhost:5432/pfe_news   --model qwen3.5:9b-q4_K_M   --interest-batch-size 10 --offset 0 --top_n 10   --rerank-run-id 23
Writing run created: writing_run_id=29 | interests=7 (0:10)
[RUN] 1/7 AI and LLMs (articles=10)
  [OK] AI and LLMs [1/10] saved (article_summary_id=1164)
  [OK] AI and LLMs [2/10] saved (article_summary_id=1165)
  [OK] AI and LLMs [3/10] saved (article_summary_id=1166)
  [OK] AI and LLMs [4/10] saved (article_summary_id=1167)
  [FAIL] AI and LLMs [5/10] saved (article_summary_id=1168)
  [OK] AI and LLMs [6/10] saved (article_summary_id=1169)
  [FAIL] AI and LLMs [7/10] saved (article_summary_id=1170)
  [OK] AI and LLMs [8/10] saved (article_summary_id=1171)
  [OK] AI and LLMs [9/10] saved (article_summary_id=1172)
  [OK] AI and LLMs [10/10] saved (article_summary_id=1173)
[RUN] 2/7 Apple (articles=10)


## completer les commandes  "recuperation des articles" avec ton code a toi qui creer le csv, si possible met ton script dans main/ingestiontable/top20


## Faire un plan complet de la presentation qui s'inspire de la presentation de mi parcours (inspire toi de toutes les commandes et fonctionnalités comme eval.py, save.py (notre propre visualisateur ect...))

pour la presentation, hanna peut prendre la db, deduplication writing, ingestion (le schema general)
moi reranker et toi retrieval ou inversement 


