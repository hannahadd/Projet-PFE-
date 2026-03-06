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