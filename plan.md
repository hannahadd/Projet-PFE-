# Plan de soutenance — Application Web IA (20 min)

## 0) Objectif de la présentation
Montrer l’aboutissement du MVP : une application web de recommandation d’actualités personnalisées basée sur un pipeline IA complet (ingestion → retrieval → dedup → reranking → writing → visualisation), avec un retour honnête sur les performances et limites.

---

## 1) Déroulé global (20 min)

- **00:00 → 10:00 : Présentation technique**
- **10:00 → 15:00 : Démonstration live scriptée**
- **15:00 → 20:00 : Questions / réponses (et plan B video si incident)**



---

## 2) Plan détaillé des 10 minutes de présentation

## 00:00 → 01:00 — Contexte & besoin
- Problème : surcharge informationnelle, difficulté à obtenir un flux d’articles pertinent par intérêt.
- Besoin MVP : générer rapidement un feed d’actualités utile, diversifié et lisible via interface web.
- Valeur ajoutée : combiner retrieval large échelle + reranking sémantique + génération de résumé.

## 01:00 → 02:00 — Vision produit / MVP
- Entrée utilisateur : un intérêt (ex. finance, Apple, SpaceX, etc.).
- Sortie utilisateur : top d’articles pertinents + résumé exploitable.
- Contraintes : volume élevé (~300k articles), temps de calcul maîtrisé, robustesse en conditions réelles.

## 02:00 → 04:00 — Architecture finale (vue pipeline)
Présenter le schéma global :
1. **Ingestion & normalisation** (CCNews + GDELT)
2. **Stockage** PostgreSQL (`articles`, runs/hits)
3. **Index dense** Qdrant
4. **Retrieval hybride** (dense + BM25)
5. **Déduplication**
6. **Reranking** (Qwen)
7. **Writing** (titre + résumé via Ollama)
8. **Visualisation web** (Flask)

Message clé : architecture modulaire, traçable (run IDs), reproductible étape par étape.

## 04:00 → 06:00 — Choix techniques IA & data
- **Retrieval** : combinaison dense + lexical pour maximiser le rappel sur gros corpus.
- **Reranking** : priorise les articles réellement intéressants pour l’intérêt utilisateur.
- **Reranking de diversité** : réduit les redondances de "même story".
- **Writing** : transforme la sortie technique en information lisible côté utilisateur.

Pourquoi ce choix : 
- Dense seul = trop large/bruité sur certains sujets,
- BM25 seul = manque de sémantique,
- Rerank seul sans retrieval large =  beaucoup trop couteux en ressources sur 300k.

note :
Simple AI = BM25 + règles/fuzzy matching,
Deep Learning = embeddings + reranker + génération LLM.

## 06:00 → 08:00 — Résultats obtenus (honnêtes et chiffrés)
Utiliser exactement vos chiffres :
- Retrieval avec `topk=1300` sur ~300k articles : **9/10 articles cibles retrouvés**, mais score faible.
- Reranking : qualité utile mais variable selon le sujet/prompt, **~7/10 articles intéressants en moyenne**.
- Reranking diversité : globalement bon, mais perturbable par bruit de contenu (éléments de navigation non filtrés).
- Performance pipeline : **4–5 min** pour créer un intérêt (hors réindexation).
- Coût principal : **indexing Qdrant ~2h** sur 300k articles.
- Writing : **1 à 2 fails sur 60** générations.


## 08:00 → 09:00 — Limites & difficultés rencontrées
- Sensibilité au prompt côté reranking.
- Concurrence d’articles proches sur un même événement (certains "prennent la place" des cibles).
- Qualité bruitée des contenus source (menus/boutons/parasites web).
- Temps de réindexation élevé.
- Robustesse writing encore perfectible (fails résiduels).

- finir sur une note positive : pour un systeme sans utilisateur, pour faire les recommandations, c'est assez bon, permet de faire un cold start, peut s'ameliorer avec quelques idées présenté plus tard

## 09:00 → 10:00 — Pistes d’amélioration (inclure la touche MLOps)
- Nettoyage HTML plus agressif en amont (anti-bruit UI/site).
- personalisé le prompting par interet sur le reranking automatique via IA.
- améliorer le modele de reranking de 4b à 8b.
- Post-check de qualité sur writing + retry contrôlé.
- Pistes MLOps : suivi métriques et parametres utilisés pour les run

---

## 3) Démonstration live (5 min) — script rigoureux

## 10:00 → 10:30 — Préambule démo
- Annoncer le scénario : "Je montre un run complet sur un intérêt et le résultat final dans l’interface".
- Vérifier rapidement services (`postgres`, `qdrant`, `ollama`) déjà démarrés.

## 10:30 → 12:00 — Exécution pipeline (version courte préparée)
- Montrer un intérêt test connu.
- Lancer la séquence déjà prête (ou orchestrateur) et afficher :
  - récupération/normalisation ok,
  - retrieval run id,
  - dedup run id,
  - rerank run id,
  - writing run id.

> Important : utiliser un dataset/état préchauffé pour tenir le timing 5 min.

## 12:00 → 13:30 — Visualisation web
- Ouvrir l’interface Flask.
- Afficher les articles finaux pour l’intérêt choisi.
- Montrer 2–3 exemples pertinents et signaler un cas limite si présent (transparence).

## 13:30 → 14:30 — Validation qualitative rapide
- "Sur ce run : X articles pertinents sur Y".
- Illustrer la diversité (pas 10 articles quasi identiques).

## 14:30 → 15:00 — Conclusion démo
- Rappeler ce qui est atteint : MVP fonctionnel de bout en bout.
- Rappeler ce qui reste : robustesse qualité sur sujets difficiles + réduction des fails writing.

---

## 4) Plan B (obligatoire) — vidéo de secours

Préparer une vidéo de 3 à 5 minutes montrant :
1. lancement pipeline,
2. IDs de run générés,
3. affichage interface finale,
4. commentaire oral sur un succès + une limite.

Usage : **uniquement** en cas d’incident technique le jour J.

---

## 5) Répartition suggérée des rôles (si passage à plusieurs)

Option alignée avec votre organisation actuelle :
- **Personne A** : ingestion, architecture globale, DB, déduplication, writing.
- **Personne B** : retrieval, reranking, analyse des scores/limites.

Transition recommandée :
- A termine l’architecture et passe à B sur "qualité ranking et résultats chiffrés".

---

## 6) Checklist répétition (J-1 / Jour J)

## Technique
- [ ] Docker `pfe-postgres` et `pfe-qdrant` démarrés
- [ ] Environnement Python activé (`.venv312`)
- [ ] Ollama + modèle disponibles
- [ ] Base déjà alimentée (éviter index full 2h en live)
- [ ] Commandes démo copiées dans un fichier prêt à exécuter

## Contenu
- [ ] Chrono respecté (10 min + 5 min)
- [ ] Chiffres clés mémorisés (9/10, 7/10, 4–5 min, 2h, 1–2/60)
- [ ] Limites assumées clairement
- [ ] 2 pistes d’amélioration prioritaires annoncées

## Backup
- [ ] Vidéo de secours testée
- [ ] Captures d’écran critiques prêtes (architecture + interface + résultats)

---

## 7) Script court d’ouverture (30 secondes)

"Notre objectif est de fournir un flux d’actualités personnalisé, pertinent et diversifié à partir d’un très grand volume d’articles. Nous avons construit un pipeline IA complet : ingestion, retrieval hybride, déduplication, reranking, génération de résumés, puis visualisation web. Nous allons présenter l’architecture, les résultats obtenus, leurs limites, puis faire une démonstration live du MVP."

---

## 8) Script court de clôture (20 secondes)

"Le MVP est fonctionnel de bout en bout et produit des résultats utiles dans des temps compatibles avec un usage quasi-opérationnel. Les principaux axes de progrès sont la robustesse du reranking sur certains sujets et la réduction des échecs résiduels au writing."
