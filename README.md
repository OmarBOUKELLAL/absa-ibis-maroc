# Exploitation des avis clients via l'analyse des sentiments pour l'optimisation de la performance hôtelière : Étude de cas — Ibis Maroc

Analyse de **30 494 avis clients** multilingues collectés sur Booking.com, TripAdvisor et Expedia pour les **15 hôtels ibis du Maroc**, afin de transformer la voix du client en un véritable outil de pilotage de la performance hôtelière.

Ce dépôt accompagne un mémoire de Master (Marketing, Revenue Management & Data Science, Université Mohammed Premier, FSJES Oujda).

---

## Objectif

Identifier, à partir des avis en ligne, les **forces et les faiblesses** opérationnelles des hôtels ibis au Maroc, et comprendre comment ces enseignements — différenciés selon le **profil du voyageur**, l'**établissement** et la **saison** — peuvent orienter une stratégie d'optimisation de la performance et de la satisfaction client.

**Questions de recherche :**

- **QR1 :** Quels aspects de l'expérience sont associés à la satisfaction / insatisfaction ?
- **QR2 :** Dans quelle mesure le profil, l'établissement et la saison modulent-ils cette perception ?
- **QR3 :** Comment ces enseignements orientent-ils la stratégie d'ibis Maroc 2025-2030 ?

---

## Méthodologie — pipeline ABSA multilingue

Le pipeline suit le cadre **CRISP-DM** :

1. **Collecte** — web scraping (Selenium pour Booking & Expedia, DrissionPage pour TripAdvisor), détection de langue avec *lingua*.
2. **Prétraitement & segmentation** — nettoyage minimal, découpage en phrases (NLTK) puis sur les connecteurs adversatifs multilingues.
3. **Embeddings** — vecteurs de sens via `paraphrase-multilingual-mpnet-base-v2` (Sentence-BERT, 768 dimensions, 50+ langues).
4. **Affectation des aspects** — 11 aspects prédéfinis ; rattachement par **similarité cosinus** : BERTopic (zeroshot, seuil 0,45) puis rattrapage cosinus (seuil 0,35).
5. **Sentiment** — classification positif / neutre / négatif par le modèle pré-entraîné `cardiffnlp/xlm-roberta-base-sentiment-multilingual`.
6. **Exploitation** — agrégation en **Net Sentiment Score** (NSS = % positifs − % négatifs), par aspect, hôtel, profil et saison.

---

## Structure du dépôt

```
├── 01_scraping_booking.ipynb          # Collecte des avis Booking.com (Selenium)
├── 02_scraping_expedia.ipynb          # Collecte des avis Expedia (Selenium)
├── 03_scraping_tripadvisor.ipynb      # Collecte des avis TripAdvisor (DrissionPage)
├── 04_consolidation_corpus.ipynb      # Fusion des 3 plateformes, uniformisation, dédoublonnage
├── 05_embeddings.ipynb                # Vecteurs de sens (mpnet)
├── 06_absa_sentiment_validation.ipynb # Affectation des aspects, sentiment (XLM-RoBERTa), validation
├── 07_resultats_visualisations.ipynb  # Net Sentiment Score, analyses croisées, figures
├── data/
│   └── echantillon_avis.csv           # Échantillon du corpus (le corpus complet est sur Kaggle)
└── README.md
```

> Les noms de notebooks sont indicatifs — adapte-les à tes fichiers réels.

---

## Les 11 aspects analysés

Personnel · Service · Chambre & propreté · Restauration · Localisation · Expérience & valeur · Rapport qualité-prix · Équipements & activités · Bruit & sommeil · Confort thermique · Wi-Fi

*(8 aspects repris d'Ounacer et al., 2023 ; 3 dimensions complémentaires — bruit, confort thermique, Wi-Fi.)*

---

## Données

Le corpus complet (30 494 avis) et les fichiers volumineux (embeddings) sont hébergés sur **Kaggle** :

> 🔗 https://www.kaggle.com/datasets/omarboukellalfsfl/df-final-absa-v4/settings

Le dossier `data/` de ce dépôt contient seulement un **échantillon anonymisé** permettant de tester le code.

---

## Validation

Le pipeline a été validé sur un **échantillon de 200 avis annotés manuellement** (jeu de test uniquement) :

| Métrique | Affectation des aspects | Classification du sentiment |
|---|---|---|
| Exactitude | 0,78 | 0,88 |
| F1 macro | 0,75 | 0,87 |
| Cohen's Kappa | 0,75 (substantiel) | 0,82 (quasi parfait) |

---

## Résultats clés

- Sentiment global : **51 % positifs · 28 % neutres · 21 % négatifs** (note moyenne 3,59/5).
- **Forces :** expérience, personnel, service — la satisfaction repose d'abord sur l'humain.
- **Irritants structurels :** bruit (−33) et confort thermique (−36), liés au bâti.
- **Wi-Fi :** aspect le plus discriminant — déficient dans les hôtels d'affaires de Casablanca.
- **Saisonnalité :** perception stable, mais pic d'insatisfaction en août (saturation opérationnelle, non thermique).

---

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn
pip install sentence-transformers bertopic umap-learn hdbscan
pip install transformers torch nltk lingua-language-detector
pip install selenium webdriver-manager DrissionPage
```

> Le calcul des embeddings et du sentiment est nettement plus rapide sur GPU.

---

## Références principales

- Ameur, A., Hamdi, S., & Ben Yahia, S. (2023). *Sentiment analysis for hotel reviews: A systematic literature review.* ACM Computing Surveys, 56(2).
- Ounacer, S., et al. (2023). *Customer sentiment analysis in hotel reviews through NLP techniques.* IJACSA, 14(1).
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT.* EMNLP.
- Grootendorst, M. (2022). *BERTopic.* arXiv:2203.05794.
- Huang, Liang & Choi (2022). *How we failed in context.* Sustainability, 14(5).

---

## Auteur

**BOUKELLAL Omar** — Master Marketing, Revenue Management & Data Science
Université Mohammed Premier — FSJES Oujda
Encadrant : Pr. Saïd EL MELHAOUI

---

## Licence

Distribué sous licence **MIT** — libre réutilisation avec attribution.
