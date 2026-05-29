# Découverte de Pattern Functional Dependencies approximatives

Mini-projet du cours Data Wrangling — K. Belhajjame, Paris Dauphine.

Le projet implémente deux approches de découverte de PFDs :
- Une approche classique par énumération exhaustive (Tâche 1)
- Une approche guidée par LLM dite "Guided Search" (Tâche 2, Workflow 2 du cours)

## Structure

```
src/                     modules du pipeline classique
  pattern_extractor.py     extraction des patterns
  candidate_generator.py   génération des candidats
  rule_generalizer.py      fusion et élagage
pfd_verifier.py          vérification d'une PFD (support, confidence)
pfd_discovery.py         orchestration du pipeline classique
notebooks/               notebooks d'exploration et d'expérience
  01 à 05                  tâche 1
  06                       tâche 2 (workflow guided search)
tests/                   63 tests unitaires et d'intégration
data/                    datasets t1, t2, t3, CHE, DGOV
results/                 PFDs découvertes au format CSV
```

## Datasets

| Fichier | Lignes | Colonnes | Description |
|---------|--------|----------|-------------|
| t1.csv  | 9 101  | 9        | Employés du comté de Montgomery |
| t2.csv  | 3 502  | 13       | Employeurs de la ville de Chicago |
| t3.csv  | 1 077  | 9        | Licences d'alcool du Maryland |

## Prérequis

- Python 3.10+
- pandas, requests, pytest
- Pour la tâche 2 : Ollama (DeepSeek en local) ou une clé API Groq

## Installation

```
pip install pandas requests pytest
```

Sur Kali Linux, ajouter `--break-system-packages`.

Pour utiliser DeepSeek en local, installer Ollama puis télécharger le modèle :

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:8b
```

Le modèle 8b demande environ 6 Go de RAM. Une version plus légère (1.5b, 1 Go) est aussi disponible :

```
ollama pull deepseek-r1:1.5b
```

Pour utiliser Groq via API, créer une clé sur https://console.groq.com puis installer la librairie :

```
pip install groq
```

La clé se renseigne dans la cellule de configuration du notebook `06_workflow2_guided_search.ipynb`.

## Exécution

Pipeline classique en ligne de commande :

```
python pfd_discovery.py --dataset data/pfd_validation/t2.csv --epsilon 0.1
```

Options : `--epsilon` (tolérance), `--min-support`, `--k-max`, `--top N`.

Pour la tâche 2, lancer Ollama si on utilise DeepSeek :

```
ollama serve
```

Puis ouvrir le notebook :

```
jupyter notebook notebooks/06_workflow2_guided_search.ipynb
```

Le choix du LLM se fait dans la cellule de configuration (`MODELE = "deepseek"` ou `"groq"`).

## Tests

```
python -m pytest tests/
```

## Auteurs

Abdrahamane CAMARA, Yuxuan CHEN, Gide FOMAT, Hongxiang LIN, Liya XU.
