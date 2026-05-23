# Mini-projet — Pattern Functional Dependencies (PFDs) approximatives

Projet universitaire de **Data Wrangling** : découverte et validation de PFDs approximatives, comparaison d'une approche algorithmique classique avec une approche agentique pilotée par un LLM.

> Voir `Approximate_PFDs.pdf` pour la définition formelle et le contexte théorique.

---

## Objectif général

Étudier la découverte de PFDs approximatives sur des données réelles, puis comparer :
- une approche **100 % algorithmique** (classique)
- une approche **agentique** où un LLM guide ou réalise une partie de la découverte

---

## Concept

Une **Pattern Functional Dependency (PFD)** est une règle de la forme :

> Si la valeur de la colonne `X` correspond au pattern `P1`,  
> alors la valeur de la colonne `Y` doit correspondre au pattern `P2`.

La version **approximative** tolère un taux d'exception `epsilon` :

```
confiance = nb(X match ET Y match) / nb(X match)
PFD valide  ⟺  confiance ≥ (1 − epsilon)
```

Exemples de PFDs utiles :
- `prefix(zip, 3)` → `city`
- `domain(email)` → `organisation`
- `first_token(name)` → `gender`

---

## Tâches à réaliser

### Tâche 1 — Découverte classique de PFDs `[ ]`

Implémenter l'algorithme complet de découverte algorithmique :

1. **Extraction de patterns** : à partir des valeurs d'une colonne, générer des patterns candidats (préfixes, tokens, sous-chaînes, n-grams, expressions régulières).
2. **Génération de candidats** : construire les paires `(pattern_X, col_Y)` candidates à tester comme PFDs.
3. **Validation** : pour chaque candidat, calculer le support et la confiance et appliquer les seuils.
4. **Généralisation** : fusionner les règles spécifiques en règles plus générales (ex. `"John*" → M` + `"James*" → M` → `first_token(name) → M`).

> **Base de travail :** `pfd_verifier.py` implémente déjà la validation (étape 3) — il sert de brique pour les étapes amont.

---

### Tâche 2 — Workflow agentique `[ ]`

Implémenter **au moins un** des trois workflows suivants où un LLM intervient dans la découverte :

#### Workflow A — Feature-Enriched Discovery
Le LLM suggère les transformations pertinentes (ex. `prefix(zip,3)`, `domain(email)`), puis l'algorithme classique réalise la découverte complète sur ces features enrichies.

```
Agent  →  suggestions de transformations
Algorithme  →  découverte complète sur la table enrichie
```

#### Workflow B — Guided Search
Le LLM suggère directement les dépendances candidates les plus prometteuses, l'algorithme se contente de les valider (support + confiance).

```
Agent  →  sélection des candidats X → Y
Algorithme  →  validation des candidats retenus uniquement
```

#### Workflow C — Agent-in-the-Loop *(le plus ambitieux)*
Boucle itérative : le LLM propose des règles, l'algorithme valide, les résultats sont renvoyés au LLM pour raffinement.

```
Agent  →  hypothèse initiale
Algorithme  →  validation + retour (violations, confiance)
Agent  →  raffinement (changer le préfixe, la colonne, le seuil…)
          [répéter jusqu'à convergence]
```

---

### Tâche 3 — Expériences et comparaison `[ ]`

Évaluer et comparer les deux approches sur les datasets fournis :

| Critère | Ce qu'on mesure |
|---------|----------------|
| **Nombre de PFDs découvertes** | Couverture des règles |
| **Qualité des règles** | Confiance, interprétabilité |
| **Performance** | Temps d'exécution, nombre de candidats explorés |
| **Comparaison de LLMs** | Tester ≥ 2 LLMs (ex. Claude, Mistral) sur le workflow agentique |

---

## Livrables attendus

| # | Livrable | Contenu |
|---|----------|---------|
| 1 | **Code** | Tâche 1 (algo classique) + Tâche 2 (workflow agentique choisi) |
| 2 | **Rapport** | Description de l'approche, résultats expérimentaux, comparaison et discussion |
| 3 | **Démo (15 min)** | Présentation du système, choix de conception, résultats |

---

## État d'avancement

- [x] **Validation de PFDs** — `pfd_verifier.py` : fonctions `verifier_pfd` et `rapport_pfd`, matching `exact / startswith / contains / regex`, gestion NaN
- [x] **Démonstration** — 6 PFDs candidates vérifiées sur les 4 datasets réels
- [ ] Tâche 1 — Algorithme de découverte classique (extraction + génération + généralisation)
- [ ] Tâche 2 — Workflow agentique (Workflow A, B ou C)
- [ ] Tâche 3 — Expériences comparatives

---

## Structure du dépôt

```
.
├── pfd_verifier.py          # Validation de PFDs (base pour la Tâche 1)
├── Approximate_PFDs.pdf     # Cours de référence
└── data/
    ├── pfd_validation/
    │   ├── t1.csv           # Employés comté de Montgomery (9 101 lignes)
    │   ├── t2.csv           # Employeurs de Chicago (3 502 lignes)
    │   ├── t3.csv           # Licences d'alcool Maryland (1 077 lignes)
    │   └── US_Phone_Code.csv
    ├── CHE/
    │   ├── mechanism_refs.csv   (9 536 lignes)
    │   ├── metabolism_refs.csv
    │   ├── protein_classification.csv
    │   ├── research_companies.csv
    │   └── variant_sequences.csv
    └── DGOV/
        └── *.csv
```

---

## Installation

Python 3.8+ requis. Seul `pandas` est nécessaire pour `pfd_verifier.py` :

```bash
pip install pandas
```

Pour les workflows agentiques, installer le SDK du LLM choisi :

```bash
pip install anthropic   # Claude
pip install mistralai   # Mistral
```

## Lancer la démonstration de validation

```bash
python pfd_verifier.py
```

Exécute 6 tests sur les datasets réels et affiche un rapport pour chacun.

---

## API — `pfd_verifier.py`

### `verifier_pfd(...)`

| Paramètre | Type | Description |
|-----------|------|-------------|
| `df` | `DataFrame` | Données à analyser |
| `col_X` | `str` | Colonne antécédent |
| `pattern_X` | `any` | Pattern à matcher sur `col_X` |
| `col_Y` | `str` | Colonne conséquent |
| `pattern_Y` | `any` | Pattern à matcher sur `col_Y` |
| `epsilon` | `float` | Tolérance aux violations (défaut `0.05`) |
| `match_type` | `str` | Type de matching par défaut (`exact`, `startswith`, `contains`, `regex`) |
| `match_type_X` | `str\|None` | Type de matching pour `col_X` uniquement |
| `match_type_Y` | `str\|None` | Type de matching pour `col_Y` uniquement |

**Retourne** un `dict` avec : `confidence`, `is_valid`, `n_matching_X`, `n_valid`, `n_violations`, `epsilon`, `examples_violations`.

### `rapport_pfd(...)`

Mêmes paramètres. Affiche un rapport formaté et retourne le même `dict`.

---

## Résultats de la démonstration (validation)

| # | Dataset | PFD | Confiance | Valide (ε) |
|---|---------|-----|-----------|------------|
| 1 | t1 | `Department` → `Department Name` | 100% | Oui (ε=0.0) |
| 2 | t1 | `Gender='F'` → `Fulltime-Regular` | 81.1% | Non (ε=0.1) |
| 3 | t3 | `Zip=20852.0` → `ROCKVILLE` | 93.7% | Non (ε=0.05) |
| 4 | t3 | `Licensee sw 'BBW'` → `On Premise` | 95.1% | Oui (ε=0.1) |
| 5 | t2 | `ZIP sw '606'` → `Chicago` | 98.3% | Oui (ε=0.1) |
| 6 | CHE | `ref_type='PubMed'` → `sw europepmc` | 99.7% | Oui (ε=0.05) |

---

## Contribuer

1. Forkez le dépôt
2. Créez une branche : `git checkout -b feature/ma-fonctionnalite`
3. Committez vos changements : `git commit -m "feat: description"`
4. Ouvrez une Pull Request
