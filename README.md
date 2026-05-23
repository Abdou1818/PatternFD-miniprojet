# Mini-projet — Pattern Functional Dependencies (PFDs) approximatives

Projet universitaire de **Data Wrangling** : implémentation d'un vérificateur de PFDs approximatives sur des données réelles.

## Concept

Une **Pattern Functional Dependency (PFD)** est une règle de la forme :

> Si la valeur de la colonne `X` correspond au pattern `P1`,  
> alors la valeur de la colonne `Y` doit correspondre au pattern `P2`.

La version **approximative** tolère un taux d'exception `epsilon` :

```
confiance = nb(X match ET Y match) / nb(X match)
PFD valide  ⟺  confiance ≥ (1 − epsilon)
```

## Structure du dépôt

```
.
├── pfd_verifier.py          # Vérificateur de PFDs (module principal)
├── Approximate_PFDs.pdf     # Article de référence
└── data/
    ├── pfd_validation/
    │   ├── t1.csv           # Employés du comté de Montgomery (9 101 lignes)
    │   ├── t2.csv           # Employeurs de Chicago (3 502 lignes)
    │   ├── t3.csv           # Licences d'alcool Maryland (1 077 lignes)
    │   └── US_Phone_Code.csv
    ├── CHE/
    │   ├── mechanism_refs.csv
    │   ├── metabolism_refs.csv
    │   ├── protein_classification.csv
    │   ├── research_companies.csv
    │   └── variant_sequences.csv
    └── DGOV/
        └── *.csv
```

## Installation

Python 3.8+ requis. Seul `pandas` est nécessaire :

```bash
pip install pandas
```

## Utilisation

### Lancer la démonstration

```bash
python pfd_verifier.py
```

Exécute 6 tests sur les datasets réels et affiche un rapport pour chacun.

### Utiliser le module dans votre code

```python
import pandas as pd
from pfd_verifier import verifier_pfd, rapport_pfd

df = pd.read_csv("data/pfd_validation/t3.csv")

# Vérification programmatique
result = verifier_pfd(
    df,
    col_X="Zip",        pattern_X=20852.0,
    col_Y="City",       pattern_Y="ROCKVILLE",
    epsilon=0.05,
    match_type="exact"
)
print(result["confidence"])   # 0.936842
print(result["is_valid"])     # False

# Rapport formaté dans le terminal
rapport_pfd(
    df,
    col_X="Licensee Number", pattern_X="BBW",
    col_Y="Channel Type",    pattern_Y="On Premise",
    epsilon=0.1,
    match_type_X="startswith",
    match_type_Y="exact"
)
```

## API

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

Mêmes paramètres que `verifier_pfd`. Affiche un rapport formaté et retourne le même `dict`.

## Résultats de la démonstration

| # | Dataset | PFD | Confiance | Valide (ε) |
|---|---------|-----|-----------|------------|
| 1 | t1 | `Department` → `Department Name` | 100% | Oui (ε=0.0) |
| 2 | t1 | `Gender='F'` → `Fulltime-Regular` | 81.1% | Non (ε=0.1) |
| 3 | t3 | `Zip=20852.0` → `ROCKVILLE` | 93.7% | Non (ε=0.05) |
| 4 | t3 | `Licensee sw 'BBW'` → `On Premise` | 95.1% | Oui (ε=0.1) |
| 5 | t2 | `ZIP sw '606'` → `Chicago` | 98.3% | Oui (ε=0.1) |
| 6 | CHE | `ref_type='PubMed'` → `sw europepmc` | 99.7% | Oui (ε=0.05) |

## Contribuer

1. Forkez le dépôt
2. Créez une branche : `git checkout -b feature/ma-fonctionnalite`
3. Committez vos changements : `git commit -m "feat: description"`
4. Ouvrez une Pull Request

## Référence

Voir `Approximate_PFDs.pdf` pour la définition formelle des PFDs approximatives.
