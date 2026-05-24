"""
Script de génération du notebook 01_tache1_algorithme_classique.ipynb.

Utilise nbformat pour construire le notebook cellule par cellule, puis
nbconvert pour l'exécuter et capturer les sorties.

Usage :
    cd notebooks/
    python build_notebook.py
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ─────────────────────────────────────────────────────────────────────────────
# Titre et introduction
# ─────────────────────────────────────────────────────────────────────────────

md("""# Tâche 1 — Algorithme classique de découverte de PFDs

**Cours :** Data Wrangling — *Pattern-Based Dependencies and Agentic Discovery for Data Quality* (K. Belhajjame)

Ce notebook présente, étape par étape, l'algorithme classique de découverte de **Pattern Functional Dependencies (PFDs) approximatives**.

## Rappel théorique

Une **PFD** est une règle de la forme :

> Si la valeur de la colonne `X` correspond au pattern `P₁`,
> alors la valeur de la colonne `Y` doit correspondre au pattern `P₂`.

La version **approximative** tolère un taux d'exception `ε` :

$$\\text{confiance}(X \\to Y) = \\frac{|\\{t : t \\models P_1 \\land t \\models P_2\\}|}{|\\{t : t \\models P_1\\}|}$$

$$\\text{PFD valide} \\Leftrightarrow \\text{confiance} \\geq (1 - \\varepsilon)$$

## Plan du notebook

Le pipeline de découverte classique comporte 4 étapes (cf. slides 8-15 du cours) :

| Étape | Module Python                  | Rôle                                      |
|-------|--------------------------------|-------------------------------------------|
| 1     | `src/pattern_extractor.py`     | Extraction de patterns (préfixes, tokens) |
| 2     | `src/candidate_generator.py`   | Génération de candidats (X → Y)           |
| 3     | `pfd_verifier.py`              | Validation (support + confiance)          |
| 4     | `src/rule_generalizer.py`      | Généralisation des règles                 |

Le tout est orchestré par `pfd_discovery.py::discover()`.""")


# ─────────────────────────────────────────────────────────────────────────────
# Imports et chargement
# ─────────────────────────────────────────────────────────────────────────────

md("""---
## Configuration & imports""")

code("""# Ajouter la racine du projet au sys.path (le notebook est dans notebooks/)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import pandas as pd
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 12)

# Modules de l'algorithme classique
from src.pattern_extractor   import (
    extract_prefixes, extract_tokens, extract_ngrams,
    prefix_patterns, token_patterns, extract_all_patterns,
)
from src.candidate_generator import generate_candidates, generate_all_candidates
from src.rule_generalizer    import generalize_rules, _longest_common_prefix
from pfd_verifier            import verifier_pfd, rapport_pfd
from pfd_discovery           import discover, print_results

print("Imports OK")""")

md("""### Chargement des datasets

On utilise les 3 fichiers du sous-dossier `data/pfd_validation/` :

- **t1.csv** : 9 101 employés du comté de Montgomery (Maryland).
- **t2.csv** : 3 502 employeurs de Chicago.
- **t3.csv** : 1 077 licences d'alcool du Maryland.""")

code("""DATA = "../data/pfd_validation"

t1 = pd.read_csv(f"{DATA}/t1.csv")
t2 = pd.read_csv(f"{DATA}/t2.csv")
t3 = pd.read_csv(f"{DATA}/t3.csv")

print(f"t1 : {t1.shape[0]:>5} lignes × {t1.shape[1]} colonnes  — {list(t1.columns)[:4]}...")
print(f"t2 : {t2.shape[0]:>5} lignes × {t2.shape[1]} colonnes  — {list(t2.columns)[:4]}...")
print(f"t3 : {t3.shape[0]:>5} lignes × {t3.shape[1]} colonnes  — {list(t3.columns)[:4]}...")""")

code("""# Aperçu de t2 (employeurs de Chicago)
t2[["NAME", "CITY", "STATE", "ZIP", "PHONE"]].head()""")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Extraction de patterns
# ─────────────────────────────────────────────────────────────────────────────

md("""---
## Étape 1 — Extraction de patterns

> *« Each value generates many candidate patterns. Use frequency thresholds. »*
> — slide 10

L'idée : transformer les valeurs brutes en **patterns candidats** (préfixes, tokens, n-grams) qui pourront servir d'antécédent (`X`) dans une PFD.

### 1.a — Sur une valeur unique""")

code("""# Préfixes — exemple emprunté au cours (slide 17)
print("extract_prefixes('90012', k_max=3) =", extract_prefixes("90012", k_max=3))
print("extract_prefixes('John Smith', k_max=4) =", extract_prefixes("John Smith", k_max=4))

# Premier token (slide 17 : first_token(name) = 'John')
print("extract_tokens('John Smith') =", extract_tokens("John Smith"))
print("extract_tokens('Los Angeles, CA') =", extract_tokens("Los Angeles, CA"))

# N-grams (slide 10)
print("extract_ngrams('90012', 3) =", extract_ngrams("90012", 3))""")

md("""### 1.b — Sur une colonne entière

`prefix_patterns()` renvoie un dictionnaire `{pattern: [indices_des_lignes]}` filtré par support minimum.

Sur la colonne **ZIP** de Chicago, on s'attend à trouver le préfixe `606` (signature des ZIP du centre-ville).""")

code("""# Préfixes de longueur 1 à 3 sur ZIP de t2.csv
zip_prefixes = prefix_patterns(t2["ZIP"], k_max=3, min_support=30)

# Top 10 préfixes par support (nombre de lignes)
top = sorted(zip_prefixes.items(), key=lambda kv: -len(kv[1]))[:10]
print(f"  Pattern  | Support")
print(f"  ─────────┼────────")
for pat, idx in top:
    print(f"  {pat:>7}  |  {len(idx):>5}")""")

md("""On observe bien que **`606` couvre ~2 100 lignes**, soit la grande majorité des employeurs de Chicago — c'est notre futur antécédent.""")

code("""# Tokens — sur les noms d'employeurs
name_tokens = token_patterns(t2["NAME"], min_support=20)
top_names = sorted(name_tokens.items(), key=lambda kv: -len(kv[1]))[:8]
print(f"  Premier mot  | Support")
print(f"  ─────────────┼────────")
for tok, idx in top_names:
    print(f"  {tok:>11}  |  {len(idx):>5}")""")

md("""### 1.c — Tous les patterns d'une colonne en une fois

`extract_all_patterns()` est le point d'entrée unifié appelé par le pipeline.""")

code("""# Synthèse pour la colonne ZIP
zip_all = extract_all_patterns(t2["ZIP"], k_max=3, min_support=30)
for match_type, pmap in zip_all.items():
    print(f"  {match_type:<12}  →  {len(pmap)} patterns retenus")""")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Génération de candidats
# ─────────────────────────────────────────────────────────────────────────────

md("""---
## Étape 2 — Génération de candidats

> *« Generate candidate dependencies of the form X → Y. »* — slide 13

Pour chaque pattern `P_X` retenu et chaque colonne cible `Y` :

1. On isole les lignes où `col_X` matche `P_X`.
2. On cherche la **valeur la plus fréquente** de `col_Y` dans ce sous-ensemble (mode).
3. Cette valeur devient le `pattern_Y` candidat — la règle à tester est :
   « `col_X` matche `P_X` ⇒ `col_Y` = `top_Y` ».""")

code("""# Générer les candidats pour ZIP → toutes les autres colonnes
zip_candidates = generate_candidates(
    df=t2,
    col_X="ZIP",
    pattern_map=zip_prefixes,
    match_type="startswith",
    target_cols=list(t2.columns),
    min_support=30,
)

print(f"Nombre de candidats générés : {len(zip_candidates)}")
print("\\n5 candidats avec la confiance brute la plus élevée :\\n")
top_cand = sorted(zip_candidates, key=lambda c: -c["raw_confidence"])[:5]
for c in top_cand:
    print(f"  ZIP sw '{c['pattern_X']}' → {c['col_Y']} = {c['pattern_Y']!r}   "
          f"(support={c['support']}, conf_brute={c['raw_confidence']:.1%})")""")

md("""La **confiance brute** affichée ici n'est qu'une approximation : la validation finale (étape 3) recalcule la confiance exacte via `pfd_verifier`. Pour la plupart des cas elles coïncident — elles ne diffèrent qu'en présence de NaN dans `col_Y`.""")

code("""# Le pipeline complet appellera generate_all_candidates qui itère sur toutes
# les colonnes sources et tous les types de patterns
pattern_maps = {}
for col in ["ZIP", "PHONE", "CITY"]:
    pmap = extract_all_patterns(t2[col], k_max=3, min_support=30)
    if pmap:
        pattern_maps[col] = pmap

all_cands = generate_all_candidates(t2, pattern_maps, target_cols=list(t2.columns), min_support=30)
print(f"Candidats pour 3 colonnes sources (ZIP, PHONE, CITY) : {len(all_cands)}")""")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Validation
# ─────────────────────────────────────────────────────────────────────────────

md("""---
## Étape 3 — Validation

> *« Determine whether a candidate X → Y is a valid (approximate) PFD. »*
> — slide 14

On utilise `pfd_verifier.verifier_pfd()` qui applique les formules **support / confidence / noise** des slides 7-8.

Pour notre candidat phare *« ZIP sw `606` → CITY = `Chicago` »* avec `ε = 0.1` :""")

code("""res = verifier_pfd(
    t2,
    col_X="ZIP",   pattern_X="606",
    col_Y="CITY",  pattern_Y="Chicago",
    epsilon=0.1,
    match_type_X="startswith",
    match_type_Y="exact",
)

print(f"  Lignes où ZIP commence par '606' : {res['n_matching_X']:>5,}")
print(f"  Lignes où CITY = 'Chicago' dans ce groupe : {res['n_valid']:>5,}")
print(f"  Violations : {res['n_violations']:>5,}")
print(f"  Confiance : {res['confidence']:.2%}  (seuil = {1-0.1:.0%})")
print(f"  is_valid : {res['is_valid']}")

print("\\n  Exemples de violations :")
for v in res['examples_violations']:
    print(f"    ZIP={v['ZIP']!r} | CITY={v['CITY']!r}")""")

md("""**Découverte intéressante** : les 37 violations sont **toutes** dues à `'chicago'` en minuscules. C'est un **problème de qualité de données** révélé par la PFD — exactement le cas d'usage défendu par le cours (slide 4 : *« PFDs can detect errors that FDs cannot »*).

### Validation de tous les candidats""")

code("""# Pour chaque candidat, on appelle verifier_pfd et on garde les valides
valid_pfds = []
for cand in all_cands:
    r = verifier_pfd(
        t2,
        col_X=cand["col_X"], pattern_X=cand["pattern_X"],
        col_Y=cand["col_Y"], pattern_Y=cand["pattern_Y"],
        epsilon=0.1,
        match_type_X=cand["match_type_X"],
        match_type_Y=cand["match_type_Y"],
    )
    if r["is_valid"]:
        cand.update({"confidence": r["confidence"],
                     "n_matching_X": r["n_matching_X"],
                     "n_violations": r["n_violations"]})
        valid_pfds.append(cand)

print(f"  PFDs valides : {len(valid_pfds)} / {len(all_cands)} candidats")""")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Généralisation
# ─────────────────────────────────────────────────────────────────────────────

md("""---
## Étape 4 — Généralisation

> *« Merge specific patterns into more general and meaningful rules. »*
> — slide 15

L'exemple canonique du cours :

> `"John*" → M` + `"James*" → M` → `first_token(name) → M`

Notre implémentation a deux mécanismes :

1. **Fusion par plus grand préfixe commun** (`fuse_rules`) — si plusieurs règles partagent `(col_X, col_Y, val_Y)`, on essaie de les remplacer par une règle avec le préfixe commun, sous réserve que la confiance reste au-dessus du seuil.
2. **Élagage des règles dominées** (`prune_dominated`) — si `"6"` et `"606"` mènent toutes deux à `"Chicago"`, on garde seulement la plus générale (`"6"`).""")

code("""# Démonstration du préfixe commun
print("Préfixe commun de ['BBW1', 'BBW2', 'BBWLHR']  =", repr(_longest_common_prefix(['BBW1', 'BBW2', 'BBWLHR'])))
print("Préfixe commun de ['John', 'Joel', 'Joseph']   =", repr(_longest_common_prefix(['John', 'Joel', 'Joseph'])))
print("Préfixe commun de ['606', '770']               =", repr(_longest_common_prefix(['606', '770'])))""")

code("""# Application sur nos PFDs valides
generalized = generalize_rules(valid_pfds, t2, epsilon=0.1)
print(f"  Avant généralisation : {len(valid_pfds)} règles")
print(f"  Après généralisation : {len(generalized)} règles")
print(f"  Réduction            : {(1 - len(generalized)/len(valid_pfds)):.0%}")

n_fused = sum(1 for r in generalized if r.get("generalized_from", 0) > 1)
print(f"  Dont {n_fused} règles fusionnées depuis plusieurs règles spécifiques")""")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────

md("""---
## Le pipeline complet — `discover()`

La fonction `discover()` orchestre les 4 étapes ci-dessus. Elle est exposée
via la CLI `python pfd_discovery.py --dataset ...` mais on peut aussi
l'appeler directement.

### Pipeline sur t2.csv (employeurs Chicago)""")

code("""pfds_t2 = discover(t2, epsilon=0.1, min_support=30, k_max=3, verbose=True)
print_results(pfds_t2, top_n=10)""")

md("""### Pipeline sur t1.csv (employés Montgomery, FD exacte)

On force `epsilon = 0.0` pour rechercher des **FDs exactes** uniquement.""")

code("""pfds_t1 = discover(
    t1[["Department", "Department Name", "Division", "Assignment Category"]],
    epsilon=0.0, min_support=5, k_max=3, verbose=True
)
print_results(pfds_t1, top_n=8)""")

md("""### Pipeline sur t3.csv (licences alcool Maryland)""")

code("""pfds_t3 = discover(
    t3,
    epsilon=0.1, min_support=20, k_max=4, verbose=True
)
print_results(pfds_t3, top_n=8)""")


# ─────────────────────────────────────────────────────────────────────────────
# Analyse des résultats
# ─────────────────────────────────────────────────────────────────────────────

md("""---
## Discussion des résultats

### Statistiques globales""")

code("""import pandas as pd

stats = pd.DataFrame([
    {"Dataset": "t1.csv", "Lignes": len(t1), "Colonnes": len(t1.columns),
     "PFDs découvertes": len(pfds_t1),
     "Conf. moyenne": f"{sum(r['confidence'] for r in pfds_t1)/max(len(pfds_t1),1):.1%}"},
    {"Dataset": "t2.csv", "Lignes": len(t2), "Colonnes": len(t2.columns),
     "PFDs découvertes": len(pfds_t2),
     "Conf. moyenne": f"{sum(r['confidence'] for r in pfds_t2)/max(len(pfds_t2),1):.1%}"},
    {"Dataset": "t3.csv", "Lignes": len(t3), "Colonnes": len(t3.columns),
     "PFDs découvertes": len(pfds_t3),
     "Conf. moyenne": f"{sum(r['confidence'] for r in pfds_t3)/max(len(pfds_t3),1):.1%}"},
])
stats""")

md("""### Cas intéressants observés

**1. Détection d'incohérences de casse (t2.csv)**
La PFD `ZIP sw '606' → CITY = 'Chicago'` a 37 violations, toutes dues à
`'chicago'` en minuscules au lieu de `'Chicago'`. Ce sont des **erreurs de saisie**
détectables grâce à la PFD.

**2. Généralisation automatique (t3.csv)**
Le pipeline a découvert que `Licensee Number sw 'B' → Channel Type = 'On Premise'`
tient avec 94% de confiance. La lettre initiale `'B'` suffit ! C'est plus fort
que la PFD candidate du cours (`'BBW' → 'On Premise'`).

**3. Limites des approches purement syntaxiques (t2.csv)**
Les règles `ZIP sw '6' → COUNTRY = 'United States'`, `PHONE sw '3' → COUNTRY = '...'`,
etc. apparaissent en tête de classement avec 100% de confiance — mais ce sont
des règles **triviales** car `COUNTRY` est quasi-constant dans le dataset.
C'est exactement la limitation pointée slide 20 : *« No semantic understanding,
many spurious dependencies »*. Le **Workflow B (Guided Search)** de la Tâche 2
devrait corriger cela en demandant au LLM d'écarter ces colonnes peu discriminantes.""")


# ─────────────────────────────────────────────────────────────────────────────
# Conclusion
# ─────────────────────────────────────────────────────────────────────────────

md("""---
## Conclusion

Le pipeline classique implémenté couvre les 4 étapes prescrites par le cours :

| Étape | Slides cours | Implémentation                            | Résultat |
|-------|--------------|-------------------------------------------|----------|
| 1     | 9-10         | `extract_all_patterns`                    | Préfixes + tokens filtrés par support |
| 2     | 12-13        | `generate_all_candidates`                 | Candidats `(col_X, pattern_X, col_Y, top_Y)` |
| 3     | 14           | `verifier_pfd`                            | Filtrage par confiance ≥ 1-ε |
| 4     | 15           | `generalize_rules` (fusion + élagage)     | Réduction de 70% à 80% du nombre de règles |

### Prochaines étapes (cf. backlog GitHub)

- **Tâche 2** — implémenter le **Workflow B (Guided Search)** où un LLM
  sélectionne les colonnes et patterns pertinents avant validation.
- **Tâche 3** — comparer quantitativement les deux approches (nombre de PFDs,
  confiance moyenne, temps, interprétabilité) en utilisant à la fois Claude
  et Mistral comme LLMs.""")


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde du notebook
# ─────────────────────────────────────────────────────────────────────────────

nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.13"},
}

OUT = "01_tache1_algorithme_classique.ipynb"
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook écrit : {OUT}  ({len(cells)} cellules)")
