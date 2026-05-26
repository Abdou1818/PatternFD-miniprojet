# Mini-projet PFDs approximatives

Cours Data Wrangling — K. Belhajjame, Paris Dauphine.

## Le concept

Une PFD c'est une règle du type "si le ZIP commence par 606, alors la ville est Chicago".
Ça généralise les dépendances fonctionnelles classiques en travaillant sur des patterns
(préfixes, premiers mots, etc.) au lieu des valeurs entières.

"Approximative" = on tolère un petit % d'exceptions (paramètre epsilon).

## Lancer

```bash
pip install pandas
python pfd_discovery.py --dataset data/pfd_validation/t2.csv --epsilon 0.1 --top 15
```

## Structure

- `src/` : les 3 modules du pipeline (extraction, candidats, généralisation)
- `pfd_verifier.py` : vérification d'une PFD (support, confidence)
- `pfd_discovery.py` : pipeline complet
- `notebooks/` : exploration et résultats étape par étape
- `data/` : datasets (t1, t2, t3 + CHE + DGOV)
