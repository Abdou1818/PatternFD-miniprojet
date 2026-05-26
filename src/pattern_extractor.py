# Extraction de patterns depuis les colonnes d'un DataFrame
# On extrait des préfixes, premiers mots et n-grams pour chaque valeur

import re
from collections import defaultdict
import pandas as pd


# -- Fonctions sur une seule valeur --

# Retourne les préfixes de longueur 1 à k_max
# ex: extract_prefixes("ABCDE", 3) -> ["A", "AB", "ABC"]
def extract_prefixes(value, k_max=5):
    if not isinstance(value, str) or not value:
        return []
    return [value[:k] for k in range(1, min(k_max + 1, len(value) + 1))]


# Découpe une chaîne en mots (espaces, tirets, virgules, etc.)
# ex: extract_tokens("John Smith") -> ["John", "Smith"]
def extract_tokens(value):
    if not isinstance(value, str) or not value:
        return []
    tokens = re.split(r'[\s\-_/,.;:]+', value.strip())
    return [t for t in tokens if t]


# Extrait les sous-chaînes de longueur n
# ex: extract_ngrams("90012", 3) -> ["900", "001", "012"]
def extract_ngrams(value, n=3):
    if not isinstance(value, str) or len(value) < n:
        return []
    return [value[i:i + n] for i in range(len(value) - n + 1)]


# -- Fonctions sur une colonne entière -> {pattern: [indices]} --

# Tous les préfixes d'une colonne, filtrés par support minimum
# Les NaN sont ignorés, les numériques convertis en str
def prefix_patterns(series, k_max=5, min_support=5):
    buckets = defaultdict(list)
    for idx, val in series.items():
        if pd.isna(val):
            continue
        s = str(val)
        for k in range(1, min(k_max + 1, len(s) + 1)):
            buckets[s[:k]].append(idx)
    return {p: idxs for p, idxs in buckets.items() if len(idxs) >= min_support}


# Premier mot de chaque valeur, filtré par support
def token_patterns(series, min_support=5):
    buckets = defaultdict(list)
    for idx, val in series.items():
        if pd.isna(val):
            continue
        tokens = extract_tokens(str(val))
        if tokens:
            buckets[tokens[0]].append(idx)
    return {t: idxs for t, idxs in buckets.items() if len(idxs) >= min_support}


# Valeurs exactes, filtrées par support
def exact_patterns(series, min_support=5):
    buckets = defaultdict(list)
    for idx, val in series.items():
        if pd.isna(val):
            continue
        buckets[val].append(idx)
    return {v: idxs for v, idxs in buckets.items() if len(idxs) >= min_support}


# Point d'entrée : extrait préfixes + premiers mots
# Retourne {"startswith": {pattern: [indices]}, "first_token": {pattern: [indices]}}
def extract_all_patterns(series, k_max=5, min_support=5):
    result = {}
    pref = prefix_patterns(series, k_max=k_max, min_support=min_support)
    if pref:
        result["startswith"] = pref
    toks = token_patterns(series, min_support=min_support)
    if toks:
        result["first_token"] = toks
    return result
