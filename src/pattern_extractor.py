"""
src/pattern_extractor.py
Extraction de patterns depuis les colonnes d'un DataFrame.

Patterns supportés :
  - Préfixes de longueur 1 à k_max      (ex. "606" pour "60603")
  - Premier token (premier mot)          (ex. "John" pour "John Smith")
  - N-grams de caractères de longueur n  (ex. "900", "001" pour "90012")
"""

import re
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# Extraction sur une valeur unique
# ---------------------------------------------------------------------------

def extract_prefixes(value: str, k_max: int = 5) -> list:
    """
    Extrait les préfixes de longueur 1 à k_max d'une chaîne.

    Paramètres
    ----------
    value : str   Valeur source.
    k_max : int   Longueur maximale (défaut 5).

    Retourne
    --------
    list[str]  Ex. : extract_prefixes("ABCDE", 3) -> ["A", "AB", "ABC"]
    """
    if not isinstance(value, str) or not value:
        return []
    return [value[:k] for k in range(1, min(k_max + 1, len(value) + 1))]


def extract_tokens(value: str) -> list:
    """
    Extrait les tokens (mots) d'une chaîne.
    Séparateurs : espaces, tirets, underscores, slash, virgule, point-virgule.

    Paramètres
    ----------
    value : str

    Retourne
    --------
    list[str]  Ex. : extract_tokens("John Smith") -> ["John", "Smith"]
    """
    if not isinstance(value, str) or not value:
        return []
    tokens = re.split(r'[\s\-_/,.;:]+', value.strip())
    return [t for t in tokens if t]


def extract_ngrams(value: str, n: int = 3) -> list:
    """
    Extrait les n-grams de caractères consécutifs d'une chaîne.

    Paramètres
    ----------
    value : str
    n     : int  Longueur du n-gram (défaut 3).

    Retourne
    --------
    list[str]  Ex. : extract_ngrams("ABCDE", 3) -> ["ABC", "BCD", "CDE"]
    """
    if not isinstance(value, str) or len(value) < n:
        return []
    return [value[i:i + n] for i in range(len(value) - n + 1)]


# ---------------------------------------------------------------------------
# Extraction sur une Series entière  →  pattern → [row_indices]
# ---------------------------------------------------------------------------

def prefix_patterns(series: pd.Series, k_max: int = 5,
                    min_support: int = 5) -> dict:
    """
    Calcule, pour chaque préfixe de longueur 1 à k_max, les indices des lignes
    dont la valeur commence par ce préfixe.

    Les valeurs NaN sont ignorées. Les colonnes numériques sont converties en str.

    Paramètres
    ----------
    series      : pd.Series
    k_max       : int  Longueur maximale des préfixes (défaut 5).
    min_support : int  Nombre minimum de lignes pour conserver un pattern.

    Retourne
    --------
    dict[str, list]  pattern -> liste d'index de lignes.

    Exemple
    -------
    >>> s = pd.Series(["60601", "60602", "77001"])
    >>> prefix_patterns(s, k_max=3, min_support=2)
    {'6': [0, 1], '60': [0, 1], '606': [0, 1]}
    """
    buckets: dict = defaultdict(list)

    for idx, val in series.items():
        if pd.isna(val):
            continue
        s = str(val)
        # Éviter les doublons si la même valeur apparaît plusieurs fois avec les mêmes préfixes
        for k in range(1, min(k_max + 1, len(s) + 1)):
            buckets[s[:k]].append(idx)

    return {p: idxs for p, idxs in buckets.items() if len(idxs) >= min_support}


def token_patterns(series: pd.Series, min_support: int = 5) -> dict:
    """
    Calcule, pour chaque premier token, les indices des lignes dont la valeur
    commence par ce token.

    Utile pour des colonnes comme "Full Name" (→ prénom) ou "Position Title".

    Paramètres
    ----------
    series      : pd.Series
    min_support : int

    Retourne
    --------
    dict[str, list]  premier_token -> liste d'index de lignes.

    Exemple
    -------
    >>> s = pd.Series(["John Smith", "John Brown", "Susan Miller"])
    >>> token_patterns(s, min_support=2)
    {'John': [0, 1]}
    """
    buckets: dict = defaultdict(list)

    for idx, val in series.items():
        if pd.isna(val):
            continue
        tokens = extract_tokens(str(val))
        if tokens:
            buckets[tokens[0]].append(idx)

    return {t: idxs for t, idxs in buckets.items() if len(idxs) >= min_support}


def exact_patterns(series: pd.Series, min_support: int = 5) -> dict:
    """
    Calcule, pour chaque valeur exacte, les indices des lignes correspondantes.

    Paramètres
    ----------
    series      : pd.Series
    min_support : int

    Retourne
    --------
    dict[any, list]  valeur_exacte -> liste d'index de lignes.
    """
    buckets: dict = defaultdict(list)

    for idx, val in series.items():
        if pd.isna(val):
            continue
        buckets[val].append(idx)

    return {v: idxs for v, idxs in buckets.items() if len(idxs) >= min_support}


def extract_all_patterns(series: pd.Series, k_max: int = 5,
                         min_support: int = 5) -> dict:
    """
    Extrait tous les types de patterns pour une Series.

    Retourne
    --------
    dict[str, dict]  match_type -> {pattern: [row_indices]}
      Clés : "startswith" (préfixes), "first_token" (premiers tokens), "exact"
    """
    result = {}

    pref = prefix_patterns(series, k_max=k_max, min_support=min_support)
    if pref:
        result["startswith"] = pref

    toks = token_patterns(series, min_support=min_support)
    if toks:
        result["first_token"] = toks

    return result
