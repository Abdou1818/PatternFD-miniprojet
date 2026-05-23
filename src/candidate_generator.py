"""
src/candidate_generator.py
Génération des candidats PFDs à partir des patterns extraits.

Stratégie :
  Pour chaque pattern P_X sur col_X (support >= min_support) et chaque col_Y :
    1. Isoler les lignes où col_X "matche" P_X.
    2. Trouver la valeur la plus fréquente de col_Y dans ce groupe (mode).
    3. Créer le candidat (col_X, P_X, match_type_X, col_Y, top_val_Y, "exact").
  La validation (confiance réelle) est faite séparément par pfd_verifier.
"""

from collections import Counter, defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _most_common_value(series: pd.Series):
    """
    Retourne (valeur_la_plus_fréquente, nb_occurrences, total_non_null).

    Paramètres
    ----------
    series : pd.Series  Sous-ensemble de la colonne cible.

    Retourne
    --------
    tuple (top_val, top_count, total)
    """
    non_null = series.dropna()
    if non_null.empty:
        return None, 0, 0
    counts = Counter(non_null.astype(str) if non_null.dtype.kind == 'f' else non_null)
    top_val, top_count = counts.most_common(1)[0]
    # Convertir la clé str en valeur originale si la colonne était numérique
    if non_null.dtype.kind in ('f', 'i'):
        try:
            top_val = type(non_null.iloc[0])(top_val)
        except (ValueError, TypeError):
            pass
    return top_val, top_count, len(non_null)


# ---------------------------------------------------------------------------
# Génération pour une colonne source
# ---------------------------------------------------------------------------

def generate_candidates(df: pd.DataFrame, col_X: str,
                        pattern_map: dict, match_type: str,
                        target_cols: list, min_support: int = 5) -> list:
    """
    Génère les candidats PFDs pour une colonne source et ses patterns.

    Pour chaque (pattern_X, col_Y) :
      - Sous-ensemble = lignes indexées par pattern_map[pattern_X]
      - top_Y = valeur la plus fréquente de col_Y dans ce sous-ensemble
      - Candidat = (col_X, pattern_X, match_type, col_Y, top_Y, "exact")

    Paramètres
    ----------
    df          : pd.DataFrame
    col_X       : str   Colonne source.
    pattern_map : dict  {pattern: [row_indices]}
    match_type  : str   Type de matching X ("startswith", "first_token", "exact").
    target_cols : list  Colonnes cibles à tester.
    min_support : int   Support minimum (double vérification).

    Retourne
    --------
    list[dict]  Candidats avec clés : col_X, pattern_X, match_type_X,
                col_Y, pattern_Y, match_type_Y, support, raw_confidence.
    """
    # "first_token" est réalisé par "startswith" dans pfd_verifier
    effective_mt_X = "startswith" if match_type == "first_token" else match_type

    candidates = []

    for pattern, indices in pattern_map.items():
        if len(indices) < min_support:
            continue

        df_group = df.loc[indices]

        for col_Y in target_cols:
            if col_Y == col_X:
                continue

            top_val, top_count, total = _most_common_value(df_group[col_Y])

            if top_val is None or total == 0:
                continue

            # Confiance brute (avant validation exacte via pfd_verifier)
            raw_conf = top_count / total

            candidates.append({
                "col_X":         col_X,
                "pattern_X":     pattern,
                "match_type_X":  effective_mt_X,
                "col_Y":         col_Y,
                "pattern_Y":     top_val,
                "match_type_Y":  "exact",
                "support":       len(indices),
                "raw_confidence": round(raw_conf, 6),
            })

    return candidates


# ---------------------------------------------------------------------------
# Génération pour toutes les colonnes d'un DataFrame
# ---------------------------------------------------------------------------

def generate_all_candidates(df: pd.DataFrame, pattern_maps: dict,
                             target_cols: list = None,
                             min_support: int = 5) -> list:
    """
    Génère tous les candidats PFDs pour un DataFrame.

    Paramètres
    ----------
    df           : pd.DataFrame
    pattern_maps : dict
        Structure : {col_name: {match_type: {pattern: [row_indices]}}}
        Produit par src.pattern_extractor.extract_all_patterns pour chaque col.
    target_cols  : list | None  Colonnes cibles (None = toutes).
    min_support  : int

    Retourne
    --------
    list[dict]  Liste dédoublonnée de candidats.
    """
    if target_cols is None:
        target_cols = list(df.columns)

    all_candidates = []

    for col_X, type_map in pattern_maps.items():
        for match_type, pmap in type_map.items():
            batch = generate_candidates(
                df, col_X, pmap, match_type, target_cols, min_support
            )
            all_candidates.extend(batch)

    # Dédoublonner : même (col_X, pattern_X, match_type_X, col_Y, pattern_Y)
    seen = set()
    unique = []
    for c in all_candidates:
        key = (
            c["col_X"],
            str(c["pattern_X"]),
            c["match_type_X"],
            c["col_Y"],
            str(c["pattern_Y"]),
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique
