# Génération des candidats PFDs à partir des patterns extraits
# Pour chaque pattern sur col_X, on regarde la valeur la plus fréquente de col_Y

from collections import Counter
import pandas as pd


# Retourne (valeur_top, nb_occurrences, total_non_null) pour une Series
def _most_common_value(series):
    non_null = series.dropna()
    if non_null.empty:
        return None, 0, 0
    counts = Counter(non_null.astype(str) if non_null.dtype.kind == 'f' else non_null)
    top_val, top_count = counts.most_common(1)[0]
    if non_null.dtype.kind in ('f', 'i'):
        try:
            top_val = type(non_null.iloc[0])(top_val)
        except (ValueError, TypeError):
            pass
    return top_val, top_count, len(non_null)


# Génère les candidats pour une colonne source et ses patterns
# Pour chaque pattern P sur col_X et chaque col_Y, on crée un candidat
# avec la valeur la plus fréquente de Y dans le groupe
def generate_candidates(df, col_X, pattern_map, match_type, target_cols, min_support=5):
    # first_token est réalisé par startswith dans le verifier
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

            raw_conf = top_count / total
            candidates.append({
                "col_X": col_X,
                "pattern_X": pattern,
                "match_type_X": effective_mt_X,
                "col_Y": col_Y,
                "pattern_Y": top_val,
                "match_type_Y": "exact",
                "support": len(indices),
                "raw_confidence": round(raw_conf, 6),
            })

    return candidates


# Génère tous les candidats pour toutes les colonnes du DataFrame
# Dédoublonne les candidats identiques
def generate_all_candidates(df, pattern_maps, target_cols=None, min_support=5):
    if target_cols is None:
        target_cols = list(df.columns)

    all_candidates = []
    for col_X, type_map in pattern_maps.items():
        for match_type, pmap in type_map.items():
            batch = generate_candidates(df, col_X, pmap, match_type, target_cols, min_support)
            all_candidates.extend(batch)

    # dédoublonnage
    seen = set()
    unique = []
    for c in all_candidates:
        key = (c["col_X"], str(c["pattern_X"]), c["match_type_X"], c["col_Y"], str(c["pattern_Y"]))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique
