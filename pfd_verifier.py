# Vérification d'une PFD approximative
# On teste si la règle "col_X matche pattern_X -> col_Y doit matcher pattern_Y"
# tient avec une confiance >= (1 - epsilon)

import re
import pandas as pd


# Applique un pattern sur une colonne et retourne un masque booléen
# Supporte : exact (==), startswith, contains, regex
def _match_pattern(series, pattern, match_type):
    if match_type == "exact":
        return series == pattern

    not_null = series.notna()
    s = series.astype(str)
    pattern_str = str(pattern)

    if match_type == "startswith":
        return not_null & s.str.startswith(pattern_str)
    elif match_type == "contains":
        return not_null & s.str.contains(pattern_str, regex=False, na=False)
    elif match_type == "regex":
        try:
            re.compile(pattern_str)
        except re.error as e:
            raise ValueError(f"Regex invalide '{pattern_str}' : {e}")
        return not_null & s.str.contains(pattern_str, regex=True, na=False)
    else:
        raise ValueError(f"match_type inconnu : '{match_type}'")


# Vérifie une PFD et retourne les métriques (confidence, support, violations, etc.)
def verifier_pfd(df, col_X, pattern_X, col_Y, pattern_Y,
                 epsilon=0.05, match_type="exact",
                 match_type_X=None, match_type_Y=None):
    mt_X = match_type_X if match_type_X is not None else match_type
    mt_Y = match_type_Y if match_type_Y is not None else match_type

    mask_X = _match_pattern(df[col_X], pattern_X, mt_X)
    n_matching_X = int(mask_X.sum())

    if n_matching_X == 0:
        return {
            "confidence": 0.0, "is_valid": False,
            "n_matching_X": 0, "n_valid": 0, "n_violations": 0,
            "epsilon": epsilon, "examples_violations": [],
        }

    df_X = df[mask_X].copy()
    mask_Y = _match_pattern(df_X[col_Y], pattern_Y, mt_Y)
    n_valid = int(mask_Y.sum())
    n_violations = n_matching_X - n_valid

    confidence = n_valid / n_matching_X
    is_valid = confidence >= (1.0 - epsilon)

    violations_df = df_X[~mask_Y]
    examples = violations_df.head(3).to_dict("records")

    return {
        "confidence": round(confidence, 6),
        "is_valid": is_valid,
        "n_matching_X": n_matching_X,
        "n_valid": n_valid,
        "n_violations": n_violations,
        "epsilon": epsilon,
        "examples_violations": examples,
    }


# Affiche un rapport lisible pour une PFD
def rapport_pfd(df, col_X, pattern_X, col_Y, pattern_Y,
                epsilon=0.05, match_type="exact",
                match_type_X=None, match_type_Y=None):
    res = verifier_pfd(
        df, col_X, pattern_X, col_Y, pattern_Y,
        epsilon=epsilon, match_type=match_type,
        match_type_X=match_type_X, match_type_Y=match_type_Y,
    )

    mt_X = match_type_X if match_type_X is not None else match_type
    mt_Y = match_type_Y if match_type_Y is not None else match_type
    statut = "VALIDE" if res["is_valid"] else "INVALIDE"

    print("=" * 60)
    print(f"  {col_X} [{mt_X}: {pattern_X!r}] -> {col_Y} [{mt_Y}: {pattern_Y!r}]")
    print(f"  epsilon={epsilon}  |  seuil={1.0 - epsilon:.1%}")
    print("-" * 60)
    print(f"  Support (X matche)  : {res['n_matching_X']:,}")
    print(f"  Valides (X et Y)    : {res['n_valid']:,}")
    print(f"  Violations          : {res['n_violations']:,}")
    print(f"  Confiance           : {res['confidence']:.2%}")
    print(f"  Résultat            : {statut}")
    if res["examples_violations"]:
        print("-" * 60)
        print("  Exemples de violations :")
        for i, ex in enumerate(res["examples_violations"], 1):
            print(f"    [{i}] {col_X}={ex.get(col_X, '?')!r}  |  {col_Y}={ex.get(col_Y, '?')!r}")
    print("=" * 60)
    print()
    return res
