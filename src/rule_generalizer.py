"""
src/rule_generalizer.py
Généralisation des règles PFDs découvertes.

Deux étapes :
  1. Fusion : regrouper les règles spécifiques qui partagent (col_X, col_Y, val_Y)
              et remplacer leurs patterns par leur plus long préfixe commun,
              si la règle généralisée tient toujours avec le même epsilon.
  2. Élagage : supprimer les règles dominées (si P1 est préfixe de P2 et donne
               le même résultat, P1 est plus général et suffit).
"""

import sys
import os
from itertools import groupby

# Accès à pfd_verifier depuis la racine du projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pfd_verifier import verifier_pfd


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _longest_common_prefix(strings: list) -> str:
    """
    Calcule le plus long préfixe commun d'une liste de chaînes.

    Exemple
    -------
    >>> _longest_common_prefix(["606", "6066", "60601"])
    '606'
    """
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _merge_group(group: list, df, epsilon: float) -> list:
    """
    Tente de fusionner un groupe de règles partageant (col_X, col_Y, val_Y).

    Retourne une liste avec la règle généralisée si la fusion tient,
    ou la liste originale sinon.
    """
    col_X    = group[0]["col_X"]
    col_Y    = group[0]["col_Y"]
    val_Y    = group[0]["pattern_Y"]

    # Construire le préfixe commun de tous les pattern_X
    patterns_X = [str(r["pattern_X"]) for r in group]
    common     = _longest_common_prefix(patterns_X)

    if not common:
        return group     # aucun préfixe commun → pas de fusion

    # Tester la règle généralisée
    res = verifier_pfd(
        df,
        col_X=col_X,    pattern_X=common,
        col_Y=col_Y,    pattern_Y=val_Y,
        epsilon=epsilon,
        match_type_X="startswith",
        match_type_Y="exact",
    )

    if res["is_valid"]:
        generalized = {
            "col_X":              col_X,
            "pattern_X":          common,
            "match_type_X":       "startswith",
            "col_Y":              col_Y,
            "pattern_Y":          val_Y,
            "match_type_Y":       "exact",
            "confidence":         res["confidence"],
            "is_valid":           True,
            "n_matching_X":       res["n_matching_X"],
            "n_valid":            res["n_valid"],
            "n_violations":       res["n_violations"],
            "epsilon":            epsilon,
            "generalized_from":   len(group),
            "examples_violations": res["examples_violations"],
        }
        return [generalized]

    # Fusion impossible → conserver les règles originales
    return group


def fuse_rules(valid_pfds: list, df, epsilon: float = 0.05) -> list:
    """
    Regroupe et fusionne les règles spécifiques en règles plus générales.

    Algorithme
    ----------
    Pour chaque groupe (col_X, match_type_X = startswith, col_Y, pattern_Y) :
      - Calculer le plus long préfixe commun des pattern_X
      - Si la règle généralisée tient (confiance >= 1-epsilon) → fusionner
      - Sinon → conserver les règles individuelles

    Paramètres
    ----------
    valid_pfds : list[dict]  Sortie de validate_candidates.
    df         : pd.DataFrame
    epsilon    : float

    Retourne
    --------
    list[dict]
    """
    # Séparer les règles "startswith" (fusionnables) des autres
    sw_rules    = [r for r in valid_pfds if r.get("match_type_X") == "startswith"]
    other_rules = [r for r in valid_pfds if r.get("match_type_X") != "startswith"]

    result = list(other_rules)

    # Clé de groupement : col_X + col_Y + valeur_Y (chaîne)
    key_fn = lambda r: (r["col_X"], r["col_Y"], str(r["pattern_Y"]))
    sorted_sw = sorted(sw_rules, key=key_fn)

    for _, grp in groupby(sorted_sw, key=key_fn):
        group_list = list(grp)
        if len(group_list) == 1:
            result.append(group_list[0])
        else:
            result.extend(_merge_group(group_list, df, epsilon))

    return result


# ---------------------------------------------------------------------------
# Élagage des règles dominées
# ---------------------------------------------------------------------------

def prune_dominated(rules: list) -> list:
    """
    Supprime les règles dominées par une règle plus générale.

    Règle B est dominée par règle A si :
      - Même col_X, col_Y, pattern_Y et match_type_X = "startswith"
      - pattern_A est un préfixe strict de pattern_B
      (A couvre tous les cas de B et plus encore)

    Paramètres
    ----------
    rules : list[dict]

    Retourne
    --------
    list[dict]  Règles non dominées.
    """
    result = []

    for i, rule in enumerate(rules):
        if rule.get("match_type_X") != "startswith":
            result.append(rule)
            continue

        dominated = False
        for j, other in enumerate(rules):
            if i == j:
                continue
            if (
                other.get("match_type_X") == "startswith"
                and other["col_X"]      == rule["col_X"]
                and other["col_Y"]      == rule["col_Y"]
                and str(other["pattern_Y"]) == str(rule["pattern_Y"])
                and str(rule["pattern_X"]).startswith(str(other["pattern_X"]))
                and str(rule["pattern_X"]) != str(other["pattern_X"])
            ):
                dominated = True
                break

        if not dominated:
            result.append(rule)

    return result


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def generalize_rules(valid_pfds: list, df, epsilon: float = 0.05) -> list:
    """
    Applique la fusion puis l'élagage sur les règles valides.

    Paramètres
    ----------
    valid_pfds : list[dict]
    df         : pd.DataFrame
    epsilon    : float

    Retourne
    --------
    list[dict]  Règles généralisées et élagées.
    """
    step1 = fuse_rules(valid_pfds, df, epsilon)
    step2 = prune_dominated(step1)
    return step2
