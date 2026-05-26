# Généralisation des règles PFDs
# 1. Fusion : on regroupe les règles qui partagent (col_X, col_Y, val_Y)
#    et on les remplace par une seule règle avec le préfixe commun
# 2. Élagage : on supprime les règles dominées par une règle plus générale

import sys
import os
from itertools import groupby

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pfd_verifier import verifier_pfd


# Plus long préfixe commun d'une liste de chaînes
# ex: ["606", "6066", "60601"] -> "606"
def _longest_common_prefix(strings):
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


# Tente de fusionner un groupe de règles qui partagent le même (col_X, col_Y, val_Y)
# Si le préfixe commun donne une règle valide, on remplace tout le groupe
def _merge_group(group, df, epsilon):
    col_X = group[0]["col_X"]
    col_Y = group[0]["col_Y"]
    val_Y = group[0]["pattern_Y"]

    patterns_X = [str(r["pattern_X"]) for r in group]
    common = _longest_common_prefix(patterns_X)

    if not common:
        return group

    res = verifier_pfd(
        df,
        col_X=col_X, pattern_X=common,
        col_Y=col_Y, pattern_Y=val_Y,
        epsilon=epsilon,
        match_type_X="startswith",
        match_type_Y="exact",
    )

    if res["is_valid"]:
        return [{
            "col_X": col_X,
            "pattern_X": common,
            "match_type_X": "startswith",
            "col_Y": col_Y,
            "pattern_Y": val_Y,
            "match_type_Y": "exact",
            "confidence": res["confidence"],
            "is_valid": True,
            "n_matching_X": res["n_matching_X"],
            "n_valid": res["n_valid"],
            "n_violations": res["n_violations"],
            "epsilon": epsilon,
            "generalized_from": len(group),
            "examples_violations": res["examples_violations"],
        }]

    return group


# Regroupe et fusionne les règles startswith qui partagent le même triplet
def fuse_rules(valid_pfds, df, epsilon=0.05):
    sw_rules = [r for r in valid_pfds if r.get("match_type_X") == "startswith"]
    other_rules = [r for r in valid_pfds if r.get("match_type_X") != "startswith"]
    result = list(other_rules)

    key_fn = lambda r: (r["col_X"], r["col_Y"], str(r["pattern_Y"]))
    sorted_sw = sorted(sw_rules, key=key_fn)

    for _, grp in groupby(sorted_sw, key=key_fn):
        group_list = list(grp)
        if len(group_list) == 1:
            result.append(group_list[0])
        else:
            result.extend(_merge_group(group_list, df, epsilon))

    return result


# Supprime les règles dominées : si "6"->Chicago existe, "606"->Chicago est inutile
def prune_dominated(rules):
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
                and other["col_X"] == rule["col_X"]
                and other["col_Y"] == rule["col_Y"]
                and str(other["pattern_Y"]) == str(rule["pattern_Y"])
                and str(rule["pattern_X"]).startswith(str(other["pattern_X"]))
                and str(rule["pattern_X"]) != str(other["pattern_X"])
            ):
                dominated = True
                break

        if not dominated:
            result.append(rule)

    return result


# Point d'entrée : fusion puis élagage
def generalize_rules(valid_pfds, df, epsilon=0.05):
    step1 = fuse_rules(valid_pfds, df, epsilon)
    step2 = prune_dominated(step1)
    return step2
