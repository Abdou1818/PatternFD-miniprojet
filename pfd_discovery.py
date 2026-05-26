# Pipeline de découverte de PFDs approximatives
# Enchaîne : extraction -> candidats -> validation -> généralisation
#
# Usage :
#   python pfd_discovery.py --dataset data/pfd_validation/t2.csv
#   python pfd_discovery.py --dataset data/pfd_validation/t1.csv --epsilon 0.0

import argparse
import csv
import os
import time
import pandas as pd

from src.pattern_extractor import extract_all_patterns
from src.candidate_generator import generate_all_candidates
from src.rule_generalizer import generalize_rules
from pfd_verifier import verifier_pfd


def discover(df, epsilon=0.05, min_support=10, k_max=5, verbose=True):
    t_start = time.time()
    cols = list(df.columns)

    # étape 1 : extraction des patterns
    if verbose:
        print(f"\n[1/4] Extraction des patterns (k_max={k_max}, min_support={min_support})...")
    pattern_maps = {}
    for col in cols:
        pmap = extract_all_patterns(df[col], k_max=k_max, min_support=min_support)
        if pmap:
            pattern_maps[col] = pmap

    total_patterns = sum(len(pm) for tm in pattern_maps.values() for pm in tm.values())
    if verbose:
        print(f"    {len(pattern_maps)} colonnes, {total_patterns} patterns")

    # étape 2 : génération des candidats
    if verbose:
        print(f"\n[2/4] Génération des candidats...")
    candidates = generate_all_candidates(df, pattern_maps, target_cols=cols, min_support=min_support)
    if verbose:
        print(f"    {len(candidates):,} candidats")

    if not candidates:
        if verbose:
            print("    Aucun candidat.")
        return []

    # étape 3 : validation
    if verbose:
        print(f"\n[3/4] Validation (epsilon={epsilon})...")
    valid_pfds = []
    for cand in candidates:
        res = verifier_pfd(
            df,
            col_X=cand["col_X"], pattern_X=cand["pattern_X"],
            col_Y=cand["col_Y"], pattern_Y=cand["pattern_Y"],
            epsilon=epsilon,
            match_type_X=cand["match_type_X"],
            match_type_Y=cand["match_type_Y"],
        )
        if res["is_valid"]:
            cand.update({
                "confidence": res["confidence"],
                "is_valid": True,
                "n_matching_X": res["n_matching_X"],
                "n_valid": res["n_valid"],
                "n_violations": res["n_violations"],
                "epsilon": epsilon,
                "examples_violations": res["examples_violations"],
            })
            valid_pfds.append(cand)

    if verbose:
        print(f"    {len(valid_pfds)} PFDs valides sur {len(candidates):,} testés")

    if not valid_pfds:
        return []

    # étape 4 : généralisation
    if verbose:
        print(f"\n[4/4] Généralisation...")
    generalized = generalize_rules(valid_pfds, df, epsilon=epsilon)
    if verbose:
        removed = len(valid_pfds) - len(generalized)
        fused = sum(1 for r in generalized if r.get("generalized_from", 0) > 1)
        print(f"    {len(generalized)} règles ({fused} généralisées, {removed} élagées)")

    generalized.sort(key=lambda r: (r.get("confidence", 0), r.get("n_matching_X", 0)), reverse=True)

    if verbose:
        print(f"\n  Temps total : {time.time() - t_start:.2f}s")

    return generalized


def print_results(pfds, top_n=None):
    if not pfds:
        print("  Aucune PFD.")
        return
    shown = pfds[:top_n] if top_n else pfds
    print(f"\n{'='*72}")
    print(f"  {'#':<4} {'Règle':<44} {'Conf':>6}  {'Support':>8}  {'Viol':>6}")
    print(f"{'='*72}")
    for i, r in enumerate(shown, 1):
        rule = f"{r['col_X']} [{r['match_type_X']}:'{r['pattern_X']}'] -> {r['col_Y']} = {r['pattern_Y']!r}"
        if len(rule) > 43:
            rule = rule[:40] + "..."
        flag = " *" if r.get("generalized_from", 0) > 1 else ""
        print(f"  {i:<4} {rule:<44} {r.get('confidence', 0):>5.1%}  {r.get('n_matching_X', 0):>8,}  {r.get('n_violations', 0):>6,}{flag}")
    print(f"{'='*72}")


def export_csv(pfds, path):
    if not pfds:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["col_X", "pattern_X", "match_type_X", "col_Y", "pattern_Y",
              "match_type_Y", "confidence", "n_matching_X", "n_valid",
              "n_violations", "epsilon", "generalized_from"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in pfds:
            row.setdefault("generalized_from", 1)
            writer.writerow(row)
    print(f"  Exporté -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Découverte de PFDs approximatives")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cols", nargs="+", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"\nChargement de : {args.dataset}")
    df = pd.read_csv(args.dataset)
    if args.cols:
        df = df[[c for c in args.cols if c in df.columns]]
    print(f"  {len(df):,} lignes × {len(df.columns)} colonnes : {list(df.columns)}")

    pfds = discover(df, epsilon=args.epsilon, min_support=args.min_support,
                    k_max=args.k_max, verbose=not args.quiet)
    print_results(pfds, top_n=args.top)

    out = args.output or os.path.join("results", f"discovered_pfds_{os.path.splitext(os.path.basename(args.dataset))[0]}.csv")
    export_csv(pfds, out)
