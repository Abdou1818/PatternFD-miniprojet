"""
pfd_discovery.py
Pipeline de découverte classique de PFDs approximatives.

Étapes :
  1. Extraction de patterns (préfixes + premier token) par colonne
  2. Génération des candidats PFDs (col_X × pattern_X × col_Y)
  3. Validation : calcul de la confiance via pfd_verifier.verifier_pfd
  4. Généralisation : fusion et élagage des règles

Usage :
    python pfd_discovery.py --dataset data/pfd_validation/t2.csv
    python pfd_discovery.py --dataset data/pfd_validation/t1.csv --epsilon 0.0 --min-support 5
    python pfd_discovery.py --dataset data/pfd_validation/t3.csv --cols Zip City --epsilon 0.1
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd

# Modules de l'algo classique
from src.pattern_extractor   import extract_all_patterns
from src.candidate_generator import generate_all_candidates
from src.rule_generalizer    import generalize_rules
from pfd_verifier            import verifier_pfd


# ---------------------------------------------------------------------------
# Découverte
# ---------------------------------------------------------------------------

def discover(df: pd.DataFrame, epsilon: float = 0.05,
             min_support: int = 10, k_max: int = 5,
             verbose: bool = True) -> list:
    """
    Pipeline complet de découverte de PFDs approximatives.

    Paramètres
    ----------
    df          : pd.DataFrame  Données à analyser.
    epsilon     : float         Tolérance aux violations (défaut 0.05).
    min_support : int           Nombre minimum de lignes pour un pattern.
    k_max       : int           Longueur maximale des préfixes.
    verbose     : bool          Affiche les étapes dans le terminal.

    Retourne
    --------
    list[dict]  PFDs valides, triées par confiance décroissante.
                Chaque dict contient : col_X, pattern_X, match_type_X,
                col_Y, pattern_Y, match_type_Y, confidence, n_matching_X,
                n_valid, n_violations, epsilon, generalized_from.
    """
    t_start = time.time()
    cols = list(df.columns)

    # ── Étape 1 : Extraction des patterns ────────────────────────────────
    if verbose:
        print(f"\n[1/4] Extraction des patterns  "
              f"(k_max={k_max}, min_support={min_support})...")

    pattern_maps = {}   # {col: {match_type: {pattern: [indices]}}}

    for col in cols:
        pmap = extract_all_patterns(df[col], k_max=k_max, min_support=min_support)
        if pmap:
            pattern_maps[col] = pmap

    total_patterns = sum(
        len(pmap)
        for tmaps in pattern_maps.values()
        for pmap in tmaps.values()
    )
    if verbose:
        print(f"    {len(pattern_maps)} colonnes analysées, "
              f"{total_patterns} patterns extraits")

    # ── Étape 2 : Génération des candidats ───────────────────────────────
    if verbose:
        print(f"\n[2/4] Génération des candidats PFDs...")

    candidates = generate_all_candidates(
        df, pattern_maps, target_cols=cols, min_support=min_support
    )

    if verbose:
        print(f"    {len(candidates):,} candidats générés")

    if not candidates:
        if verbose:
            print("    Aucun candidat — essayez de réduire --min-support.")
        return []

    # ── Étape 3 : Validation (confiance réelle) ───────────────────────────
    if verbose:
        print(f"\n[3/4] Validation (epsilon={epsilon})...")

    valid_pfds = []
    for cand in candidates:
        res = verifier_pfd(
            df,
            col_X=cand["col_X"],        pattern_X=cand["pattern_X"],
            col_Y=cand["col_Y"],        pattern_Y=cand["pattern_Y"],
            epsilon=epsilon,
            match_type_X=cand["match_type_X"],
            match_type_Y=cand["match_type_Y"],
        )
        if res["is_valid"]:
            cand.update({
                "confidence":         res["confidence"],
                "is_valid":           True,
                "n_matching_X":       res["n_matching_X"],
                "n_valid":            res["n_valid"],
                "n_violations":       res["n_violations"],
                "epsilon":            epsilon,
                "examples_violations": res["examples_violations"],
            })
            valid_pfds.append(cand)

    if verbose:
        print(f"    {len(valid_pfds)} PFDs valides "
              f"(sur {len(candidates):,} candidats testés)")

    if not valid_pfds:
        return []

    # ── Étape 4 : Généralisation ──────────────────────────────────────────
    if verbose:
        print(f"\n[4/4] Généralisation des règles...")

    generalized = generalize_rules(valid_pfds, df, epsilon=epsilon)

    if verbose:
        removed = len(valid_pfds) - len(generalized)
        fused   = sum(1 for r in generalized if r.get("generalized_from", 0) > 1)
        print(f"    {len(generalized)} règles conservées "
              f"({fused} généralisées, {removed} élagées/fusionnées)")

    # Tri par confiance décroissante, puis support décroissant
    generalized.sort(
        key=lambda r: (r.get("confidence", 0), r.get("n_matching_X", 0)),
        reverse=True
    )

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n  Temps total : {elapsed:.2f}s")

    return generalized


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def print_results(pfds: list, top_n: int = None):
    """
    Affiche les PFDs découvertes sous forme de tableau dans le terminal.
    """
    if not pfds:
        print("  Aucune PFD découverte.")
        return

    shown = pfds[:top_n] if top_n else pfds
    W = 72

    print(f"\n{'='*W}")
    print(f"  {'#':<4} {'Règle PFD':<44} {'Conf':>6}  {'Support':>8}  {'Viol':>6}")
    print(f"{'='*W}")

    for i, r in enumerate(shown, 1):
        rule = (f"{r['col_X']} [{r['match_type_X']}:'{r['pattern_X']}']"
                f" -> {r['col_Y']} = {r['pattern_Y']!r}")
        if len(rule) > 43:
            rule = rule[:40] + "..."
        flag = " *" if r.get("generalized_from", 0) > 1 else ""
        print(f"  {i:<4} {rule:<44} "
              f"{r.get('confidence', 0):>5.1%}  "
              f"{r.get('n_matching_X', 0):>8,}  "
              f"{r.get('n_violations', 0):>6,}{flag}")

    print(f"{'='*W}")
    if any(r.get("generalized_from", 0) > 1 for r in shown):
        print("  * règle généralisée depuis plusieurs règles spécifiques")
    print()


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

def export_csv(pfds: list, path: str):
    """
    Exporte les PFDs découvertes dans un fichier CSV.
    """
    if not pfds:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fields = [
        "col_X", "pattern_X", "match_type_X",
        "col_Y", "pattern_Y", "match_type_Y",
        "confidence", "n_matching_X", "n_valid",
        "n_violations", "epsilon", "generalized_from",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in pfds:
            row.setdefault("generalized_from", 1)
            writer.writerow(row)

    print(f"  Résultats exportés -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Découverte classique de PFDs approximatives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",     required=True,
                   help="Chemin vers le fichier CSV.")
    p.add_argument("--epsilon",     type=float, default=0.05,
                   help="Tolérance aux violations.")
    p.add_argument("--min-support", type=int,   default=10,
                   help="Support minimum par pattern.")
    p.add_argument("--k-max",       type=int,   default=5,
                   help="Longueur maximale des préfixes.")
    p.add_argument("--top",         type=int,   default=None,
                   help="Afficher les N meilleures règles.")
    p.add_argument("--output",      default=None,
                   help="Fichier CSV de sortie (défaut: results/discovered_pfds_<dataset>.csv).")
    p.add_argument("--cols",        nargs="+",  default=None,
                   help="Colonnes à analyser (défaut: toutes).")
    p.add_argument("--quiet",       action="store_true",
                   help="Désactive les messages de progression.")
    return p


def main():
    args = _build_parser().parse_args()

    print(f"\nChargement de : {args.dataset}")
    df = pd.read_csv(args.dataset)
    if args.cols:
        df = df[[c for c in args.cols if c in df.columns]]
    print(f"  {len(df):,} lignes × {len(df.columns)} colonnes : {list(df.columns)}")

    pfds = discover(
        df,
        epsilon=args.epsilon,
        min_support=args.min_support,
        k_max=args.k_max,
        verbose=not args.quiet,
    )

    print_results(pfds, top_n=args.top)

    # Export CSV
    if args.output:
        out_path = args.output
    else:
        name = os.path.splitext(os.path.basename(args.dataset))[0]
        out_path = os.path.join("results", f"discovered_pfds_{name}.csv")

    export_csv(pfds, out_path)


if __name__ == "__main__":
    main()
