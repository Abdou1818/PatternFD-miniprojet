"""
pfd_verifier.py
Vérificateur de Pattern Functional Dependencies (PFDs) approximatives.

Une PFD est une règle de la forme :
    si col_X correspond au pattern P1  ->  col_Y doit correspondre au pattern P2

La version approximative tolère un taux d'exception epsilon :
    confiance = nb(X match ET Y match) / nb(X match)
    PFD valide si confiance >= (1 - epsilon)

Librairies utilisées : pandas, re (bibliothèque standard)
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Fonction interne de matching
# ---------------------------------------------------------------------------

def _match_pattern(series, pattern, match_type):
    """
    Applique un matching de pattern sur une Series pandas.

    Les valeurs NaN ne correspondent jamais à un pattern.

    Paramètres
    ----------
    series     : pd.Series
        Colonne à tester.
    pattern    : any
        Valeur ou expression à rechercher.
    match_type : str
        Type de matching :
          - 'exact'      : égalité stricte (series == pattern)
          - 'startswith' : la valeur (convertie en str) commence par pattern
          - 'contains'   : la valeur (convertie en str) contient pattern
          - 'regex'      : expression régulière appliquée sur la valeur en str

    Retourne
    --------
    pd.Series[bool]
        Masque booléen de même index que series.
    """
    if match_type == "exact":
        # NaN == X est False nativement dans pandas, pas besoin de notna()
        return series == pattern

    # Pour startswith / contains / regex : conversion en str obligatoire.
    # NaN devient la chaîne "nan" après astype(str) — on l'exclut via notna().
    not_null = series.notna()
    s = series.astype(str)
    pattern_str = str(pattern)

    if match_type == "startswith":
        return not_null & s.str.startswith(pattern_str)

    elif match_type == "contains":
        return not_null & s.str.contains(pattern_str, regex=False, na=False)

    elif match_type == "regex":
        # Validation de la syntaxe avant application pour un message d'erreur clair
        try:
            re.compile(pattern_str)
        except re.error as e:
            raise ValueError(f"Pattern regex invalide '{pattern_str}' : {e}")
        # Recherche de la regex n'importe où dans la valeur (str.contains)
        return not_null & s.str.contains(pattern_str, regex=True, na=False)

    else:
        raise ValueError(
            f"match_type inconnu : '{match_type}'. "
            "Valeurs acceptées : 'exact', 'startswith', 'contains', 'regex'."
        )


# ---------------------------------------------------------------------------
# Vérification principale
# ---------------------------------------------------------------------------

def verifier_pfd(df, col_X, pattern_X, col_Y, pattern_Y,
                 epsilon=0.05, match_type="exact",
                 match_type_X=None, match_type_Y=None):
    """
    Vérifie une PFD approximative sur un DataFrame pandas.

    La règle vérifiée est :
        col_X correspond à pattern_X  ->  col_Y correspond à pattern_Y

    Confiance = nb(X match ET Y match) / nb(X match)
    PFD valide si confiance >= (1 - epsilon)

    Paramètres
    ----------
    df           : pd.DataFrame
        Données à analyser.
    col_X        : str
        Nom de la colonne antécédent (source de la PFD).
    pattern_X    : any
        Pattern à matcher sur col_X.
    col_Y        : str
        Nom de la colonne conséquent (cible de la PFD).
    pattern_Y    : any
        Pattern à matcher sur col_Y.
    epsilon      : float, optionnel (défaut 0.05)
        Tolérance aux violations. 0.0 = FD exacte, 0.05 = 5% max de violations.
    match_type   : str, optionnel (défaut 'exact')
        Type de matching par défaut appliqué aux deux colonnes.
    match_type_X : str | None, optionnel
        Type de matching pour col_X uniquement (remplace match_type si fourni).
    match_type_Y : str | None, optionnel
        Type de matching pour col_Y uniquement (remplace match_type si fourni).

    Retourne
    --------
    dict avec les clés :
        confidence          float  score entre 0.0 et 1.0
        is_valid            bool   True si confiance >= (1 - epsilon)
        n_matching_X        int    nombre de lignes où col_X matche pattern_X
        n_valid             int    nombre de lignes où X ET Y matchent
        n_violations        int    lignes où X matche mais Y ne matche pas
        epsilon             float  seuil utilisé pour la décision
        examples_violations list   jusqu'à 3 lignes en violation (liste de dicts)
    """
    # Résolution des types de matching pour X et Y séparément
    mt_X = match_type_X if match_type_X is not None else match_type
    mt_Y = match_type_Y if match_type_Y is not None else match_type

    # --- Étape 1 : masque sur l'antécédent (col_X) ---
    mask_X = _match_pattern(df[col_X], pattern_X, mt_X)
    n_matching_X = int(mask_X.sum())

    # Cas dégénéré : aucune ligne ne correspond au pattern X
    if n_matching_X == 0:
        return {
            "confidence": 0.0,
            "is_valid": False,
            "n_matching_X": 0,
            "n_valid": 0,
            "n_violations": 0,
            "epsilon": epsilon,
            "examples_violations": [],
        }

    # --- Étape 2 : restreindre aux lignes où X matche ---
    df_X = df[mask_X].copy()

    # --- Étape 3 : masque sur le conséquent (col_Y) dans ce sous-ensemble ---
    mask_Y = _match_pattern(df_X[col_Y], pattern_Y, mt_Y)
    n_valid = int(mask_Y.sum())
    n_violations = n_matching_X - n_valid

    # --- Calcul de la confiance et décision ---
    confidence = n_valid / n_matching_X
    is_valid = confidence >= (1.0 - epsilon)

    # --- Collecte des 3 premières violations pour débogage ---
    violations_df = df_X[~mask_Y]
    examples_violations = violations_df.head(3).to_dict("records")

    return {
        "confidence": round(confidence, 6),
        "is_valid": is_valid,
        "n_matching_X": n_matching_X,
        "n_valid": n_valid,
        "n_violations": n_violations,
        "epsilon": epsilon,
        "examples_violations": examples_violations,
    }


# ---------------------------------------------------------------------------
# Rapport formaté
# ---------------------------------------------------------------------------

def rapport_pfd(df, col_X, pattern_X, col_Y, pattern_Y,
                epsilon=0.05, match_type="exact",
                match_type_X=None, match_type_Y=None):
    """
    Affiche un rapport lisible dans le terminal pour une PFD.

    Appelle verifier_pfd() et présente les résultats avec des séparateurs visuels.

    Paramètres
    ----------
    (identiques à verifier_pfd)

    Retourne
    --------
    dict identique à celui de verifier_pfd() pour usage programmatique.
    """
    res = verifier_pfd(
        df, col_X, pattern_X, col_Y, pattern_Y,
        epsilon=epsilon, match_type=match_type,
        match_type_X=match_type_X, match_type_Y=match_type_Y,
    )

    # Résolution des types pour l'affichage
    mt_X = match_type_X if match_type_X is not None else match_type
    mt_Y = match_type_Y if match_type_Y is not None else match_type

    statut = "VALIDE" if res["is_valid"] else "INVALIDE"
    seuil_min = 1.0 - epsilon

    print("=" * 68)
    print(f"  PFD : {col_X} [{mt_X}: {pattern_X!r}]")
    print(f"     -> {col_Y} [{mt_Y}: {pattern_Y!r}]")
    print(f"  Epsilon : {epsilon}  |  Seuil de confiance minimum : {seuil_min:.1%}")
    print("-" * 68)
    print(f"  Lignes où X matche (antécédent)    : {res['n_matching_X']:>8,}")
    print(f"  Lignes valides (X et Y matchent)   : {res['n_valid']:>8,}")
    print(f"  Violations (X matche, Y non)       : {res['n_violations']:>8,}")
    print(f"  Confiance calculée                 : {res['confidence']:>8.2%}")
    print(f"  Résultat                           :  {statut}")
    if res["examples_violations"]:
        print("-" * 68)
        print("  Exemples de violations (jusqu'a 3) :")
        for i, ex in enumerate(res["examples_violations"], start=1):
            val_x = ex.get(col_X, "N/A")
            val_y = ex.get(col_Y, "N/A")
            print(f"    [{i}]  {col_X} = {val_x!r}   |   {col_Y} = {val_y!r}")
    print("=" * 68)
    print()

    return res


# ---------------------------------------------------------------------------
# Script de démonstration sur les datasets réels
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Chemin de base vers le dossier data (relatif à ce fichier)
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    print("\n" + "#" * 68)
    print("#  DEMONSTRATION — Verification de PFDs sur donnees reelles")
    print("#" * 68 + "\n")

    # Chargement des quatre datasets
    t1 = pd.read_csv(os.path.join(BASE, "pfd_validation", "t1.csv"))
    t2 = pd.read_csv(os.path.join(BASE, "pfd_validation", "t2.csv"))
    t3 = pd.read_csv(os.path.join(BASE, "pfd_validation", "t3.csv"))
    mr = pd.read_csv(os.path.join(BASE, "CHE", "mechanism_refs.csv"))

    # -----------------------------------------------------------------------
    # TEST 1 — t1.csv : FD exacte Department -> Department Name (epsilon=0.0)
    #
    # Une FD couvre toutes les valeurs de la colonne source, pas un seul pattern.
    # On itère donc sur chaque valeur distincte de Department et on vérifie
    # que chaque Department code mappe vers un seul Department Name (confiance=1.0).
    # -----------------------------------------------------------------------
    print("=" * 68)
    print("  TEST 1 — FD exacte : Department -> Department Name (epsilon=0.0)")
    print("  Verification sur toutes les valeurs distinctes de Department")
    print("=" * 68)

    violations_totales = 0
    depts_invalides = []

    for dept in sorted(t1["Department"].dropna().unique()):
        # Nom de département le plus fréquent pour ce code (mode)
        noms = t1.loc[t1["Department"] == dept, "Department Name"].dropna()
        if noms.empty:
            continue
        nom_attendu = noms.value_counts().index[0]

        res = verifier_pfd(
            t1,
            col_X="Department", pattern_X=dept,
            col_Y="Department Name", pattern_Y=nom_attendu,
            epsilon=0.0, match_type="exact",
        )
        violations_totales += res["n_violations"]
        if not res["is_valid"]:
            depts_invalides.append(dept)

    nb_depts = t1["Department"].dropna().nunique()
    fd_valide = violations_totales == 0
    print(f"  Departements testes   : {nb_depts}")
    print(f"  Violations totales    : {violations_totales}")
    if fd_valide:
        print("  FD valide             : OUI — aucune violation (FD exacte confirmee)")
    else:
        print(f"  FD valide             : NON — dept(s) invalides : {depts_invalides}")
    print()

    # -----------------------------------------------------------------------
    # TEST 2 — t1.csv : Gender='F' -> Assignment Category='Fulltime-Regular'
    # epsilon=0.1 : on tolère jusqu'à 10% de femmes hors de cette catégorie
    # -----------------------------------------------------------------------
    print("TEST 2 — t1.csv : Gender='F' -> Assignment Category='Fulltime-Regular' (epsilon=0.1)")
    rapport_pfd(
        t1,
        col_X="Gender", pattern_X="F",
        col_Y="Assignment Category", pattern_Y="Fulltime-Regular",
        epsilon=0.1,
        match_type="exact",
    )

    # -----------------------------------------------------------------------
    # TEST 3 — t3.csv : Zip=20852.0 -> City='ROCKVILLE' (epsilon=0.05)
    # Zip est de type float64 : la comparaison exacte avec le float fonctionne.
    # -----------------------------------------------------------------------
    print("TEST 3 — t3.csv : Zip=20852.0 -> City='ROCKVILLE' (epsilon=0.05)")
    rapport_pfd(
        t3,
        col_X="Zip", pattern_X=20852.0,
        col_Y="City", pattern_Y="ROCKVILLE",
        epsilon=0.05,
        match_type="exact",
    )

    # -----------------------------------------------------------------------
    # TEST 4 — t3.csv : Licensee Number startswith 'BBW' -> Channel Type='On Premise'
    # X et Y utilisent des types de matching différents :
    #   match_type_X='startswith' pour le préfixe du numéro de licence
    #   match_type_Y='exact'      pour la valeur exacte du type de canal
    # -----------------------------------------------------------------------
    print("TEST 4 — t3.csv : Licensee Number sw 'BBW' -> Channel Type='On Premise' (epsilon=0.1)")
    rapport_pfd(
        t3,
        col_X="Licensee Number", pattern_X="BBW",
        col_Y="Channel Type", pattern_Y="On Premise",
        epsilon=0.1,
        match_type_X="startswith",
        match_type_Y="exact",
    )

    # -----------------------------------------------------------------------
    # TEST 5 — t2.csv : ZIP startswith '606' -> CITY='Chicago' (epsilon=0.1)
    # ZIP est de type str dans ce dataset, startswith s'applique directement.
    # -----------------------------------------------------------------------
    print("TEST 5 — t2.csv : ZIP startswith '606' -> CITY='Chicago' (epsilon=0.1)")
    rapport_pfd(
        t2,
        col_X="ZIP", pattern_X="606",
        col_Y="CITY", pattern_Y="Chicago",
        epsilon=0.1,
        match_type_X="startswith",
        match_type_Y="exact",
    )

    # -----------------------------------------------------------------------
    # TEST 6 — mechanism_refs.csv : ref_type='PubMed' -> ref_url sw 'http://europepmc.org'
    # X utilise 'exact', Y utilise 'startswith' pour vérifier le préfixe d'URL.
    # -----------------------------------------------------------------------
    print("TEST 6 — mechanism_refs.csv : ref_type='PubMed' -> ref_url sw 'http://europepmc.org' (epsilon=0.05)")
    rapport_pfd(
        mr,
        col_X="ref_type", pattern_X="PubMed",
        col_Y="ref_url", pattern_Y="http://europepmc.org",
        epsilon=0.05,
        match_type_X="exact",
        match_type_Y="startswith",
    )
