"""
tests/test_pipeline.py
Tests d'intégration du pipeline complet et non-régression de pfd_verifier.
"""

import os
import pytest
import pandas as pd
from pfd_discovery import discover
from pfd_verifier  import verifier_pfd

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "pfd_validation")


# ---------------------------------------------------------------------------
# Tests du pipeline discover()
# ---------------------------------------------------------------------------

class TestDiscover:

    def test_discover_t2_finds_zip_chicago(self):
        """Pipeline classique : t2.csv doit trouver ZIP sw '606' -> CITY='Chicago'."""
        t2 = pd.read_csv(os.path.join(DATA, "t2.csv"))
        pfds = discover(t2[["ZIP", "CITY"]], epsilon=0.1, min_support=50,
                        k_max=3, verbose=False)
        assert pfds, "Aucune PFD trouvée"
        found = any(
            r["col_X"] == "ZIP"
            and "606" in str(r["pattern_X"])
            and "Chicago" in str(r["pattern_Y"])
            for r in pfds
        )
        assert found, "Attendu : ZIP sw '606' -> CITY='Chicago'"

    def test_discover_t3_finds_licensee_channel(self):
        """t3.csv : un préfixe de 'BBW' doit mener à Channel Type='On Premise'.

        La généralisation peut élargir 'BBW' en 'BB' ou 'B' si la règle tient
        encore avec epsilon=0.1 — c'est le comportement attendu de l'algorithme.
        """
        t3 = pd.read_csv(os.path.join(DATA, "t3.csv"))
        pfds = discover(
            t3[["Licensee Number", "Channel Type"]],
            epsilon=0.1, min_support=20, k_max=5, verbose=False
        )
        # Vérifier qu'une règle couvre les Licensee Number 'BBW...' → 'On Premise'
        # (le pattern peut être 'B', 'BB', 'BBW' ou plus long selon la généralisation)
        found = any(
            r["col_X"] == "Licensee Number"
            and "BBW".startswith(str(r["pattern_X"]))   # pattern est un préfixe de BBW
            and "On Premise" in str(r["pattern_Y"])
            for r in pfds
        )
        assert found, (
            "Attendu : une règle Licensee Number -> 'On Premise' couvrant les cas 'BBW...'. "
            f"Règles trouvées : {[(r['pattern_X'], r['pattern_Y']) for r in pfds]}"
        )

    def test_discover_t1_fd_department(self):
        """t1.csv : FD exacte Department -> Department Name (0 violation)."""
        t1 = pd.read_csv(os.path.join(DATA, "t1.csv"))
        pfds = discover(
            t1[["Department", "Department Name"]],
            epsilon=0.0, min_support=5, k_max=3, verbose=False
        )
        # Au moins une règle Department -> Department Name doit être trouvée
        assert any(
            r["col_X"] == "Department" and r["col_Y"] == "Department Name"
            for r in pfds
        ), "Attendu : FD Department -> Department Name"

    def test_discover_empty_result_high_support(self):
        """Support trop élevé → aucun candidat ne passe."""
        t2 = pd.read_csv(os.path.join(DATA, "t2.csv"))
        pfds = discover(t2[["ZIP", "CITY"]], epsilon=0.0,
                        min_support=100_000, k_max=3, verbose=False)
        assert pfds == []

    def test_discover_results_sorted_by_confidence(self):
        """Les PFDs retournées doivent être triées par confiance décroissante."""
        t2 = pd.read_csv(os.path.join(DATA, "t2.csv"))
        pfds = discover(t2[["ZIP", "CITY"]], epsilon=0.1,
                        min_support=50, k_max=3, verbose=False)
        if len(pfds) >= 2:
            confs = [r["confidence"] for r in pfds]
            assert confs == sorted(confs, reverse=True)

    def test_discover_all_valid_confidence(self):
        """Toutes les règles retournées doivent avoir confiance >= 1 - epsilon."""
        t2 = pd.read_csv(os.path.join(DATA, "t2.csv"))
        epsilon = 0.1
        pfds = discover(t2[["ZIP", "CITY", "STATE"]], epsilon=epsilon,
                        min_support=30, k_max=3, verbose=False)
        for r in pfds:
            assert r["confidence"] >= (1 - epsilon) - 1e-9, (
                f"Confiance {r['confidence']:.4f} < seuil {1-epsilon} "
                f"pour {r['col_X']}:{r['pattern_X']} -> {r['col_Y']}"
            )


# ---------------------------------------------------------------------------
# Non-régression : les 6 PFDs connues de pfd_verifier doivent toujours passer
# ---------------------------------------------------------------------------

class TestNonRegression:

    def setup_method(self):
        self.t1 = pd.read_csv(os.path.join(DATA, "t1.csv"))
        self.t2 = pd.read_csv(os.path.join(DATA, "t2.csv"))
        self.t3 = pd.read_csv(os.path.join(DATA, "t3.csv"))

    def test_t1_gender_assignment(self):
        res = verifier_pfd(self.t1, "Gender", "F",
                           "Assignment Category", "Fulltime-Regular",
                           epsilon=0.1, match_type="exact")
        # Confiance connue ~81% → INVALIDE avec epsilon=0.1
        assert not res["is_valid"]
        assert abs(res["confidence"] - 0.8114) < 0.01

    def test_t3_zip_rockville(self):
        res = verifier_pfd(self.t3, "Zip", 20852.0, "City", "ROCKVILLE",
                           epsilon=0.05, match_type="exact")
        assert not res["is_valid"]
        assert res["n_matching_X"] == 95

    def test_t3_licensee_bbw(self):
        res = verifier_pfd(self.t3,
                           "Licensee Number", "BBW", "Channel Type", "On Premise",
                           epsilon=0.1,
                           match_type_X="startswith", match_type_Y="exact")
        assert res["is_valid"]
        assert res["confidence"] >= 0.90

    def test_t2_zip_chicago(self):
        res = verifier_pfd(self.t2, "ZIP", "606", "CITY", "Chicago",
                           epsilon=0.1,
                           match_type_X="startswith", match_type_Y="exact")
        assert res["is_valid"]
        assert res["confidence"] >= 0.95

    def test_t1_department_fd(self):
        """FD exacte : chaque Department code mappe vers un seul Department Name."""
        violations = 0
        for dept in self.t1["Department"].dropna().unique():
            nom = (self.t1
                   .loc[self.t1["Department"] == dept, "Department Name"]
                   .dropna()
                   .value_counts()
                   .index[0])
            res = verifier_pfd(self.t1, "Department", dept,
                               "Department Name", nom,
                               epsilon=0.0, match_type="exact")
            violations += res["n_violations"]
        assert violations == 0, f"FD exacte violée : {violations} violation(s)"
