# Ajoute la racine du projet au sys.path pour que pytest trouve tous les modules.
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
