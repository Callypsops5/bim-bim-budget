import os

# Répertoire racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Chemins des différentes couches du Data Lake
BRONZE_PATH = os.path.join(BASE_DIR, "data", "bronze")
SILVER_PATH = os.path.join(BASE_DIR, "data", "silver")
GOLD_PATH = os.path.join(BASE_DIR, "data", "gold")
MARTS_PATH = os.path.join(BASE_DIR, "data", "marts")

# Chemin des fichiers d'entrée
INPUT_PATH = os.path.join(BASE_DIR, "inputs")
