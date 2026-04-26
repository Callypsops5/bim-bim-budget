import os

def ensure_directory_exists(path: str) -> None:
    """
    Crée le dossier s'il n'existe pas déjà.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
