import shutil
import subprocess
import sys
from pathlib import Path

from lacs import LAKES_CH, LAKES_FR_FOLDERS

RACINE = Path(__file__).resolve().parent.parent
SRC = RACINE / "src"
OUTPUT_DIR = RACINE / "outputs"
ESPACE_MIN_GO = 3.0


def extraire_lac(script: str, argument: str, lake_id: str, refaire: bool) -> bool:
    csv = OUTPUT_DIR / f"{lake_id}_features.csv"
    if csv.exists() and not refaire:
        print(f"{lake_id} deja fait, saute")
        return True
    print(f"\n{lake_id}")
    resultat = subprocess.run([sys.executable, str(SRC / script), argument], cwd=RACINE)
    return resultat.returncode == 0


def espace_libre_go() -> float:
    return shutil.disk_usage(RACINE).free / 1e9


def main() -> None:
    refaire = "--refaire" in sys.argv
    lacs = []
    if "--france" not in sys.argv:
        lacs += [("extract_features.py", lake_id, lake_id) for lake_id in LAKES_CH]
    if "--suisse" not in sys.argv:
        lacs += [("extract_features_france.py", dossier, lake_id) for lake_id, dossier in LAKES_FR_FOLDERS.items()]

    echecs = []
    for script, argument, lake_id in lacs:
        if espace_libre_go() < ESPACE_MIN_GO:
            print(f"\nArret : moins de {ESPACE_MIN_GO} Go libres sur le disque. Relancer apres avoir fait de la place.")
            break
        if not extraire_lac(script, argument, lake_id, refaire):
            echecs.append(lake_id)
    print(f"\nTermine. {len(echecs)} lac(s) en echec : {', '.join(echecs) if echecs else 'aucun'}")


if __name__ == "__main__":
    main()
