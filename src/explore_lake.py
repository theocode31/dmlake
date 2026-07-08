"""
Jour 1 : premier contact avec les données.

Objectif : verifier qu'on sait recuperer un jeu de donnees bathymetriques et le
MNT correspondant pour un lac, et jeter un premier oeil visuel avant de batir
quoi que ce soit de plus complique (masquage, extraction de profils, ML...).

Lac choisi : Lac de Joux (Suisse), un petit lac peu profond, pratique pour un
premier test rapide.
"""
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests

LAKE_ID = "lacdejoux"
BATHY_URL = (
    f"https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_{LAKE_ID}/"
    f"swissbathy3d_{LAKE_ID}_2056_5728.xyz.zip"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def download_bathymetry() -> pd.DataFrame:
    """Telecharge le nuage de points bathymetriques et le charge en DataFrame."""
    print(f"Telechargement de {BATHY_URL} ...")
    response = requests.get(BATHY_URL, timeout=60)
    response.raise_for_status()

    frames = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for name in archive.namelist():
            if not name.endswith(".xyz"):
                continue
            with archive.open(name) as f:
                tile_df = pd.read_csv(f, sep=r"\s+", skiprows=1, names=["x", "y", "z"])
                frames.append(tile_df)

    return pd.concat(frames, ignore_index=True)


def plot_bathymetry(df: pd.DataFrame) -> None:
    """Nuage de points colore par profondeur, pour un premier controle visuel."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df.x, df.y, c=df.z, cmap="viridis_r", s=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (EPSG:2056)")
    ax.set_ylabel("y (EPSG:2056)")
    ax.set_title(f"Bathymetrie brute -- {LAKE_ID}")
    fig.colorbar(scatter, ax=ax, label="Altitude du fond (m)")
    out_path = OUTPUT_DIR / f"{LAKE_ID}_bathymetry.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {out_path}")


def main() -> None:
    bathy_df = download_bathymetry()

    print(f"\n{len(bathy_df)} points bathymetriques charges")
    print(f"Altitude du fond -- min: {bathy_df.z.min():.1f} m, max: {bathy_df.z.max():.1f} m")
    print(f"Emprise x: [{bathy_df.x.min():.0f}, {bathy_df.x.max():.0f}]")
    print(f"Emprise y: [{bathy_df.y.min():.0f}, {bathy_df.y.max():.0f}]")

    plot_bathymetry(bathy_df)

    sample_path = OUTPUT_DIR / f"{LAKE_ID}_bathymetry_sample.csv"
    bathy_df.sample(n=min(2000, len(bathy_df)), random_state=42).to_csv(sample_path, index=False)
    print(f"Echantillon de 2000 points sauvegarde : {sample_path}")


if __name__ == "__main__":
    main()
