import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.transform import array_bounds

LAKE_FOLDER = "L80_ASCONIT_2012"
LAKE_ID = LAKE_FOLDER.split("_")[0]
BATHY_URL = f"https://adour-garonne.eaufrance.fr/upload/DATA/SIG/BATHYMETRIE/xyz_pointcloud/{LAKE_FOLDER}/bathy_profondeurs.xyz"
WMS_URL = "https://data.geopf.fr/wms-r/wms"
DEM_RESOLUTION = 5.0  # m, resolution native de RGE ALTI (vs 2m pour swissALTI3D)
BUFFER_M = 1000.0
WMS_MAX_PX = 5000  # limite d'une requete WMS unique

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DEM_DIR = OUTPUT_DIR / "dem_france"
DEM_DIR.mkdir(parents=True, exist_ok=True)


def download_bathymetry() -> pd.DataFrame:
    print(f"Telechargement de {BATHY_URL} ...")
    # adour-garonne.eaufrance.fr sert une chaine de certificat incomplete -- donnee
    # publique en clair, risque faible (meme choix que process_eaufrance_lake.py du stage).
    response = requests.get(BATHY_URL, timeout=60, verify=False)
    response.raise_for_status()
    df = pd.read_csv(
        io.BytesIO(response.content), sep=r"\s+", skiprows=1, names=["x", "y", "z"], encoding="utf-8-sig"
    )
    # z ici est deja une PROFONDEUR mesuree (valeurs <= 0, relatives a la surface du lac),
    # pas une altitude absolue comme le z suisse -- cf extract_features_france.py.
    return df


def fetch_dem(bathy_df: pd.DataFrame):
    xmin, xmax = bathy_df.x.min() - BUFFER_M, bathy_df.x.max() + BUFFER_M
    ymin, ymax = bathy_df.y.min() - BUFFER_M, bathy_df.y.max() + BUFFER_M
    width_px = int((xmax - xmin) / DEM_RESOLUTION)
    height_px = int((ymax - ymin) / DEM_RESOLUTION)
    if width_px > WMS_MAX_PX or height_px > WMS_MAX_PX:
        raise ValueError(f"Zone trop grande pour une requete WMS unique ({width_px}x{height_px}px)")

    dem_path = DEM_DIR / f"{LAKE_ID}_dem.tif"
    if not dem_path.exists():
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES",
            "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
            "CRS": "EPSG:2154",
            "WIDTH": width_px,
            "HEIGHT": height_px,
            "FORMAT": "image/geotiff",
            "STYLES": "normal",
        }
        print(f"Telechargement MNT RGE ALTI ({width_px}x{height_px}px)...")
        r = requests.get(WMS_URL, params=params, timeout=120)
        r.raise_for_status()
        dem_path.write_bytes(r.content)

    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(float)
        transform = ds.transform
        nodata = ds.nodata if ds.nodata is not None else -9999.0
    dem[dem == nodata] = np.nan
    return dem, transform


def plot_overlay(dem: np.ndarray, transform, bathy_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    left, bottom, right, top = array_bounds(dem.shape[0], dem.shape[1], transform)
    im = ax.imshow(dem, cmap="terrain", extent=(left, right, bottom, top), origin="upper")
    ax.scatter(bathy_df.x, bathy_df.y, c="blue", s=1, alpha=0.3, label="points bathy (fond du lac)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (EPSG:2154)")
    ax.set_ylabel("y (EPSG:2154)")
    ax.set_title(f"MNT + bathymetrie -- {LAKE_ID}")
    fig.colorbar(im, ax=ax, label="Altitude MNT (m)")
    ax.legend(loc="upper right")
    out_path = OUTPUT_DIR / f"{LAKE_ID}_dem_overlay.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {out_path}")


def main() -> None:
    bathy_df = download_bathymetry()
    print(f"{len(bathy_df)} points bathymetriques, profondeur {bathy_df.z.min():.1f} a {bathy_df.z.max():.1f} m")

    dem, transform = fetch_dem(bathy_df)
    print(f"MNT : {dem.shape}, altitude min/max = {np.nanmin(dem):.1f}/{np.nanmax(dem):.1f} m")

    plot_overlay(dem, transform, bathy_df)


if __name__ == "__main__":
    main()
