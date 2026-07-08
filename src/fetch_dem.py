"""
Jour 2 : ajouter le MNT et verifier l'alignement spatial avec la bathymetrie.

Objectif : recuperer le MNT (swissALTI3D, 2m) autour du Lac de Joux, le
mosaiquer, et le superposer avec le nuage de points bathymetriques du jour 1
pour verifier qu'on est bien dans le meme referentiel spatial (EPSG:2056)
avant de commencer a extraire des features de relief (jour 3+).
"""
import io
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import array_bounds, rowcol

LAKE_ID = "lacdejoux"
BATHY_URL = (
    f"https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_{LAKE_ID}/"
    f"swissbathy3d_{LAKE_ID}_2056_5728.xyz.zip"
)
STAC_ITEMS_URL = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DEM_DIR = OUTPUT_DIR / f"dem_{LAKE_ID}"
DEM_DIR.mkdir(parents=True, exist_ok=True)


def download_bathymetry_with_cells() -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    """Telecharge le nuage de points bathymetriques et releve au passage les
    cellules de la grille kilometrique (E, N) couvertes, pour cibler ensuite
    le telechargement du MNT sur la meme zone."""
    print(f"Telechargement de {BATHY_URL} ...")
    response = requests.get(BATHY_URL, timeout=60)
    response.raise_for_status()

    frames = []
    cells = set()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for name in archive.namelist():
            if not name.endswith(".xyz"):
                continue
            m = re.search(r"_(\d{4})_(\d{4})\.xyz$", name)
            if m:
                cells.add((int(m.group(1)), int(m.group(2))))
            with archive.open(name) as f:
                tile_df = pd.read_csv(f, sep=r"\s+", skiprows=1, names=["x", "y", "z"])
                frames.append(tile_df)

    return pd.concat(frames, ignore_index=True), cells


def buffer_cells(cells: set[tuple[int, int]], ring: int = 1) -> set[tuple[int, int]]:
    """Ajoute une couronne de cellules voisines autour des cellules bathy, pour
    que le MNT deborde un peu sur le rivage (utile pour les features de jour 3)."""
    buffered = set()
    for e, n in cells:
        for de in range(-ring, ring + 1):
            for dn in range(-ring, ring + 1):
                buffered.add((e + de, n + dn))
    return buffered


def find_dem_tiles(cells: set[tuple[int, int]]) -> dict[tuple[int, int], str]:
    """Interroge l'API STAC de swisstopo pour trouver les tuiles MNT 2m qui
    couvrent les cellules demandees."""
    es = [c[0] for c in cells]
    ns = [c[1] for c in cells]
    min_e, max_e = min(es), max(es)
    min_n, max_n = min(ns), max(ns)

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lon1, lat1 = transformer.transform(min_e * 1000, min_n * 1000)
    lon2, lat2 = transformer.transform((max_e + 1) * 1000, (max_n + 1) * 1000)
    bbox = f"{min(lon1, lon2)},{min(lat1, lat2)},{max(lon1, lon2)},{max(lat1, lat2)}"

    session = requests.Session()
    url = STAC_ITEMS_URL
    params = {"bbox": bbox, "limit": 100}
    found = {}
    while url:
        resp = session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for feat in data.get("features", []):
            m = re.match(r"swissalti3d_(\d+)_(\d{4})-(\d{4})$", feat["id"])
            if not m:
                continue
            e, n = int(m.group(2)), int(m.group(3))
            if (e, n) not in cells:
                continue
            for akey, aval in feat.get("assets", {}).items():
                if akey.endswith("_2_2056_5728.tif"):
                    found[(e, n)] = aval["href"]
        next_link = next((l["href"] for l in data.get("links", []) if l.get("rel") == "next"), None)
        url = next_link
        params = None

    print(f"{len(found)}/{len(cells)} tuiles MNT 2m trouvees")
    return found


def download_dem_tiles(tiles: dict[tuple[int, int], str]) -> list[Path]:
    paths = []
    for (_e, _n), href in sorted(tiles.items()):
        fname = DEM_DIR / Path(href).name
        if not fname.exists():
            r = requests.get(href, timeout=120)
            r.raise_for_status()
            fname.write_bytes(r.content)
        paths.append(fname)
    return paths


def build_mosaic(tile_paths: list[Path]):
    srcs = [rasterio.open(p) for p in tile_paths]
    nodata = srcs[0].nodata
    mosaic, transform = merge(srcs)
    for s in srcs:
        s.close()
    mosaic = mosaic[0].astype(float)
    if nodata is not None:
        mosaic[mosaic == nodata] = np.nan
    return mosaic, transform


def sample_dem_at_points(mosaic: np.ndarray, transform, xs, ys) -> np.ndarray:
    """Echantillonne l'altitude du MNT (nearest) aux coordonnees (x, y) donnees."""
    rows, cols = rowcol(transform, xs, ys)
    rows = np.clip(np.asarray(rows), 0, mosaic.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, mosaic.shape[1] - 1)
    return mosaic[rows, cols]


def plot_overlay(mosaic: np.ndarray, transform, bathy_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    left, bottom, right, top = array_bounds(mosaic.shape[0], mosaic.shape[1], transform)
    im = ax.imshow(mosaic, cmap="terrain", extent=(left, right, bottom, top), origin="upper")
    ax.scatter(bathy_df.x, bathy_df.y, c="blue", s=1, alpha=0.3, label="points bathy (fond du lac)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (EPSG:2056)")
    ax.set_ylabel("y (EPSG:2056)")
    ax.set_title(f"MNT + bathymetrie -- {LAKE_ID}")
    fig.colorbar(im, ax=ax, label="Altitude MNT (m)")
    ax.legend(loc="upper right")
    out_path = OUTPUT_DIR / f"{LAKE_ID}_dem_overlay.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {out_path}")


def main() -> None:
    bathy_df, cells = download_bathymetry_with_cells()
    print(f"{len(bathy_df)} points bathymetriques, {len(cells)} cellules km bathy")

    target_cells = buffer_cells(cells, ring=1)
    print(f"{len(target_cells)} cellules cibles pour le MNT (avec couronne de 1km)")

    tiles = find_dem_tiles(target_cells)
    tile_paths = download_dem_tiles(tiles)
    print(f"{len(tile_paths)} tuiles MNT telechargees dans {DEM_DIR}")

    mosaic, transform = build_mosaic(tile_paths)
    print(
        f"Mosaique MNT : {mosaic.shape}, "
        f"altitude min/max = {np.nanmin(mosaic):.1f}/{np.nanmax(mosaic):.1f} m"
    )

    # Verification d'alignement : le MNT swissALTI3D ne voit pas le fond du lac
    # (LiDAR aerien, pas de penetration dans l'eau) -- au niveau des points bathy
    # on s'attend donc a une altitude MNT ~constante, proche du niveau du lac,
    # nettement au-dessus de l'altitude du fond (bathy z). Si c'est le cas, les
    # deux jeux de donnees sont bien dans le meme referentiel spatial.
    sample = bathy_df.sample(n=min(500, len(bathy_df)), random_state=42)
    dem_at_bathy = sample_dem_at_points(mosaic, transform, sample.x.values, sample.y.values)
    print(
        f"Altitude MNT au niveau de {len(sample)} points bathy (echantillon) : "
        f"min={dem_at_bathy.min():.1f}, max={dem_at_bathy.max():.1f}, "
        f"mean={dem_at_bathy.mean():.1f} m (attendu : ~niveau du lac, quasi constant)"
    )
    print(
        f"Altitude du fond (bathy z) sur le meme echantillon : "
        f"min={sample.z.min():.1f}, max={sample.z.max():.1f} m"
    )

    plot_overlay(mosaic, transform, bathy_df)


if __name__ == "__main__":
    main()
