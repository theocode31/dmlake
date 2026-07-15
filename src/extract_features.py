"""
Jour 3 : extraction des features de relief autour de chaque point bathymetrique.

Objectif : reproduire la methode "profils cross-shore" du stage de reference
(Kacimi, DEM4Lakes) -- pour chaque point du fond du lac, dans 8 directions
(0 a 315 degres, pas de 45, 0 = nord, sens horaire) : chercher d'abord le
rivage (transition eau -> terre dans le masque d'eau), puis prolonger le rayon
sur terre pour mesurer la pente et le denivele a 4 distances (150/300/600/900 m).
Ces features decrivent la morphologie du terrain environnant, utilisee ensuite
pour predire la profondeur sans mesure directe.

Masque d'eau : rasterisation de l'enveloppe convexe des points bathymetriques
sur la grille du MNT (jour 2). Suffisant pour un lac simple comme Lac de Joux
(pas besoin du filtrage Sobel utilise par Kacimi pour des lacs plus complexes
ou seulement echantillonnes).

Limite connue : le MNT (jour 2) n'a ete telecharge qu'avec une couronne de 1km
autour du lac. Certains rayons, selon leur orientation, peuvent donc sortir de
la zone couverte avant d'atteindre le vrai rivage -- ces cas sont detailles par
le flag `angleX_shore_extrapolated` (rivage trouve au bord de la zone couverte,
pas un vrai contact avec la terre) plutot que d'etre ignores silencieusement.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio.features
from rasterio.transform import rowcol, array_bounds
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

from fetch_dem import (
    LAKE_ID,
    OUTPUT_DIR,
    download_bathymetry_with_cells,
    buffer_cells,
    find_dem_tiles,
    download_dem_tiles,
    build_mosaic,
)

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
DISTANCES = [150, 300, 600, 900]
STEP = 2.0  # colle a la resolution du MNT
SHORE_SEARCH_MARGIN = 1.1  # marge de securite sur la diagonale du lac (voir main())
N_SAMPLE = 2000
NODATA = -9999.0


def build_water_mask(bathy_df: pd.DataFrame, mosaic_shape: tuple[int, int], transform):
    """Masque d'eau = enveloppe convexe des points bathy, rasterisee sur la
    grille du MNT."""
    pts = bathy_df[["x", "y"]].to_numpy()
    hull = ConvexHull(pts)
    hull_poly = Polygon(pts[hull.vertices])
    mask = rasterio.features.rasterize(
        [(hull_poly, 1)],
        out_shape=mosaic_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask, hull_poly


def sample_along_ray(x0, y0, angle_deg, max_dist, step, arr, transform, fill_value):
    """Echantillonne `arr` le long d'un rayon parti de (x0, y0). Renvoie aussi
    `inside`, qui distingue une vraie valeur lue d'une valeur hors-grille."""
    theta = np.radians(angle_deg)
    dxu, dyu = np.sin(theta), np.cos(theta)
    n = int(max_dist / step)
    dists = np.arange(1, n + 1) * step
    xs = x0 + dxu * dists
    ys = y0 + dyu * dists
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    inside = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    vals = np.full(n, fill_value, dtype=float)
    vals[inside] = arr[rows[inside], cols[inside]]
    return dists, vals, inside


def extract_point_features(x0, y0, z_bed, mask, dem, transform, max_shore_search):
    rec = {"x": x0, "y": y0, "z": z_bed}
    for angle in ANGLES:
        dists, vals, inside = sample_along_ray(
            x0, y0, angle, max_shore_search, STEP, mask, transform, fill_value=0
        )
        land_idx = np.argmax(vals == 0)
        if vals[land_idx] != 0:
            # jamais quitte l'eau dans le rayon de recherche
            rec[f"angle{angle}_shoreline_distance"] = np.nan
            rec[f"angle{angle}_shore_extrapolated"] = np.nan
            for d in DISTANCES:
                rec[f"angle{angle}_m{d}"] = np.nan
                rec[f"angle{angle}_elevdiff{d}"] = np.nan
            continue

        shore_dist = dists[land_idx]
        theta = np.radians(angle)
        shore_x = x0 + np.sin(theta) * shore_dist
        shore_y = y0 + np.cos(theta) * shore_dist
        rec[f"angle{angle}_shoreline_distance"] = shore_dist
        # si le "rivage" trouve est en fait le bord de la zone MNT couverte
        # (pas une vraie valeur "terre" lue), le signaler plutot que le taire
        rec[f"angle{angle}_shore_extrapolated"] = not bool(inside[land_idx])

        if angle == 0:
            srow, scol = rowcol(transform, shore_x, shore_y)
            if 0 <= srow < dem.shape[0] and 0 <= scol < dem.shape[1]:
                shore_elev = dem[srow, scol]
            else:
                shore_elev = np.nan
            rec["angle0_z_DEM_ref"] = (
                z_bed - shore_elev if not np.isnan(shore_elev) else np.nan
            )

        pdists, pvals, pinside = sample_along_ray(
            shore_x, shore_y, angle, max(DISTANCES), STEP, dem, transform, fill_value=NODATA
        )
        for d in DISTANCES:
            seg_mask = (pdists <= d) & pinside & (pvals != NODATA) & ~np.isnan(pvals)
            seg_d, seg_v = pdists[seg_mask], pvals[seg_mask]
            if seg_d.size >= 2:
                slope = np.polyfit(seg_d, seg_v, 1)[0]
                elevdiff = seg_v.max() - seg_v.min()
            else:
                slope, elevdiff = np.nan, np.nan
            rec[f"angle{angle}_m{d}"] = slope
            rec[f"angle{angle}_elevdiff{d}"] = elevdiff
    return rec


def plot_sanity_check(mosaic, transform, mask, sample_df, hull_poly):
    """Superpose MNT, masque d'eau et quelques rayons pour verifier visuellement
    que la methode fait ce qu'on croit."""
    fig, ax = plt.subplots(figsize=(8, 6))
    left, bottom, right, top = array_bounds(mosaic.shape[0], mosaic.shape[1], transform)

    # Sous-echantillonne l'affichage pour les tres grandes mosaiques (Bodensee/Leman
    # font des dizaines de milliers de pixels de cote -- trop pour matplotlib en
    # memoire). L'extraction de features, elle, utilise deja le mosaic complet plus
    # haut ; seul cet apercu visuel est degrade.
    stride = max(1, max(mosaic.shape) // 2000)
    mosaic_disp = mosaic[::stride, ::stride]
    mask_disp = mask[::stride, ::stride]

    ax.imshow(mosaic_disp, cmap="terrain", extent=(left, right, bottom, top), origin="upper", alpha=0.8)
    ax.contour(
        mask_disp,
        levels=[0.5],
        colors="blue",
        extent=(left, right, bottom, top),
        origin="upper",
    )

    rng = np.random.default_rng(42)
    check_points = sample_df.sample(n=min(6, len(sample_df)), random_state=42)
    for _, row in check_points.iterrows():
        x0, y0 = row.x, row.y
        for angle in ANGLES:
            theta = np.radians(angle)
            sd = row.get(f"angle{angle}_shoreline_distance", np.nan)
            if np.isnan(sd):
                continue
            x1 = x0 + np.sin(theta) * sd
            y1 = y0 + np.cos(theta) * sd
            ax.plot([x0, x1], [y0, y1], color="red", linewidth=0.6)
        ax.scatter([x0], [y0], color="black", s=8, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x (EPSG:2056)")
    ax.set_ylabel("y (EPSG:2056)")
    ax.set_title(f"Profils cross-shore (echantillon) -- {LAKE_ID}")
    out_path = OUTPUT_DIR / f"{LAKE_ID}_cross_shore_profiles.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {out_path}")


def main() -> None:
    bathy_df, cells = download_bathymetry_with_cells()
    print(f"{len(bathy_df)} points bathymetriques, {len(cells)} cellules km bathy")

    target_cells = buffer_cells(cells, ring=1)
    tiles = find_dem_tiles(target_cells)
    tile_paths = download_dem_tiles(tiles)
    mosaic, transform = build_mosaic(tile_paths)
    print(f"Mosaique MNT : {mosaic.shape}")

    mask, hull_poly = build_water_mask(bathy_df, mosaic.shape, transform)
    surface_area_ha = hull_poly.area / 10000.0
    # coefficient isoperimetrique (compacite de la forme, 4piA/P^2, 1 = cercle parfait) --
    # feature suggeree par Kacimi (rapport de stage) pour remplacer surface_area, qui
    # dominait l'importance des features sans bien generaliser (jour 7).
    isoperimetric_coeff = 4 * np.pi * hull_poly.area / hull_poly.length**2
    print(f"Masque d'eau construit. Surface (enveloppe convexe) = {surface_area_ha:.1f} ha, coefficient isoperimetrique = {isoperimetric_coeff:.4f}")

    # Rayon de recherche du rivage adapte a la taille du lac : la diagonale de son
    # emprise (bathy_df) borne la distance max entre un point du fond et le rivage,
    # quelle que soit la direction -- avec une marge de securite. Un rayon fixe
    # (2500m, jour 3) etait trop court pour les lacs allonges (jusqu'a 64% de NaN
    # sur l'axe le plus long, decouvert au jour 6 en comparant plusieurs lacs).
    diag = np.hypot(bathy_df.x.max() - bathy_df.x.min(), bathy_df.y.max() - bathy_df.y.min())
    max_shore_search = diag * SHORE_SEARCH_MARGIN
    print(f"Rayon de recherche du rivage : {max_shore_search:.0f} m (diagonale du lac = {diag:.0f} m)")

    sample_df = bathy_df.sample(n=min(N_SAMPLE, len(bathy_df)), random_state=42).reset_index(drop=True)
    print(f"Extraction des features sur {len(sample_df)} points echantillonnes...")

    records = []
    for i, row in sample_df.iterrows():
        rec = extract_point_features(row.x, row.y, row.z, mask, mosaic, transform, max_shore_search)
        rec["surface_area"] = surface_area_ha
        rec["isoperimetric_coeff"] = isoperimetric_coeff
        rec["survey_id"] = LAKE_ID
        records.append(rec)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(sample_df)} points traites")

    out_df = pd.DataFrame(records)
    out_path = OUTPUT_DIR / f"{LAKE_ID}_features.csv"
    out_df.to_csv(out_path, index=False)
    print(f"DONE. {len(out_df)} lignes, {out_df.shape[1]} colonnes -> {out_path}")

    extrap_cols = [c for c in out_df.columns if c.endswith("_shore_extrapolated")]
    extrap_rate = out_df[extrap_cols].mean().mean()
    print(f"Taux de rivages extrapoles (bord de la zone MNT couverte, pas un vrai contact terre) : {extrap_rate:.1%}")
    print(out_df[["angle0_z_DEM_ref", "angle0_shoreline_distance", "angle0_m300"]].describe())

    plot_sanity_check(mosaic, transform, mask, out_df.assign(x=sample_df.x, y=sample_df.y), hull_poly)


if __name__ == "__main__":
    main()
