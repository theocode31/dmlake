import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.transform import rowcol, array_bounds

from fetch_dem_france import LAKE_ID, OUTPUT_DIR, telecharger_bathymetrie, recuperer_mnt
# masque de Sobel (methode de Kacimi, 20/07), deplace dans masque_eau.py pour que le
# cote suisse utilise le meme. Le resultat cote francais ne change pas.
from masque_eau import construire_masque_eau, points_dans_masque

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
DISTANCES = [150, 300, 600, 900]
STEP = 5.0  # colle a la resolution de RGE ALTI (vs 2.0 pour swissALTI3D)
SHORE_SEARCH_MARGIN = 1.1
N_SAMPLE = 2000
NODATA = -9999.0


def echantillonner_rayon(x0, y0, angle_deg, max_dist, step, arr, transform, fill_value):
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


def extraire_features_point(x0, y0, z_bed, mask, dem, transform, max_shore_search):
    rec = {"x": x0, "y": y0, "z": z_bed}
    # z_bed est deja une profondeur mesuree (EauFrance : "bathy_profondeurs.xyz"), pas
    # une altitude absolue comme cote suisse -- pas besoin de la corriger avec l'altitude
    # du rivage, elle sert directement de cible (cf angle0_z_DEM_ref cote suisse).
    rec["angle0_z_DEM_ref"] = z_bed
    for angle in ANGLES:
        dists, vals, inside = echantillonner_rayon(
            x0, y0, angle, max_shore_search, STEP, mask, transform, fill_value=0
        )
        land_idx = np.argmax(vals == 0)
        if vals[land_idx] != 0:
            rec[f"angle{angle}_shoreline_distance"] = np.nan
            for d in DISTANCES:
                rec[f"angle{angle}_m{d}"] = np.nan
                rec[f"angle{angle}_elevdiff{d}"] = np.nan
            continue

        shore_dist = dists[land_idx]
        theta = np.radians(angle)
        shore_x = x0 + np.sin(theta) * shore_dist
        shore_y = y0 + np.cos(theta) * shore_dist
        rec[f"angle{angle}_shoreline_distance"] = shore_dist

        pdists, pvals, pinside = echantillonner_rayon(
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


def tracer_verification(dem, transform, mask, sample_df, hull_poly):
    fig, ax = plt.subplots(figsize=(8, 6))
    left, bottom, right, top = array_bounds(dem.shape[0], dem.shape[1], transform)
    ax.imshow(dem, cmap="terrain", extent=(left, right, bottom, top), origin="upper", alpha=0.8)
    ax.contour(mask, levels=[0.5], colors="blue", extent=(left, right, bottom, top), origin="upper")

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
    ax.set_xlabel("x (EPSG:2154)")
    ax.set_ylabel("y (EPSG:2154)")
    ax.set_title(f"Profils cross-shore (echantillon) -- {LAKE_ID}")
    out_path = OUTPUT_DIR / f"{LAKE_ID}_cross_shore_profiles.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {out_path}")


def main() -> None:
    bathy_df = telecharger_bathymetrie()
    print(f"{len(bathy_df)} points bathymetriques")

    dem, transform = recuperer_mnt(bathy_df)
    print(f"MNT : {dem.shape}")

    # on garde la zone plate qui contient le plus de points bathy, et plus la plus grande
    # zone plate de l'image : pour 22 lacs sur 98, la plus grande etait une autre zone
    # (plaine, autre plan d'eau) et aucun point du lac ne tombait dedans
    points_lac = bathy_df.sample(n=min(500, len(bathy_df)), random_state=42)
    mask, hull_poly = construire_masque_eau(
        dem, transform, nan_est_eau=True, points_x=points_lac.x.values, points_y=points_lac.y.values
    )
    surface_area_ha = hull_poly.area / 10000.0
    isoperimetric_coeff = 4 * np.pi * hull_poly.area / hull_poly.length**2
    print(f"Masque d'eau construit. Surface = {surface_area_ha:.1f} ha, coefficient isoperimetrique = {isoperimetric_coeff:.4f}")

    diag = np.hypot(bathy_df.x.max() - bathy_df.x.min(), bathy_df.y.max() - bathy_df.y.min())
    max_shore_search = diag * SHORE_SEARCH_MARGIN
    print(f"Rayon de recherche du rivage : {max_shore_search:.0f} m (diagonale du lac = {diag:.0f} m)")

    # on ne garde que les points dans le masque d'eau : hors du masque, le rayon part de la
    # terre et toutes les distances au rivage valent 1 pixel (418 points sur 2000 pour L80 avant)
    dans_lac = points_dans_masque(mask, transform, bathy_df.x.values, bathy_df.y.values)
    print(f"{int(dans_lac.sum())} points dans le masque d'eau gardes sur {len(bathy_df)}")
    bathy_df = bathy_df[dans_lac]

    sample_df = bathy_df.sample(n=min(N_SAMPLE, len(bathy_df)), random_state=42).reset_index(drop=True)
    print(f"Extraction des features sur {len(sample_df)} points echantillonnes...")

    records = []
    for i, row in sample_df.iterrows():
        rec = extraire_features_point(row.x, row.y, row.z, mask, dem, transform, max_shore_search)
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

    tracer_verification(dem, transform, mask, out_df.assign(x=sample_df.x, y=sample_df.y), hull_poly)


if __name__ == "__main__":
    main()
