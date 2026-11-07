import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.transform import rowcol, array_bounds

from fetch_dem import (
    LAKE_ID,
    OUTPUT_DIR,
    telecharger_bathymetrie_et_cellules,
    elargir_cellules,
    trouver_tuiles_mnt,
    telecharger_tuiles_mnt,
    construire_mosaique,
    lire_mnt_aux_points,
)
from masque_eau import construire_masque_eau, points_dans_masque

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
DISTANCES = [150, 300, 600, 900]
STEP = 2.0  # colle a la resolution du MNT
SHORE_SEARCH_MARGIN = 1.1  # marge de securite sur la diagonale du lac (voir main())
N_SAMPLE = 2000
NODATA = -9999.0
# on ne charge que 5% des points bathy (tires au hasard dans chaque tuile) : on n'en garde
# que 2000 a la fin, et lacneuchatel complet ne tient pas en memoire
FRACTION_BATHY = 0.05


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


def extraire_features_point(x0, y0, z_bed, mask, dem, transform, max_shore_search, water_level):
    rec = {"x": x0, "y": y0, "z": z_bed}
    # profondeur sous le niveau du lac (negative), meme definition que cote francais
    rec["angle0_z_DEM_ref"] = z_bed - water_level
    for angle in ANGLES:
        dists, vals, inside = echantillonner_rayon(
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


def tracer_verification(mosaic, transform, mask, sample_df, hull_poly):
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
    bathy_df, cells = telecharger_bathymetrie_et_cellules(fraction=FRACTION_BATHY)
    print(f"{len(bathy_df)} points bathymetriques ({FRACTION_BATHY:.0%} tires au hasard), {len(cells)} cellules km bathy")

    target_cells = elargir_cellules(cells, ring=1)
    tiles = trouver_tuiles_mnt(target_cells)
    tile_paths = telecharger_tuiles_mnt(tiles)
    mosaic, transform = construire_mosaique(tile_paths)
    print(f"Mosaique MNT : {mosaic.shape}")

    # quelques points bathy servent juste a reperer quelle zone plate est le lac
    points_lac = bathy_df.sample(n=min(500, len(bathy_df)), random_state=42)
    mask, hull_poly = construire_masque_eau(
        mosaic, transform, nan_est_eau=False, points_x=points_lac.x.values, points_y=points_lac.y.values
    )
    surface_area_ha = hull_poly.area / 10000.0

    # niveau du lac : le MNT donne la surface de l'eau au-dessus des points bathy du masque
    dans_lac = points_dans_masque(mask, transform, bathy_df.x.values, bathy_df.y.values)
    points_eau = bathy_df[dans_lac].sample(n=min(500, int(dans_lac.sum())), random_state=42)
    mnt_sur_lac = lire_mnt_aux_points(mosaic, transform, points_eau.x.values, points_eau.y.values)
    water_level = np.nanmedian(mnt_sur_lac)
    print(f"Niveau du lac = {water_level:.2f} m (ecart-type du MNT sur le lac : {np.nanstd(mnt_sur_lac):.2f} m)")

    # swissBATHY3D contient aussi des points de la berge, au-dessus de l'eau (10% des points
    # tires pour lacdejoux, jusqu'a +47m) : on ne garde que le fond, dans le masque et sous le niveau du lac
    fond = dans_lac & (bathy_df.z.values < water_level)
    print(f"{int(fond.sum())} points du fond gardes sur {len(bathy_df)} "
          f"({int((~dans_lac).sum())} hors du masque d'eau, {int((dans_lac & ~fond).sum())} au-dessus du niveau du lac)")
    bathy_fond = bathy_df[fond]
    # coefficient isoperimetrique (compacite de la forme, 4piA/P^2, 1 = cercle parfait) --
    # feature suggeree par Kacimi (rapport de stage) pour remplacer surface_area, qui
    # dominait l'importance des features sans bien generaliser (14/07).
    isoperimetric_coeff = 4 * np.pi * hull_poly.area / hull_poly.length**2
    print(f"Masque d'eau construit. Surface = {surface_area_ha:.1f} ha, coefficient isoperimetrique = {isoperimetric_coeff:.4f}")

    # Rayon de recherche du rivage adapte a la taille du lac : la diagonale de son
    # emprise (bathy_df) borne la distance max entre un point du fond et le rivage,
    # quelle que soit la direction -- avec une marge de securite. Un rayon fixe
    # (2500m, 07/07) etait trop court pour les lacs allonges (jusqu'a 64% de NaN
    # sur l'axe le plus long, decouvert le 14/07 en comparant plusieurs lacs).
    diag = np.hypot(bathy_df.x.max() - bathy_df.x.min(), bathy_df.y.max() - bathy_df.y.min())
    max_shore_search = diag * SHORE_SEARCH_MARGIN
    print(f"Rayon de recherche du rivage : {max_shore_search:.0f} m (diagonale du lac = {diag:.0f} m)")

    sample_df = bathy_fond.sample(n=min(N_SAMPLE, len(bathy_fond)), random_state=42).reset_index(drop=True)
    print(f"Extraction des features sur {len(sample_df)} points echantillonnes...")

    records = []
    for i, row in sample_df.iterrows():
        rec = extraire_features_point(row.x, row.y, row.z, mask, mosaic, transform, max_shore_search, water_level)
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

    tracer_verification(mosaic, transform, mask, out_df.assign(x=sample_df.x, y=sample_df.y), hull_poly)


if __name__ == "__main__":
    main()
