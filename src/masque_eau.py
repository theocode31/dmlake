import numpy as np
import rasterio.features
from rasterio.transform import rowcol
from scipy import ndimage
from scipy.ndimage import sobel
from shapely.geometry import shape as shapely_shape

SOBEL_SLOPE_THRESHOLD = 0.04


def construire_masque_eau(dem: np.ndarray, transform, nan_est_eau: bool = True, points_x=None, points_y=None):
    valid = ~np.isnan(dem)
    # float32 pour tenir en memoire sur les grands lacs suisses (MNT de 2m)
    filled = np.where(valid, dem, np.nanmean(dem[valid])).astype(np.float32)
    dx = sobel(filled, axis=1, output=np.float32)
    dy = sobel(filled, axis=0, output=np.float32)
    pixel_size = abs(transform.a)
    slope = np.sqrt(dx**2 + dy**2) / 8.0 / pixel_size
    water = slope < SOBEL_SLOPE_THRESHOLD
    water = (water | ~valid) if nan_est_eau else (water & valid)

    if points_x is not None:
        labels, _ = ndimage.label(water)
        rows, cols = rowcol(transform, points_x, points_y)
        rows = np.clip(np.asarray(rows), 0, dem.shape[0] - 1)
        cols = np.clip(np.asarray(cols), 0, dem.shape[1] - 1)
        seed_labels = labels[rows, cols]
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size == 0:
            raise ValueError("aucun point du lac ne tombe dans une zone plate, verifier le seuil de Sobel")
        water = labels == np.bincount(seed_labels).argmax()

    water = water.astype("uint8")
    polys = [
        shapely_shape(geom)
        for geom, val in rasterio.features.shapes(water, mask=(water == 1), transform=transform)
    ]
    lake_poly = max(polys, key=lambda p: p.area)
    # le contour polygonise suit la grille pixel par pixel (effet "escalier"), ce qui gonfle
    # artificiellement le perimetre sans changer l'aire -- simplifie (tolerance = 2 pixels)
    # pour que le coefficient isoperimetrique reste interpretable.
    lake_poly = lake_poly.simplify(2 * pixel_size, preserve_topology=True)

    mask = rasterio.features.rasterize(
        [(lake_poly, 1)], out_shape=dem.shape, transform=transform, fill=0, dtype="uint8"
    )
    return mask, lake_poly


def points_dans_masque(mask: np.ndarray, transform, xs, ys) -> np.ndarray:
    rows, cols = rowcol(transform, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    dedans = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    resultat = np.zeros(len(rows), dtype=bool)
    resultat[dedans] = mask[rows[dedans], cols[dedans]] == 1
    return resultat
