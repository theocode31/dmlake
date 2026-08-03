import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

FEATURES_CSV = "outputs/lacdejoux_features.csv"
EXCLUDED_ANGLES = []  # obsolete depuis jour 6 (rayon dynamique)
NON_FEATURE_COLS = ["x", "y", "z", "survey_id"]
DEPTH_TARGET_COL = "angle0_z_DEM_ref"  # profondeur relative au rivage
DEPTH_RATIO_COL = "depth_ratio"  # profondeur normalisee par la profondeur max du lac
LAKES_CH = [
    "lacdejoux", "lungernsee", "aegerisee", "baldeggersee", "hallwilersee",
    "bielersee", "brienzersee", "lacneuchatel", "lagomaggiore",
]  # bodensee/lacleman exclus -- MNT suisse hors frontiere
LAKES_FR = [
    "L1", "L2", "L3", "L4", "L6", "L7", "L9", "L10", "L11", "L12", "L13", "L15", "L16", "L17",
    "L18", "L19", "L22", "L23", "L26", "L27", "L28", "L29", "L30", "L31", "L32", "L33", "L34",
    "L35", "L36", "L37", "L38", "L39", "L40", "L41", "L42", "L43", "L44", "L45", "L46", "L47",
    "L48", "L49", "L50", "L51", "L52", "L53", "L54", "L55", "L56", "L57", "L58", "L59", "L60",
    "L61", "L62", "L63", "L64", "L65", "L66", "L67", "L68", "L69", "L70", "L71", "L72", "L73",
    "L74", "L76", "L77", "L78", "L79", "L80", "L81", "L82", "L83", "L84", "L85", "L86", "L87",
    "L88", "L89", "L90", "L91", "L93", "L94", "L95", "L96", "L97", "L98", "L99", "L100", "L101",
    "L102", "L103", "L104", "L105", "L106", "L107",
]  # jour 21 : elargi 12->98 lacs, 8 exclus (donnees source corrompues)
LAKES = LAKES_CH + LAKES_FR
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    drop_cols = [c for c in df.columns if any(c.startswith(f"angle{a}_") for a in EXCLUDED_ANGLES)]
    df = df.drop(columns=drop_cols)
    extrap_cols = [c for c in df.columns if c.endswith("_shore_extrapolated")]
    df = df.drop(columns=extrap_cols)  # flags non exploitables comme features
    df = df.drop(columns=["angle0_z_DEM_ref"])  # fuite de donnees (quasi = cible)
    return df


def load_dataset_for(lake_id: str) -> pd.DataFrame:
    """Comme load_dataset(), mais pour un lac quelconque, et garde
    DEPTH_TARGET_COL (retire par load_dataset()) -- utilise pour le test de
    generalisation cross-lac (train sur un lac, test sur l'autre)."""
    df = pd.read_csv(f"outputs/{lake_id}_features.csv")
    drop_cols = [c for c in df.columns if any(c.startswith(f"angle{a}_") for a in EXCLUDED_ANGLES)]
    df = df.drop(columns=drop_cols)
    extrap_cols = [c for c in df.columns if c.endswith("_shore_extrapolated")]
    df = df.drop(columns=extrap_cols)
    df = df.dropna()
    df[DEPTH_RATIO_COL] = df[DEPTH_TARGET_COL] / df[DEPTH_TARGET_COL].min()  # normalise par profondeur max (jour 7)
    return df


def evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str, label: str) -> None:
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, max_depth=20)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{label} (train={len(X_train)}, test={len(X_test)})")
    print(f"RMSE = {rmse:.3f} m, MAE = {mae:.3f} m, R2 = {r2:.3f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 5 features (importance Random Forest) :")
    print(importances.head(5))


def main() -> None:
    df = load_dataset()
    print(f"{len(df)} lignes, {df.shape[1]} colonnes")

    before = len(df)
    df = df.dropna()
    print(f"{len(df)} lignes apres dropna sur les colonnes restantes ({before - len(df)} lignes perdues)")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["z"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, max_depth=20)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nRMSE = {rmse:.3f} m, MAE = {mae:.3f} m, R2 = {r2:.3f}")
    print(f"(pour reference : z varie de {y.min():.1f} a {y.max():.1f} m, ecart-type {y.std():.1f} m)")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 features (importance Random Forest) :")
    print(importances.head(10))

    # leave-one-lake-out, cible=DEPTH_TARGET_COL (z absolu ne marche pas, jour 5)
    datasets = {lake_id: load_dataset_for(lake_id) for lake_id in LAKES}
    for lake_id, d in datasets.items():
        print(f"{lake_id}: {len(d)} lignes apres nettoyage")

    cross_feature_cols = [c for c in feature_cols if c not in (DEPTH_TARGET_COL, "isoperimetric_coeff")]  # iso exclu (CH seulement)
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, datasets[test_id], cross_feature_cols, DEPTH_TARGET_COL, f"train=autres lacs / test={test_id}")

    print("\n\n=== Meme test, cible normalisee (depth_ratio) ===")  # decouple forme et echelle (jour 16)
    ratio_feature_cols = [c for c in cross_feature_cols if c != DEPTH_RATIO_COL]
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, datasets[test_id], ratio_feature_cols, DEPTH_RATIO_COL, f"[ratio] train=autres lacs / test={test_id}")

    # isoperimetric_coeff, lacs CH seulement (jour 8, piste Kacimi)
    ch_datasets = {lid: d for lid, d in datasets.items() if lid in LAKES_CH}
    ch_feature_cols = [c for c in ch_datasets["lacdejoux"].columns if c not in NON_FEATURE_COLS + [DEPTH_TARGET_COL, DEPTH_RATIO_COL]]
    baseline_cols = [c for c in ch_feature_cols if c != "isoperimetric_coeff"]
    with_surface_cols = ch_feature_cols
    without_surface_cols = [c for c in ch_feature_cols if c != "surface_area"]

    print("\n\n=== Lacs suisses seuls, baseline SANS isoperimetric_coeff ===")
    for test_id in LAKES_CH:
        train_df = pd.concat([d for lid, d in ch_datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, ch_datasets[test_id], baseline_cols, DEPTH_RATIO_COL, f"[CH-baseline] test={test_id}")

    print("\n\n=== Lacs suisses seuls, avec surface_area ET isoperimetric_coeff ===")
    for test_id in LAKES_CH:
        train_df = pd.concat([d for lid, d in ch_datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, ch_datasets[test_id], with_surface_cols, DEPTH_RATIO_COL, f"[CH+iso] test={test_id}")

    print("\n\n=== Lacs suisses seuls, isoperimetric_coeff SANS surface_area ===")
    for test_id in LAKES_CH:
        train_df = pd.concat([d for lid, d in ch_datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, ch_datasets[test_id], without_surface_cols, DEPTH_RATIO_COL, f"[CH+iso-nosurf] test={test_id}")

    # jour 9 : config retenue, tous les lacs
    all_feature_cols = [c for c in datasets["lacdejoux"].columns if c not in NON_FEATURE_COLS + [DEPTH_TARGET_COL, DEPTH_RATIO_COL, "surface_area"]]
    print("\n\n=== Test final, 21 lacs (9 CH + 12 FR), depth_ratio + isoperimetric_coeff sans surface_area ===")
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, datasets[test_id], all_feature_cols, DEPTH_RATIO_COL, f"[FINAL] test={test_id}")

    # isole l'effet isoperimetric_coeff vs nouvelles donnees FR
    surface_only_cols = [c for c in all_feature_cols if c != "isoperimetric_coeff"] + ["surface_area"]
    print("\n\n=== Isolation : 21 lacs, depth_ratio + surface_area (sans isoperimetric_coeff), nouvelles donnees FR ===")
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, datasets[test_id], surface_only_cols, DEPTH_RATIO_COL, f"[ISOLATION-surface] test={test_id}")

    # jour 21 : modele deux etages, tente de reconstituer l'echelle separement
    print("\n\n=== Jour 21 : modele deux etages (echelle + forme) pour reconstituer la profondeur absolue ===")
    scale_table = build_lake_scale_table(datasets)
    scale_feature_cols = [c for c in scale_table.columns if c != "max_depth"]

    scale_preds = {}
    for test_id in LAKES:
        train_scale = scale_table.drop(index=test_id)
        scale_model = RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, max_depth=4, min_samples_leaf=2
        )  # regularise (peu de lacs)
        scale_model.fit(train_scale[scale_feature_cols], np.log(train_scale["max_depth"]))  # cible en log (2 ordres de grandeur)
        pred_log = scale_model.predict(scale_table.loc[[test_id], scale_feature_cols])[0]
        scale_preds[test_id] = np.exp(pred_log)

    scale_summary = pd.DataFrame({
        "max_depth_reel": scale_table["max_depth"],
        "max_depth_predit": pd.Series(scale_preds),
    })
    scale_summary["erreur_pct"] = (
        (scale_summary["max_depth_predit"] - scale_summary["max_depth_reel"]) / scale_summary["max_depth_reel"] * 100
    )
    print("\nEtage 1 (echelle), leave-one-lake-out :")
    print(scale_summary.round(1))

    # etage 2, cache pour reutiliser plus bas (variante log-lineaire)
    print("\nModele combine (echelle etage 1 RF x ratio etage 2), profondeur absolue :")
    problem_lakes = ["lungernsee", "bielersee", "lacneuchatel", "lagomaggiore", "L9"]
    ratio_preds, depth_trues = {}, {}
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, max_depth=20)
        model.fit(train_df[all_feature_cols], train_df[DEPTH_RATIO_COL])
        ratio_pred = model.predict(datasets[test_id][all_feature_cols])
        ratio_preds[test_id] = ratio_pred
        depth_trues[test_id] = datasets[test_id][DEPTH_TARGET_COL].abs()

        depth_true = depth_trues[test_id]
        depth_pred_combined = ratio_pred * scale_preds[test_id]
        depth_pred_ratio_only = ratio_pred * scale_table.loc[test_id, "max_depth"]  # oracle : vraie echelle

        r2_combined = r2_score(depth_true, depth_pred_combined)
        rmse_combined = root_mean_squared_error(depth_true, depth_pred_combined)
        r2_oracle = r2_score(depth_true, depth_pred_ratio_only)
        flag = " <-- lac probleme" if test_id in problem_lakes else ""
        print(
            f"{test_id:15s} combine: R2={r2_combined:7.3f} RMSE={rmse_combined:7.2f}m | "
            f"oracle (vraie echelle): R2={r2_oracle:7.3f}{flag}"
        )

    print("\n\n=== Variante etage 1 : regression log-lineaire (extrapole, contrairement au RF) ===")
    loglinear_cols = ["surface_area", "mean_shore_dist"]  # les 2 plus correlees a max_depth (0.61 et 0.77)
    scale_table_log = scale_table.copy()
    scale_table_log[loglinear_cols] = np.log(scale_table_log[loglinear_cols])

    scale_preds_loglinear = {}
    for test_id in LAKES:
        train_scale = scale_table_log.drop(index=test_id)
        lin_model = LinearRegression()
        lin_model.fit(train_scale[loglinear_cols], np.log(train_scale["max_depth"]))
        pred_log = lin_model.predict(scale_table_log.loc[[test_id], loglinear_cols])[0]
        scale_preds_loglinear[test_id] = np.exp(pred_log)

    scale_summary_log = pd.DataFrame({
        "max_depth_reel": scale_table["max_depth"],
        "max_depth_predit": pd.Series(scale_preds_loglinear),
    })
    scale_summary_log["erreur_pct"] = (
        (scale_summary_log["max_depth_predit"] - scale_summary_log["max_depth_reel"])
        / scale_summary_log["max_depth_reel"] * 100
    )
    print("\nEtage 1 (echelle log-lineaire), leave-one-lake-out :")
    print(scale_summary_log.round(1))

    print("\nModele combine (echelle etage 1 log-lineaire x ratio etage 2), profondeur absolue :")
    for test_id in LAKES:
        depth_true = depth_trues[test_id]
        depth_pred_combined = ratio_preds[test_id] * scale_preds_loglinear[test_id]
        r2_combined = r2_score(depth_true, depth_pred_combined)
        rmse_combined = root_mean_squared_error(depth_true, depth_pred_combined)
        flag = " <-- lac probleme" if test_id in problem_lakes else ""
        print(f"{test_id:15s} combine: R2={r2_combined:7.3f} RMSE={rmse_combined:7.2f}m{flag}")

    # resultat : pire que le RF (L9 +1711% vs +636%) -- manque de lacs, pas le modele


def build_lake_scale_table(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une ligne par lac : features macro (taille, forme, relief moyen alentour) +
    profondeur max reelle. Sert a l'etage 1 du modele du jour 21 (predire l'echelle du
    lac independamment de la forme relief->profondeur traitee par depth_ratio)."""
    elevdiff900_cols = [c for c in next(iter(datasets.values())).columns if "elevdiff900" in c]
    elevdiff300_cols = [c for c in next(iter(datasets.values())).columns if "elevdiff300" in c]
    dist_cols = [c for c in next(iter(datasets.values())).columns if c.endswith("shoreline_distance")]

    rows = {}
    for lake_id, df in datasets.items():
        rows[lake_id] = {
            "surface_area": df["surface_area"].iloc[0],
            "isoperimetric_coeff": df["isoperimetric_coeff"].iloc[0],
            "mean_elevdiff900": df[elevdiff900_cols].mean().mean(),
            "mean_elevdiff300": df[elevdiff300_cols].mean().mean(),
            "mean_shore_dist": df[dist_cols].mean().mean(),
            "max_depth": df[DEPTH_TARGET_COL].abs().max(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


if __name__ == "__main__":
    main()
