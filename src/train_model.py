import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

FEATURES_CSV = "outputs/lacdejoux_features.csv"
EXCLUDED_ANGLES = []  # plus necessaire depuis le rayon de recherche dynamique (jour 6, extract_features.py)
NON_FEATURE_COLS = ["x", "y", "z", "survey_id"]
DEPTH_TARGET_COL = "angle0_z_DEM_ref"  # profondeur relative au rivage -- cf load_dataset_for
DEPTH_RATIO_COL = "depth_ratio"  # profondeur normalisee par la profondeur max du lac -- cf load_dataset_for
LAKES_CH = [
    "lacdejoux", "lungernsee", "aegerisee", "baldeggersee", "hallwilersee",
    "bielersee", "brienzersee", "lacneuchatel", "lagomaggiore",
]  # bodensee (0 pt valide) et lacleman (18 pts) exclus -- MNT suisse ne couvre pas la partie du lac hors de Suisse
LAKES_FR = ["L1", "L2", "L9", "L17", "L30", "L40", "L50", "L60", "L66", "L70", "L80", "L90"]
LAKES = LAKES_CH + LAKES_FR
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    drop_cols = [c for c in df.columns if any(c.startswith(f"angle{a}_") for a in EXCLUDED_ANGLES)]
    df = df.drop(columns=drop_cols)
    # les flags shore_extrapolated ne sont pas des features numeriques exploitables telles
    # quelles ici (0% d'extrapolation observe au jour 3) -- on les retire du jeu de features
    extrap_cols = [c for c in df.columns if c.endswith("_shore_extrapolated")]
    df = df.drop(columns=extrap_cols)
    # angle0_z_DEM_ref = z - altitude_rivage (jour 3) : quasi-reformulation lineaire de la
    # cible z elle-meme (l'altitude du rivage varie tres peu sur un meme lac) -> fuite de
    # donnees si on la garde comme feature. Chez Kacimi c'est justement l'inverse : cette
    # variable sert de CIBLE alternative (profondeur relative au rivage), pas de feature.
    df = df.drop(columns=["angle0_z_DEM_ref"])
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
    # profondeur normalisee par la profondeur max observee sur CE lac (min de
    # DEPTH_TARGET_COL, qui est <= 0) -- sert a tester si la relation relief->forme
    # de la profondeur generalise mieux une fois decouplee de l'echelle absolue du
    # lac (jour 7 : L9, tres peu profond, avait un R2=-171 avec la cible en metres).
    df[DEPTH_RATIO_COL] = df[DEPTH_TARGET_COL] / df[DEPTH_TARGET_COL].min()
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

    # Test de generalisation, leave-one-lake-out sur LAKES : train sur tous les lacs sauf
    # un, test sur celui laisse de cote, a tour de role. Cible = DEPTH_TARGET_COL (profondeur
    # relative au rivage), pas z (altitude absolue) : un premier essai avec z donnait
    # RMSE~344m d'un lac a l'autre, simplement parce que Lac de Joux (~1000m d'altitude) et
    # Lungernsee (~650m) n'ont pas la meme altitude de base -- z absolu n'a aucun sens a
    # comparer entre deux lacs.
    datasets = {lake_id: load_dataset_for(lake_id) for lake_id in LAKES}
    for lake_id, d in datasets.items():
        print(f"{lake_id}: {len(d)} lignes apres nettoyage")

    # isoperimetric_coeff (jour 8) n'existe que sur les lacs suisses re-extraits -- exclu ici
    # pour que le jeu de features reste identique entre lacs suisses et francais (cf plus bas
    # pour un test dedie, lacs suisses uniquement, qui l'inclut).
    cross_feature_cols = [c for c in feature_cols if c not in (DEPTH_TARGET_COL, "isoperimetric_coeff")]
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, datasets[test_id], cross_feature_cols, DEPTH_TARGET_COL, f"train=autres lacs / test={test_id}")

    # Meme leave-one-lake-out, mais cible = DEPTH_RATIO_COL (profondeur / profondeur max
    # du lac, donc sans unite, comparable entre un lac de 6m et un lac de 260m). Le R2 est
    # calcule directement sur cette echelle normalisee -- ce test isole la question "la
    # forme relief->profondeur generalise-t-elle ?" de la question separee "quelle est
    # l'echelle absolue de ce lac ?" (qu'il faudrait resoudre autrement en pratique, ex.
    # via surface_area, pas testee ici).
    print("\n\n=== Meme test, cible normalisee (depth_ratio) ===")
    ratio_feature_cols = [c for c in cross_feature_cols if c != DEPTH_RATIO_COL]
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, datasets[test_id], ratio_feature_cols, DEPTH_RATIO_COL, f"[ratio] train=autres lacs / test={test_id}")

    # Coefficient isoperimetrique (jour 8, piste de Kacimi) : uniquement sur les lacs suisses
    # pour l'instant, les lacs francais repris du stage n'ont pas encore cette colonne.
    # Kacimi trouve que ce coefficient seul (sans surface_area) donne le meilleur R2 --
    # on teste donc les deux variantes en cible normalisee (depth_ratio).
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

    # Test final, jour 9 : les 12 lacs francais ont maintenant aussi isoperimetric_coeff
    # (pipeline france code de zero, cf fetch_dem_france.py/extract_features_france.py),
    # donc plus besoin de se limiter aux lacs suisses. Configuration retenue d'apres les
    # tests ci-dessus : cible normalisee (depth_ratio), isoperimetric_coeff SANS surface_area.
    all_feature_cols = [c for c in datasets["lacdejoux"].columns if c not in NON_FEATURE_COLS + [DEPTH_TARGET_COL, DEPTH_RATIO_COL, "surface_area"]]
    print("\n\n=== Test final, 21 lacs (9 CH + 12 FR), depth_ratio + isoperimetric_coeff sans surface_area ===")
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, datasets[test_id], all_feature_cols, DEPTH_RATIO_COL, f"[FINAL] test={test_id}")

    # Isolation : le test [FINAL] est nettement pire que l'ancien test [ratio] (5/21 vs 15/21
    # lacs en R2 positif), mais deux choses ont change en meme temps -- le remplacement
    # surface_area -> isoperimetric_coeff, ET les nouvelles donnees francaises recalculees par
    # notre pipeline. Ce test isole la premiere variable : meme feature set que l'ancien test
    # [ratio] (surface_area, sans isoperimetric_coeff), mais sur les nouvelles donnees FR.
    surface_only_cols = [c for c in all_feature_cols if c != "isoperimetric_coeff"] + ["surface_area"]
    print("\n\n=== Isolation : 21 lacs, depth_ratio + surface_area (sans isoperimetric_coeff), nouvelles donnees FR ===")
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, datasets[test_id], surface_only_cols, DEPTH_RATIO_COL, f"[ISOLATION-surface] test={test_id}")


if __name__ == "__main__":
    main()
