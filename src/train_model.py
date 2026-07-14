import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

FEATURES_CSV = "outputs/lacdejoux_features.csv"
EXCLUDED_ANGLES = [45, 225]  # axe aligne avec le grand axe du lac, ~60% de NaN
NON_FEATURE_COLS = ["x", "y", "z", "survey_id"]
DEPTH_TARGET_COL = "angle0_z_DEM_ref"  # profondeur relative au rivage -- cf load_dataset_for
LAKES = ["lacdejoux", "lungernsee", "aegerisee", "baldeggersee", "hallwilersee"]
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
    return df.dropna()


def evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str, label: str) -> None:
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
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
    print(f"{len(df)} lignes apres exclusion angle45/angle225, {df.shape[1]} colonnes")

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

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
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

    cross_feature_cols = [c for c in feature_cols if c != DEPTH_TARGET_COL]
    for test_id in LAKES:
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        evaluate(train_df, datasets[test_id], cross_feature_cols, DEPTH_TARGET_COL, f"train=autres lacs / test={test_id}")


if __name__ == "__main__":
    main()
