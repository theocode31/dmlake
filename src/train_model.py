"""
Jour 4 : premier modele ML, prediction de la profondeur a partir des features
de relief extraites au jour 3.

Constat avant d'entrainer : angle45 et angle225 (un seul axe, deux sens
opposes) ont ~60% de valeurs manquantes -- le grand axe du Lac de Joux est
quasiment aligne avec cette diagonale, donc les rayons tires dans ces deux
directions depuis un point central doivent parcourir presque toute la
longueur du lac avant de toucher la terre, ce qui depasse le rayon de
recherche du rivage (2500m) et la couronne MNT (1km) du jour 2/3. Ce n'est
pas un bug d'extraction, c'est une consequence geometrique de la forme du
lac -- mais garder ces colonnes ferait perdre ~64% des lignes au dropna.
Ces deux directions sont donc exclues des features (a corriger plus tard en
elargissant la couverture MNT si besoin), les 6 autres angles sont gardes.

Protocole d'evaluation : reprend celui de Kacimi pour rester comparable
(split 75/25, Random Forest, RMSE/MAE/R², feature importance).

Limite connue, non corrigee : le split train/test est aleatoire par point,
sur un seul lac. Les features spatiales (distances au rivage) agissent comme
une quasi-empreinte de position (x, y) -- un point de test peut donc etre
spatialement tres proche d'un point d'entrainement, et le modele interpole
plutot qu'il ne generalise une vraie relation relief -> profondeur. Kacimi
evite ce biais en splittant par survey_id (lac entier en train ou en test),
impossible ici avec un seul lac. Plutot qu'un split spatial artificiel du
Lac de Joux (bricolage difficile a interpreter), le choix est de documenter
la limite et d'evaluer la generalisation reelle avec un deuxieme lac
(entrainement sur un lac, test sur l'autre) des que les features seront
disponibles pour ce second lac.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

FEATURES_CSV = "outputs/lacdejoux_features.csv"
EXCLUDED_ANGLES = [45, 225]  # axe aligne avec le grand axe du lac, ~60% de NaN
NON_FEATURE_COLS = ["x", "y", "z", "survey_id"]
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


if __name__ == "__main__":
    main()
