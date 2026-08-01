import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

    # Jour 21 -- le probleme d'echelle (jour 16-20) persiste meme avec depth_ratio : les lacs
    # hors gamme de profondeur "normale" du jeu d'entrainement (lungernsee, bielersee,
    # lacneuchatel, lagomaggiore, L9) restent en R2 negatif, parce que depth_ratio decouple la
    # FORME (relief -> position relative dans le lac) de l'ECHELLE (profondeur max du lac), mais
    # cette echelle n'est ensuite jamais reconstituee : on compare des ratios entre eux, jamais
    # la profondeur reelle. Idee du jour 4 reprise ici : predire l'echelle separement, avec un
    # second modele "macro" (une ligne par lac, pas par point), puis recombiner
    # profondeur_predite = ratio_predit x echelle_predite. Verification prealable (corr. entre
    # features macro et profondeur max reelle sur les 21 lacs) : mean_shore_dist=0.77,
    # surface_area=0.61, isoperimetric_coeff=0.56, mean_elevdiff900=0.39 -- signal present,
    # notamment sur la taille/forme du lac, plus faible sur le relief environnant.
    print("\n\n=== Jour 21 : modele deux etages (echelle + forme) pour reconstituer la profondeur absolue ===")
    scale_table = build_lake_scale_table(datasets)
    scale_feature_cols = [c for c in scale_table.columns if c != "max_depth"]

    scale_preds = {}
    for test_id in LAKES:
        train_scale = scale_table.drop(index=test_id)
        # RF regularise (peu d'arbres, profondeur limitee) : seulement ~20 lignes en
        # entrainement ici, un RF aussi permissif que celui utilise au niveau point (jour 9,
        # max_depth=20) sur-apprendrait immediatement.
        scale_model = RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, max_depth=4, min_samples_leaf=2
        )
        # cible en log : la profondeur max varie sur presque 2 ordres de grandeur (5.6m a
        # 546m), un ecart absolu de qq metres compte beaucoup plus pour L9 (5.6m) que pour
        # brienzersee (546m) -- cf. le meme raisonnement qui a motive depth_ratio au jour 16.
        scale_model.fit(train_scale[scale_feature_cols], np.log(train_scale["max_depth"]))
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

    # Etage 2 : reprend le meme modele depth_ratio que le test [FINAL] ci-dessus (LOLO,
    # ratio_feature_cols), mais cette fois on recombine avec l'echelle predite a l'etage 1
    # pour obtenir une vraie profondeur en metres, comparable entre lacs. ratio_preds/depth_true
    # sont mis en cache par lac pour ne pas re-entrainer ce RF (couteux, 200 arbres) une
    # deuxieme fois plus bas pour la variante log-lineaire de l'etage 1.
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

    # Variante etage 1 : regression log-lineaire au lieu du RF. Un RF predit une moyenne de
    # feuilles vues a l'entrainement, donc ne peut jamais sortir de [min, max] des profondeurs
    # d'entrainement -- explique l'echec sur L9 (le plus petit lac exclu -> plancher impose par
    # le lac suivant le plus petit). Une regression lineaire log(surface)->log(profondeur), en
    # revanche, extrapole nativement le long de la droite ajustee -- c'est la relation
    # d'echelle classique utilisee en limnologie (loi de puissance aire/profondeur).
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

    # Resultat : pire que le RF, pas mieux. L9 (5.6m reel) predit a 100.7m (+1711%, contre
    # +636% avec le RF). L'hypothese "le RF ne peut pas extrapoler, un modele lineaire le
    # peut" etait correcte en theorie mais ne suffit pas ici en pratique : avec seulement ~20
    # lacs tres heterogenes (pipelines CH/FR differents, profondeur de 5.6m a 546m, donc 2
    # ordres de grandeur), un ajustement log-log sur 2 features suffit a peine a capter la
    # tendance generale, et sous leave-one-lake-out chaque lac exclu peut faire basculer la
    # pente de la droite. En espace log, une petite erreur de pente se traduit par une erreur
    # enorme une fois repassee en metres (exp()). Le probleme n'est donc pas le choix RF vs
    # lineaire, mais le manque de lacs pour estimer une relation d'echelle fiable -- 20 lacs,
    # c'est trop peu pour une regression a l'echelle du lac, meme simple.
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
