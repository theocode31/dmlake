import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from lacs import LAKES_CH, LAKES_FR, LAKES_ALREADY_SEEN
from train_model import charger_donnees_lac, NON_FEATURE_COLS, DEPTH_TARGET_COL, DEPTH_RATIO_COL, RANDOM_STATE

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
# en dessous, le R2 d'un lac ne veut pas dire grand-chose (14/07 : aegerisee a 91 points, R2 = -212)
MIN_POINTS = 200


def separer_lacs() -> tuple[list[str], list[str]]:
    nouveaux = [lid for lid in LAKES_FR if lid not in LAKES_ALREADY_SEEN]
    rng = np.random.default_rng(RANDOM_STATE)
    mis_de_cote = rng.choice(nouveaux, size=len(nouveaux) // 2, replace=False).tolist()
    mis_de_cote = sorted(mis_de_cote, key=lambda lid: int(lid[1:]))
    developpement = [lid for lid in LAKES_CH + LAKES_FR if lid not in mis_de_cote]
    return developpement, mis_de_cote


def charger_lacs(lacs: list[str]) -> dict[str, pd.DataFrame]:
    datasets = {}
    for lake_id in lacs:
        if not (OUTPUT_DIR / f"{lake_id}_features.csv").exists():
            print(f"{lake_id} : pas de CSV, lance d'abord extraire_tous_les_lacs.py")
            continue
        if (OUTPUT_DIR / f"{lake_id}_features.csv").stat().st_size < 10:
            print(f"{lake_id} : CSV vide (aucun point dans le masque d'eau), ignore")
            continue
        d = charger_donnees_lac(lake_id)
        if len(d) < MIN_POINTS:
            print(f"{lake_id} : {len(d)} points valides seulement (moins de {MIN_POINTS}), ignore")
            continue
        datasets[lake_id] = d
    return datasets


def colonnes_retenues(datasets: dict[str, pd.DataFrame]) -> list[str]:
    # config du 20/07 : depth_ratio + isoperimetric_coeff, sans surface_area
    exemple = next(iter(datasets.values()))
    exclues = NON_FEATURE_COLS + [DEPTH_TARGET_COL, DEPTH_RATIO_COL, "surface_area"]
    return [c for c in exemple.columns if c not in exclues]


def creer_modele() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, max_depth=20)


def mesurer(vrai, predit) -> dict:
    return {
        "R2": r2_score(vrai, predit),
        "RMSE": root_mean_squared_error(vrai, predit),
        "MAE": mean_absolute_error(vrai, predit),
        "n_points": len(vrai),
    }


def un_lac_de_cote(datasets: dict[str, pd.DataFrame], colonnes: list[str]) -> pd.DataFrame:
    resultats = {}
    for test_id, test_df in datasets.items():
        train_df = pd.concat([d for lid, d in datasets.items() if lid != test_id], ignore_index=True)
        modele = creer_modele().fit(train_df[colonnes], train_df[DEPTH_RATIO_COL])
        resultats[test_id] = mesurer(test_df[DEPTH_RATIO_COL], modele.predict(test_df[colonnes]))
    return pd.DataFrame.from_dict(resultats, orient="index")


def resumer(res: pd.DataFrame, titre: str) -> None:
    r2 = res["R2"]
    print(f"\n{titre} ({len(res)} lacs)")
    print(f"R2 median = {r2.median():.3f} (un lac sur deux entre {r2.quantile(0.25):.3f} et {r2.quantile(0.75):.3f}), "
          f"R2 > 0 pour {(r2 > 0).sum()} lacs sur {len(res)}")
    print(f"RMSE du ratio : mediane = {res['RMSE'].median():.3f}, moyenne = {res['RMSE'].mean():.3f}")


def comparer(res_a: pd.DataFrame, res_b: pd.DataFrame, nom_a: str, nom_b: str) -> None:
    communs = res_a.index.intersection(res_b.index)
    ecart = res_b.loc[communs, "RMSE"] - res_a.loc[communs, "RMSE"]
    print(f"\n{nom_b} contre {nom_a}, lac par lac ({len(communs)} lacs) : "
          f"{(ecart < 0).sum()} lacs mieux, {(ecart > 0).sum()} moins bien, "
          f"ecart median de RMSE = {ecart.median():+.3f}")


def main() -> None:
    developpement, mis_de_cote = separer_lacs()
    print(f"{len(developpement)} lacs de developpement, {len(mis_de_cote)} mis de cote : {', '.join(mis_de_cote)}")

    dev = charger_lacs(developpement)
    if len(dev) < 2:
        print("Il faut au moins 2 lacs de developpement extraits.")
        return
    colonnes = colonnes_retenues(dev)

    if "--test-final" in sys.argv:
        # le leave-one-lake-out ne change pas le test final, et il prend 3 h 30 par config
        tester_lacs_mis_de_cote(dev, mis_de_cote, colonnes)
        return

    res_retenue = un_lac_de_cote(dev, colonnes)
    resumer(res_retenue, "Config retenue, un lac de cote")
    print(res_retenue.round(3).to_string())

    # exemple de comparaison lac par lac : le choix du 20/07 (coefficient isoperimetrique),
    # fait sur 9 lacs suisses avec la moyenne des R2, revu sur tous les lacs de developpement
    colonnes_sans_iso = [c for c in colonnes if c != "isoperimetric_coeff"]
    res_sans_iso = un_lac_de_cote(dev, colonnes_sans_iso)
    resumer(res_sans_iso, "Sans isoperimetric_coeff, un lac de cote")
    comparer(res_sans_iso, res_retenue, "sans iso", "avec iso")

    print("\nTest final pas lance. Quand plus aucun choix ne doit changer : "
          "python src/evaluation_finale.py --test-final")


def tester_lacs_mis_de_cote(dev: dict[str, pd.DataFrame], mis_de_cote: list[str], colonnes: list[str]) -> None:
    test = charger_lacs(mis_de_cote)
    if not test:
        print("Aucun lac mis de cote n'a de CSV.")
        return
    train_df = pd.concat(dev.values(), ignore_index=True)
    modele = creer_modele().fit(train_df[colonnes], train_df[DEPTH_RATIO_COL])
    res_test = pd.DataFrame.from_dict(
        {lid: mesurer(d[DEPTH_RATIO_COL], modele.predict(d[colonnes])) for lid, d in test.items()},
        orient="index",
    )
    resumer(res_test, "TEST FINAL sur les lacs mis de cote (entraine sur tous les lacs de developpement)")
    print(res_test.round(3).to_string())
    res_test.to_csv(OUTPUT_DIR / "test_final.csv", index_label="lac")
    tracer_test_final(res_test, len(dev))


def tracer_test_final(res: pd.DataFrame, nb_lacs_dev: int) -> None:
    res = res.sort_values("R2")
    couleurs = ["#c8553d" if r < 0 else "#2a7f9e" for r in res.R2]
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(res.index, res.R2, color=couleurs, height=0.7)
    ax.axvline(0, color="#444444", linewidth=0.8)
    mediane = res.R2.median()
    ax.axvline(mediane, color="#444444", linewidth=1, linestyle="--")
    ax.text(mediane, len(res) - 0.4, f" mediane = {mediane:.2f}", va="bottom", fontsize=9)
    ax.set_xlabel("R2 sur le ratio de profondeur (rouge = pire que predire la moyenne)")
    ax.set_title(f"Test final : {len(res)} lacs francais jamais vus\n"
                 f"(Random Forest entraine sur les {nb_lacs_dev} lacs de developpement)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for cote in ["top", "right"]:
        ax.spines[cote].set_visible(False)
    out_path = OUTPUT_DIR / "test_final_r2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardee : {out_path}")


if __name__ == "__main__":
    main()
