"""[S25] replication on the FIADB table (paper/FIA_REBUILD_DESIGN.md section 7): stage the inputs of the frozen
experiments/make_stress.py so that it runs UNCHANGED, with the fia_db results in place of the old FIA table.

Into results/fia_db/stress/ go st_twomap_fia.csv (arm A = results/st_twomap_fia_db.csv), st_signal_fia.csv (arm B FIA
setting = results/st_signal_fia_db.csv, dataset column relabelled 'fia') and the other ten st_signal_<ds>.csv copied
byte for byte from results/. No fia_post2000 file (the sensitivity arm does not apply). Then run
    QW2_FIA=fia STRESS_OUT=results/fia_db/stress ~/miniforge3/envs/favela/bin/python -B experiments/make_stress.py
(QW2_FIA=fia keeps the frozen script's dataset names; STRESS_OUT keeps its outputs out of results/).
"""
import os, shutil
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "fia_db", "stress")
OTHERS = ["poverty", "poverty_lab", "camels", "voting", "seattle", "era5",
          "health_stroke", "health_diabetes", "health_bphigh", "health_obesity"]


def main():
    os.makedirs(OUT, exist_ok=True)
    shutil.copyfile(os.path.join(RES, "st_twomap_fia_db.csv"), os.path.join(OUT, "st_twomap_fia.csv"))
    b = pd.read_csv(os.path.join(RES, "st_signal_fia_db.csv"))
    assert set(b.dataset) == {"fia_db"}, set(b.dataset)
    b.assign(dataset="fia").to_csv(os.path.join(OUT, "st_signal_fia.csv"), index=False)
    for nm in OTHERS:
        shutil.copyfile(os.path.join(RES, f"st_signal_{nm}.csv"), os.path.join(OUT, f"st_signal_{nm}.csv"))
    print("staged", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
