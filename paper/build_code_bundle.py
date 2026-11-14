"""The code and results packages for peer review: paper/build/QW2_code_for_review.zip (code, small data, figure data) and
paper/build/QW2_results_for_review.zip (per-split results and logs), split so that the code can be uploaded with the manuscript.
paper/build_github_release.py turns the code archive into the public repository.

Follows paper/CODE_DEPOSIT_PLAN.md. DHS-derived data never leave this project: the cluster table, every unit-level file (.npz), the
cache and the deep-backbone prediction files are left out, and DHS results keep only country- and split-level rows (rows for
spatial cells, which can be smaller than a country, are dropped). The one unit-level file kept is the FIA split behind Fig. 1b
(FIA plots are public domain and already in data/). The script then scans the package and stops if anything unit-level from DHS
is found.
    ~/miniforge3/envs/favela/bin/python paper/build_code_bundle.py
"""
import glob
import gzip
import hashlib
import io
import os
import re
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import build_manuscript_pdf                                           # noqa: E402

OUT = {"code": os.path.join(HERE, "build", "QW2_code_for_review.zip"), "results": os.path.join(HERE, "build", "QW2_results_for_review.zip")}
CODE = ["experiments", "excp_select", "vendor/excp_20261021", "figures", "tests", "package", "website"]
FILES = ["data/fia_fiadb95_plots.csv", "data/fia_cci_v7_2020.csv", "paper/build_code_bundle.py", "paper/build_github_release.py"]
DIRS_DATA = ["data/natural_earth_110m_v5.1.2", "figs/source_data"]
SKIP = re.compile(r"(__pycache__|\.pyc$|\.DS_Store$|/build/|\.egg-info|/\.|website/upload|MERGE_PROPOSAL\.md$)")   # last: internal note
DHS = re.compile(r"poverty|togo|dhs", re.I)
ALLOW_NPZ = {"results/units_fia_db_rep0.npz"}                          # FIA split of Fig. 1b; no DHS units
ENV = ("Python 3.11 with numpy, scipy, pandas, scikit-learn and matplotlib; rasterio for the ESA CCI extraction. "
       "Runs used at most four processes.")
# 2026-11-14 pre-submission check: logs carried this machine's absolute paths (user name, Google Drive folder, e-mail address).
# Text files are stored with the project root shortened to QW2 and the home folder to ~; the final scan stops on any trace left.
TEXT = (".py", ".md", ".csv", ".txt", ".log", ".err", ".json", ".jsonl", ".cff", ".toml", ".sh", ".html", ".js", ".in",
        ".sha256", ".geojson")
HOME = os.path.expanduser("~")
LOCAL = [(re.compile(r"/(?:Users|home)/[^/\s'\"]+/[^\s'\"]*?/" + re.escape(os.path.basename(ROOT)) + r"(?![\w-])"), "QW2"),
         (re.compile(r"/private/(?:tmp|var)/[^\s'\"]*"), "<tmp>"),
         (re.compile(r"/(?:Users|home)/[^/\s'\"]+"), "~")]           # any machine the runs were made on, not only this one
GENERIC = re.compile(r"/Users/|/home/[a-z]|GoogleDrive|CloudStorage|@gmail\.com|云端硬盘")   # every file but the two build scripts
OWN = ("QW2/paper/build_code_bundle.py", "QW2/paper/build_github_release.py")
_drive = {c for c in ROOT.split(os.sep) if "GoogleDrive" in c or "@" in c or "云端" in c}
PERSONAL = re.compile("|".join(re.escape(s) for s in sorted(
    {HOME, os.path.basename(HOME)} | _drive | {re.split(r"[-@]", c)[1] for c in _drive if c.startswith("GoogleDrive-")}, key=len, reverse=True)))
CACHE_KEEP = re.compile(r"_frozen|FROZEN_SHA256|_build_log\.json$|SHA256SUMS|^results/cache/tests/")   # cited in the SI Appendix


def title():
    """The manuscript title, from the title-page draft (one source for both READMEs)."""
    t = open(os.path.join(HERE, "TITLE_PAGE_draft.md"), encoding="utf-8").read()
    return re.search(r"^\*\*Title\.\*\* (.+)$", t, re.M).group(1).strip()


def results_files():
    out = []
    for f in glob.glob(os.path.join(ROOT, "results", "**", "*"), recursive=True):
        r = os.path.relpath(f, ROOT)
        if not os.path.isfile(f) or SKIP.search(r) or (r.endswith(".npz") and r not in ALLOW_NPZ):
            continue
        if r.startswith("results/cache/"):
            if CACHE_KEEP.search(os.path.basename(r)) or CACHE_KEEP.search(r):
                out.append(r)       # frozen designs and digests (the preregistration record); extraction logs, download checksums, test logs
            continue
        if r.startswith("results/old_fia_table"):
            continue
        out.append(r)
    return sorted(out)


def payload(rel):
    """Bytes to store. DHS result tables lose their spatial-cell rows, and a table with one row per selected village (the
    pilot's results/pilot_l2_poverty_lab.csv) becomes its country-and-split summary (2026-11-14). Text files have local paths
    shortened."""
    path = os.path.join(ROOT, rel)
    if DHS.search(os.path.basename(rel)) and rel.endswith((".csv", ".csv.gz")):
        d = pd.read_csv(path, low_memory=False)
        bad = [c for c in d.columns if c.lower() in ("lat", "lon", "latitude", "longitude", "wealthpooled", "cluster", "idx")]
        if bad:
            raise SystemExit(f"{rel} has unit-level columns {bad}")
        if "gtype" in d.columns:
            d = d[d["gtype"].astype(str) != "cell"]
        if {"cov", "finite", "hw"} <= set(d.columns):            # one row per selected unit
            keys = [c for c in d.columns if c not in ("cov", "finite", "hw", "R")]
            d = (d.assign(hw_fin=d["hw"].where(d["finite"].astype(bool)))
                 .groupby(keys, as_index=False, dropna=False)
                 .agg(n=("cov", "size"), n_cov=("cov", "sum"), n_finite=("finite", "sum"), hw_median_finite=("hw_fin", "median"),
                      hw_mean_finite=("hw_fin", "mean"), R_mean=("R", "mean")))
        if "cov" in d.columns and "n" not in d.columns:
            raise SystemExit(f"{rel}: per-unit DHS rows (column 'cov') left in the archive")
        buf = io.BytesIO()
        d.to_csv(buf, index=False, compression="gzip" if rel.endswith(".gz") else None)
        b = buf.getvalue()
    else:
        b = open(path, "rb").read()
    if rel.endswith(TEXT) and not rel.endswith(".py"):          # code is stored as written; the final scan checks it
        s = b.decode("utf-8", "surrogateescape")
        for a, c in LOCAL:
            s = a.sub(c, s)
        b = s.encode("utf-8", "surrogateescape")
    return b


def contents(github=False):
    """'Contents' and 'Not included', shared by README_REVIEW.md and the public repository's README.md."""
    res = "- `results/` (the results archive; see Results above): " if github else "- `results/`: "
    return [
        "## Contents",
        "- `package/`: mapselect, the methods as an installable Python package (numpy, scipy; MIT licence). `pip install ./package`, then `python package/tests/test_mapselect.py` (20 tests); `python tests/test_planner.py` checks the audit planner against enumeration.",
        "- `excp_select/`, `vendor/excp_20261021/`: the code behind the results (closed-form reference sets; SC-GeoCCP, which the paper calls RegionCP; total bounds) and the conformal engine it uses.",
        "- `experiments/`: every run, summary and judgement script; each file's docstring gives its command and outputs.",
        "- `figures/`: figure scripts, each naming its inputs in its docstring; `figs/source_data/`: figure data under working names (`paper/build_source_data.py` copies them under the final figure numbers, as uploaded with the manuscript).",
        "- `tests/`: brute-force swap, Monte Carlo coverage, pool-lemma and SC-GeoCCP checks.",
        res + "per-split and per-group results, logs and the `RESULTS_*.md` tables cited in the SI Appendix; `results/cache/*_frozen*` hold the designs frozen before each run, with SHA-256 digests (the design notes are in Chinese); `results/cache/` also keeps the FIA and ESA CCI extraction logs, the checksums of the raw downloads and the swap-test log. In logs, local paths are shortened to `QW2/` and `~/`.",
        "- `data/`: the FIA plot table rebuilt from FIADB 9.5 and the ESA CCI values at the plots; Natural Earth boundaries.",
        "- `website/`: the interactive site (https://peng-luo.com/mapselect/) and the script that builds its data.",
        "- `paper/`: the designs frozen before each run (`*_DESIGN.md`, with their execution records); `build_code_bundle.py`, which made the review archives and removed the DHS rows; and `build_github_release.py`, which made the public repository from the code archive (both read manuscript sources, which are not included).",
        "",
        "## Not included",
        "- DHS data: the cluster table, unit-level predictions and labels, and the deep-backbone prediction files, under the DHS terms (access: https://dhsprogram.com; PovertyMap-WILDS). DHS results appear only as country and split summaries: rows for spatial cells were removed, and the pilot's per-village table `results/pilot_l2_poverty_lab.csv` is replaced by its summary by country and split (selected, covered and finite counts, median and mean finite half-width, mean reference-set size).",
        "- Unit-level files of the other sources, except the FIA split behind Fig. 1b (`results/units_fia_db_rep0.npz`).",
        "- Raw downloads (FIADB state files, ESA CCI tiles, WILDS image features) and the run cache; the scripts give their sources.",
        "- Other data sources (ERA5, CAMELS, county vote share, King County sales, CDC PLACES) are read from their published tables; see the data-source table of the SI Appendix.",
    ]


def labels(N):
    """Working labels of supplementary items and their final labels (from the manuscript build)."""
    return ["## Working codes in the code and notes",
            "Scripts and result tables use working labels for supplementary items. Their final labels, as in the PDFs (sub-tables and figure panels keep their letters; sections of a note keep their numbers):",
            "",
            "| Working label | Final |", "|---|---|"] + [f"| {a} | {b} |" for a, b in N.mapping()]


def readme(n_files, N):
    return "\n".join([
        "# Code and results for peer review",
        "",
        f"Manuscript: *{title()}*. Provided for review; the same code is in the public repository given in the Code availability statement. Two archives unpack into one folder `QW2/`: the code archive and the results archive.",
        "",
        *contents(),
        "",
        "## Environment",
        ENV,
        "",
        *labels(N),
        "", f"{n_files} files."]) + "\n"


def main():
    files = []
    for d in CODE + DIRS_DATA:
        for f in glob.glob(os.path.join(ROOT, d, "**", "*"), recursive=True):
            r = os.path.relpath(f, ROOT)
            if os.path.isfile(f) and not SKIP.search("/" + r):
                files.append(r)
    files += [f for f in FILES if os.path.exists(os.path.join(ROOT, f))]
    files += results_files()
    files += [os.path.relpath(f, ROOT) for f in glob.glob(os.path.join(HERE, "*_DESIGN.md"))]
    files = sorted(set(files))
    forbidden = [f for f in files if f not in ALLOW_NPZ and re.search(
        r"povertymap_clusters|units_.*\.npz|\.npz$|togo_map", f) or (f.startswith("results/cache/") and not CACHE_KEEP.search(f))
                 or (f.startswith("results/cache/") and DHS.search(f))]
    if forbidden or any(DHS.search(f) for f in ALLOW_NPZ):
        raise SystemExit(f"forbidden files selected: {forbidden[:5]}")
    N = build_manuscript_pdf.numbering()
    os.makedirs(os.path.dirname(OUT["code"]), exist_ok=True)
    parts = {"code": [f for f in files if not f.startswith("results/")], "results": [f for f in files if f.startswith("results/")]}
    for kind, names in parts.items():
        digests = []
        with zipfile.ZipFile(OUT[kind], "w", zipfile.ZIP_DEFLATED) as z:
            for r in names:
                b = payload(r); z.writestr("QW2/" + r, b); digests.append(f"{hashlib.sha256(b).hexdigest()}  {r}")
            z.writestr("QW2/README_REVIEW.md", readme(len(files), N))
            z.writestr(f"QW2/MANIFEST_{kind}.sha256", "\n".join(digests) + "\n")
        with zipfile.ZipFile(OUT[kind]) as z:                       # final scan of what was written
            for n in z.namelist():
                raw = z.read(n)
                if n.endswith(".gz"):
                    raw = gzip.decompress(raw)
                s = raw.decode("utf-8", "ignore")
                m = PERSONAL.search(s) or (GENERIC.search(s) if n not in OWN else None)   # local paths, user name, e-mail
                if m:
                    raise SystemExit(f"local trace in {n}: ...{s[max(0, m.start() - 40):m.end() + 20]!r}")
                if DHS.search(os.path.basename(n)) and re.search(r"\.csv(\.gz)?$", n):
                    head = s[:4000].split("\n")[0].lower()
                    if re.search(r"\b(lat|lon|wealthpooled|cluster)\b", head) or re.search(r"(^|,)cov(,|$)", head):
                        raise SystemExit(f"unit-level DHS columns in {n}")
        print(f"{kind}: {len(names)} files, {os.path.getsize(OUT[kind]) / 2 ** 20:.1f} MB -> {OUT[kind]}")


if __name__ == "__main__":
    main()
