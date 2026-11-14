"""The public GitHub repository, made from the code archive for review (run after paper/build_code_bundle.py).

The repository holds the files of paper/build/QW2_code_for_review.zip byte for byte (MANIFEST_code.sha256 lists their
digests), without the 'QW2/' prefix and with README_REVIEW.md replaced by README.md, plus LICENSE (MIT), CITATION.cff and
.gitignore. The results archive (paper/build/QW2_results_for_review.zip, about 150 MB) exceeds GitHub's 100 MB file limit and
is meant to be attached to the GitHub release; README.md says how to unpack it.
    ~/miniforge3/envs/favela/bin/python -I -B paper/build_github_release.py
writes github_release/winners-curse-of-maps/. Since 2026-11-14 the author's clone of the repository (GitHub Desktop) is
github_release/winners-curse-of-maps/winners-curse-of-maps/: when it exists, the files are written there instead, everything in
it except .git is replaced (git shows the changes before any commit), and the outer folder keeps only the clone. A .git
anywhere else under the output folder stops the script rather than be deleted.
"""
import fnmatch
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import build_code_bundle as B                                         # noqa: E402
import build_manuscript_pdf                                            # noqa: E402

OUT = os.path.join(ROOT, "github_release", "winners-curse-of-maps")
CLONE = os.path.join(OUT, "winners-curse-of-maps")                 # the author's git clone (2026-11-14)
AUTHOR = ("Luo", "Peng", "School of Earth, Environment, and Sustainability, University of Iowa")
REPO = "https://github.com/pengtum/winners-curse-of-maps"      # confirmed by the author, 2026-11-10
IGNORE = ["__pycache__/", "*.py[cod]", "*.egg-info/", ".DS_Store", ".ipynb_checkpoints/", "",
          "# written by the scripts (the results archive is attached to the GitHub release)",
          "results/", "figs/*.pdf", "figs/*.png", "figs/*.svg", "figs/qa/", "website/upload/"]


def pattern(names):
    return re.compile("|".join(re.escape(s) for s in sorted(names) if len(s) > 3))


HOME = os.path.expanduser("~")
# names of this machine and its accounts, found at run time so that none is written in this file (checked in every file)
PERSONAL = pattern({HOME, os.path.basename(HOME)} | {c for c in ROOT.split(os.sep) if "GoogleDrive" in c or "@" in c})
GENERIC = pattern({"/Users/", "GoogleDrive", "@gmail.com"})              # local-path traces (every file but this one)
PUBLIC_URLS = (REPO, "https://peng-luo.com/mapselect/")                 # author-confirmed public destinations


def keywords():
    t = open(os.path.join(HERE, "TITLE_PAGE_draft.md"), encoding="utf-8").read()
    return [k.strip() for k in re.search(r"^\*\*Keywords\.\*\* (.+?)\.?$", t, re.M).group(1).split(";")]


def readme(N):
    return "\n".join([
        "# Winner's curse of maps",
        "",
        f"Code, small data and figure data for the manuscript *{B.title()}* ({AUTHOR[1]} {AUTHOR[0]}, {AUTHOR[2]}; 2026).",
        "",
        "Maps made by machine learning are used to choose places: the forest plots with the most carbon for credits, the poorest "
        "villages for aid. Intervals that are valid on average over a map can fail at exactly the places such a rule selects. "
        "The Python package **mapselect** (`package/`) calibrates each selected place only on audited places that the same rule "
        "would have selected, gives coverage floors for every region (RegionCP), bounds selected totals, and tells you before any "
        "field work how large the audit has to be. The rest of the repository holds every experiment, figure and test of the paper.",
        "",
        "## Quick start",
        "```bash",
        "pip install ./package",
        "python package/tests/test_mapselect.py   # 20 tests",
        "python tests/test_planner.py             # the audit planner against enumeration",
        "```",
        "`package/README.md` shows the package in use. " + B.ENV,
        "",
        "## Results",
        "Per-split and per-group results, logs and the result tables cited in the SI Appendix are in `QW2_results_for_review.zip` "
        f"(about {round(os.path.getsize(B.OUT['results']) / 1e6, -1):.0f} MB), attached to the GitHub release because it exceeds "
        "GitHub's file-size limit. Unpack it so that `results/` "
        "sits at the top of the repository:",
        "```bash",
        "unzip QW2_results_for_review.zip 'QW2/results/*' && mv QW2/results results && rm -r QW2",
        "```",
        "Six of the seven main figures then redraw from these tables and `data/` (`python figures/make_fig_regions.py` draws Fig. 6; "
        "each script's docstring names its figure and inputs). Fig. 4 (aid targeting) also needs the DHS files, which are not "
        "shared, as do the supplementary figures built on the deep-backbone predictions.",
        "",
        *B.contents(github=True),
        "",
        "## Data read by the scripts",
        "Included, under their own terms:",
        "- `data/fia_fiadb95_plots.csv`: forest plots rebuilt from the USDA Forest Service FIA database by "
        "`experiments/build_fia_fiadb.py` (FIADB 9.5; https://doi.org/10.2737/RDS-2001-FIADB; public domain).",
        "- `data/fia_cci_v7_2020.csv`: ESA CCI Biomass v7.0 (2020) at those plots, by `experiments/build_fia_cci.py` "
        "(https://doi.org/10.5285/6429d1aafe1e43b9b414e4a5a7f8b903), under the Biomass CCI terms, which require acknowledging "
        "the ESA Climate Change Initiative and the Biomass CCI project.",
        "- `data/natural_earth_110m_v5.1.2/`: Natural Earth boundaries (public domain).",
        "",
        "The other loaders in `experiments/data.py` read the published source tables (file names in `experiments/data.py`; "
        "sources, versions and licences in the data-source table of the SI Appendix) from a folder "
        "`0-Xiayin_research/15_RobustGeoCP/datasets/` next to the folder that holds this repository, as on the author's machine:",
        "```",
        "work/",
        "├── 0-Xiayin_research/15_RobustGeoCP/datasets/   source tables (not redistributed)",
        "└── code/",
        "    └── winners-curse-of-maps/                   this repository",
        "```",
        "`experiments/data.py` is kept as it was run, because its SHA-256 digest is part of a preregistration record "
        "(`results/cache/DOFA_FORMAL_FROZEN_SHA256.txt`). PovertyMap-WILDS features are read from `~/data/wilds`; DHS data are "
        "available from the DHS Program on registration and are not shared here.",
        "",
        "## Licence",
        "- Code: MIT (`LICENSE`), including `package/` and `vendor/excp_20261021/`, a copy of the author's conformal-inference "
        "package (imported as `excp`; current name geounicp).",
        "- Results, logs and figure data made in this study (`results/`, `figs/source_data/`): CC BY 4.0.",
        "- Data in `data/`: their own terms, listed above.",
        "",
        "## Citation",
        "`CITATION.cff` gives the citation of this code; the article reference will be added on publication.",
        "",
        *B.labels(N),
    ]) + "\n"


def citation():
    kw = "\n".join(f"  - {k}" for k in keywords())
    return (f'cff-version: 1.2.0\nmessage: "If you use this code, please cite it as below."\ntype: software\n'
            f'title: "Code for: {B.title()}"\nauthors:\n  - family-names: {AUTHOR[0]}\n    given-names: {AUTHOR[1]}\n'
            f'    affiliation: "{AUTHOR[2]}"\nrepository-code: "{REPO}"\nurl: "{REPO}"\nlicense: MIT\nkeywords:\n{kw}\n')


def licence():
    t = open(os.path.join(ROOT, "package", "LICENSE"), encoding="utf-8").read()
    assert t.startswith("MIT License") and f"Copyright (c) 2026 {AUTHOR[1]} {AUTHOR[0]}" in t, "package/LICENSE changed"
    return t


def ignored(f, p):
    """Would the .gitignore pattern p hide file f? (git's rules for the patterns used here)"""
    d, p, parts = p.endswith("/"), p.rstrip("/"), f.split("/")
    if "/" not in p:                                                   # any level
        return any(fnmatch.fnmatch(c, p) for c in (parts[:-1] if d else parts))
    if d:                                                              # anchored folder
        return any(fnmatch.fnmatch("/".join(parts[:i]), p) for i in range(1, len(parts)))
    return f.count("/") == p.count("/") and fnmatch.fnmatch(f, p)      # anchored file pattern


def clear(folder, keep):
    """Remove everything in folder except the entries named in keep."""
    for e in os.listdir(folder):
        if e in keep:
            continue
        p = os.path.join(folder, e)
        shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)


def main():
    global OUT
    if os.path.isdir(os.path.join(OUT, ".git")):
        raise SystemExit(f"{OUT} is a git repository; copy the new files in by hand or remove it first")
    gits = [d for d, ds, _ in os.walk(OUT) if ".git" in ds and d != CLONE] if os.path.exists(OUT) else []
    if gits:
        raise SystemExit(f"git repositories under {OUT} would be deleted: {gits}")
    zpath = B.OUT["code"]
    if os.path.isdir(os.path.join(CLONE, ".git")):            # write into the author's clone, keeping its .git
        clear(OUT, {os.path.basename(CLONE)})
        clear(CLONE, {".git"})
        OUT = CLONE
    else:
        if os.path.exists(OUT):
            shutil.rmtree(OUT)
        os.makedirs(OUT)
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        for n in names:
            assert n.startswith("QW2/") and ".." not in n, n
            rel = n[len("QW2/"):]
            if rel == "README_REVIEW.md":
                continue
            dst = os.path.join(OUT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(z.read(n))
    N = build_manuscript_pdf.numbering()
    extra = {"README.md": readme(N), "LICENSE": licence(), "CITATION.cff": citation(), ".gitignore": "\n".join(IGNORE) + "\n"}
    for rel, text in extra.items():
        with open(os.path.join(OUT, rel), "w", encoding="utf-8") as f:
            f.write(text)
    # checks: nothing the .gitignore would hide, nothing near GitHub's limits, no local paths or account names
    files = sorted(os.path.relpath(os.path.join(d, f), OUT) for d, _, fs in os.walk(OUT) for f in fs
                   if not os.path.relpath(d, OUT).split(os.sep)[0] == ".git")
    pats = [p for p in IGNORE if p and not p.startswith("#")]
    hidden = [f for f in files if any(ignored(f, p) for p in pats)]
    big = [f for f in files if os.path.getsize(os.path.join(OUT, f)) > 50 * 2 ** 20]
    local = []
    for f in files:
        if f.endswith((".py", ".md", ".txt", ".toml", ".cff", ".json", ".js", ".html", ".css", ".sh", ".R")):
            s = open(os.path.join(OUT, f), encoding="utf-8", errors="ignore").read()
            # A local username can be part of an intentional public URL. Exempt only these exact destinations;
            # continue checking the rest of the text for local paths and account traces.
            for url in PUBLIC_URLS:
                s = s.replace(url, "<public URL>")
            if PERSONAL.search(s) or (f not in ("paper/build_github_release.py", "paper/build_code_bundle.py")   # their own scan rules
                                      and GENERIC.search(s)):
                local.append(f)
    if hidden or big or local:
        raise SystemExit(f"check failed: ignored {hidden[:5]}, large {big[:5]}, local paths {local[:5]}")
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in files) / 2 ** 20
    print(f"{len(files)} files, {size:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
