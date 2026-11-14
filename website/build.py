"""Fill website/src.html with website/data.json to give index.html (for the artifact host, which adds the page skeleton) and
standalone.html (a complete page for any static host). data.json holds aggregate results only.
    ~/miniforge3/envs/favela/bin/python website/build.py
"""
import os
H = os.path.dirname(os.path.abspath(__file__))
t = open(os.path.join(H, "src.html"), encoding="utf-8").read()
d = open(os.path.join(H, "data.json"), encoding="utf-8").read()
assert t.count("__DATA__") == 1
page = t.replace("__DATA__", d)
open(os.path.join(H, "index.html"), "w", encoding="utf-8").write(page)
head = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body style="margin:0">'
open(os.path.join(H, "standalone.html"), "w", encoding="utf-8").write(head + page + "</body></html>")
