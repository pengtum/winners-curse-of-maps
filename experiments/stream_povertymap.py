"""Stream WILDS PovertyMap v1.1 (13.1 GB tar) from stdin and keep only cluster-level data.

    curl -sS -k -L <codalab url> | ~/miniforge3/envs/favela/bin/python -I experiments/stream_povertymap.py ~/data/wilds

The archive is never written to disk (the disk has ~15 GiB free). Every images/*.npz member is
read in memory with np.load(allow_pickle=False) and reduced to per-band summary features; any
.csv member (dhs_metadata.csv) is saved as is. Output in <out_dir>:
  poverty_features.csv   idx + 8 bands x (mean, sd, p10, p25, p50, p75, p90, |grad| mean,
                         centre-minus-whole mean) = 72 features
  dhs_metadata.csv       WILDS metadata (lat/lon noised for anonymity, country, urban, wealthpooled)
The download used curl -k because CodaLab's TLS certificate had expired (user-approved,
2026-10-21); the result is sanity-checked against the WILDS paper (19,669 clusters).
"""
import io, os, re, sys, tarfile, time
import numpy as np

out = os.path.expanduser(sys.argv[1]); os.makedirs(out, exist_ok=True)
fpath = os.path.join(out, "poverty_features.csv")
STATS = ["mean", "sd", "p10", "p25", "p50", "p75", "p90", "grad", "centre"]


def feats(a):
    a = np.asarray(a, np.float32)
    ax = [i for i, s in enumerate(a.shape) if s == 8]
    a = np.moveaxis(a, ax[0], 0) if ax else a          # (8, H, W)
    flat = a.reshape(a.shape[0], -1)
    q = np.percentile(flat, [10, 25, 50, 75, 90], axis=1)
    gy, gx = np.gradient(a, axis=(1, 2))
    H, W = a.shape[1:]; c = a[:, H // 2 - 32:H // 2 + 32, W // 2 - 32:W // 2 + 32]
    cols = [flat.mean(1), flat.std(1), *q, np.sqrt(gx ** 2 + gy ** 2).reshape(a.shape[0], -1).mean(1),
            c.reshape(a.shape[0], -1).mean(1) - flat.mean(1)]
    return np.stack(cols, 1).ravel()                    # band-major


t0 = time.time(); n = 0; buf = []; shape = None
header_written = os.path.exists(fpath)
with open(fpath, "a") as fo:
    tar = tarfile.open(fileobj=sys.stdin.buffer, mode="r|*")
    for m in tar:
        if not m.isfile():
            continue
        if m.name.endswith(".npz"):
            idx = int(re.search(r"(\d+)\.npz$", m.name).group(1))
            z = np.load(io.BytesIO(tar.extractfile(m).read()), allow_pickle=False)
            arr = z[z.files[0]]
            if shape is None:
                shape = arr.shape; print("image shape", shape, z.files, flush=True)
            if not header_written:
                fo.write("idx," + ",".join(f"b{b}_{s}" for b in range(8) for s in STATS) + "\n")
                header_written = True
            buf.append(f"{idx}," + ",".join(f"{v:.6g}" for v in feats(arr)))
            n += 1
            if len(buf) >= 500:
                fo.write("\n".join(buf) + "\n"); fo.flush(); buf = []
                print(f"{n} images, {time.time()-t0:.0f}s", flush=True)
        elif m.name.endswith(".csv"):
            name = os.path.basename(m.name)
            open(os.path.join(out, name), "wb").write(tar.extractfile(m).read())
            print("saved", name, m.size, flush=True)
        else:
            print("skip", m.name, m.size, flush=True)
    if buf:
        fo.write("\n".join(buf) + "\n")
print(f"done: {n} images in {time.time()-t0:.0f}s", flush=True)
