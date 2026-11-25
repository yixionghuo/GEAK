#!/usr/bin/env python3
"""Reduce per-region end stamps of a fused kernel (tools/region_stamp.h) to a region table and a
critical path.

Input directory:
  meta.json                       {"nslots": N, "ncta": C, "tick_hz": HZ, "names": [...N names]}
  stamps_rank<r>_rep<i>.bin       C*N little-endian u64, slot 0 = CTA entry, slot k = end of region k

Output per rank (median over reps of every cell):
  region | end median | end max | imbalance (max-median) | dur median | dur max | slowest CTA
  critical path: chain of (CTA, region) segments ending at the latest region end

--wait R:P declares that region R of every CTA waits for region P of *all* CTAs (a grid-wide
readiness edge: counter, last-arriver, barrier). The critical path then jumps from R to the CTA
whose P ended last, unless the CTA's own previous region (R-1) ended later than that: then R was
gated by the CTA itself and the path stays on it. Undeclared regions are assumed to depend
only on the previous region of the same CTA. Cross-rank waits cannot be followed (clocks of
different GPUs are not comparable); the path stops at them and reports the wait as a segment.

Usage:
  python3 region_timeline.py DIR [--wait wait:publish ...] [--json out.json]
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

import numpy as np


def load(dirpath):
    meta = json.load(open(os.path.join(dirpath, "meta.json")))
    n, c = meta["nslots"], meta["ncta"]
    runs = {}
    for path in sorted(glob.glob(os.path.join(dirpath, "stamps_rank*_rep*.bin"))):
        m = re.search(r"stamps_rank(\d+)_rep(\d+)\.bin$", path)
        a = np.fromfile(path, dtype="<u8")
        if a.size != n * c:
            sys.exit(f"{path}: {a.size} values, expected {n}*{c}")
        runs.setdefault(int(m.group(1)), []).append(a.reshape(c, n))
    if not runs:
        sys.exit(f"no stamps_rank*_rep*.bin in {dirpath}")
    return meta, runs


def analyse(stamps, tick_hz, waits):
    """stamps: [ncta, nslots] u64 -> dict with table rows and critical path (times in us)."""
    t0 = stamps[:, 0].min()
    end = (stamps.astype(np.int64) - np.int64(t0)) * 1e6 / tick_hz  # us since first CTA entry
    ncta, nslots = end.shape
    if (np.diff(end, axis=1) < 0).any():
        bad = np.argwhere(np.diff(end, axis=1) < 0)[0]
        raise ValueError(f"non-monotone stamps at CTA {bad[0]} slot {bad[1] + 1}: slot not written or reordered")
    dur = np.diff(end, axis=1)
    rows = []
    for r in range(nslots):
        e = end[:, r]
        row = {"slot": r, "end_median": float(np.median(e)), "end_max": float(e.max()),
               "imbalance": float(e.max() - np.median(e)), "slowest_cta": int(e.argmax())}
        if r > 0:
            row["dur_median"] = float(np.median(dur[:, r - 1]))
            row["dur_max"] = float(dur[:, r - 1].max())
        rows.append(row)
    # critical path: walk back from the latest end
    cta, r = int(end[:, nslots - 1].argmax()), nslots - 1
    path = []
    while r > 0:
        if r in waits and end[int(end[:, waits[r]].argmax()), waits[r]] > end[cta, r - 1]:
            p = waits[r]
            pc = int(end[:, p].argmax())
            path.append({"cta": cta, "slot": r, "kind": f"wait<-{p}",
                         "us": float(end[cta, r] - end[pc, p]), "gated_by_cta": pc})
            cta, r = pc, p
        else:
            path.append({"cta": cta, "slot": r, "kind": "work", "us": float(end[cta, r] - end[cta, r - 1])})
            r -= 1
    path.append({"cta": cta, "slot": 0, "kind": "entry offset", "us": float(end[cta, 0])})
    path.reverse()
    return {"span": float(end[:, nslots - 1].max()), "rows": rows, "path": path}


def median_runs(results):
    """Median over reps, cell by cell; the critical path of the rep with the median span."""
    out = {"span": statistics.median(r["span"] for r in results), "reps": len(results)}
    rows = []
    for i in range(len(results[0]["rows"])):
        cells = [r["rows"][i] for r in results]
        row = {k: statistics.median(c[k] for c in cells) if k != "slowest_cta" else cells[0][k]
               for k in cells[0]}
        row["slowest_cta"] = statistics.mode(c["slowest_cta"] for c in cells)
        rows.append(row)
    out["rows"] = rows
    mid = sorted(results, key=lambda r: r["span"])[len(results) // 2]
    out["path"] = mid["path"]
    out["path_span"] = mid["span"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--wait", action="append", default=[], help="R:P, region names or slot numbers")
    ap.add_argument("--json")
    a = ap.parse_args()
    meta, runs = load(a.dir)
    names = meta.get("names") or [f"r{i}" for i in range(meta["nslots"])]
    idx = lambda s: int(s) if s.isdigit() else names.index(s)
    waits = {}
    for w in a.wait:
        r, p = w.split(":")
        waits[idx(r)] = idx(p)
    report = {}
    for rank, reps in sorted(runs.items()):
        res = median_runs([analyse(s, meta["tick_hz"], waits) for s in reps])
        report[rank] = res
        print(f"\n== rank {rank}: span {res['span']:.2f} us (median of {res['reps']} reps), {meta['ncta']} CTAs")
        print(f"{'region':<14}{'end med':>9}{'end max':>9}{'imbal':>8}{'dur med':>9}{'dur max':>9}{'slow CTA':>9}")
        for row in res["rows"]:
            n = names[row["slot"]]
            dm = f"{row['dur_median']:9.2f}{row['dur_max']:9.2f}" if "dur_median" in row else " " * 18
            print(f"{n:<14}{row['end_median']:9.2f}{row['end_max']:9.2f}{row['imbalance']:8.2f}{dm}{row['slowest_cta']:9d}")
        print(f"critical path (rep with median span {res['path_span']:.2f} us):")
        for seg in res["path"]:
            extra = f"  gated by CTA {seg['gated_by_cta']}" if "gated_by_cta" in seg else ""
            print(f"  CTA {seg['cta']:>4}  {names[seg['slot']]:<12} {seg['kind']:<13} {seg['us']:8.2f} us{extra}")
    if len(report) > 1:
        print(f"\nrank-max span: {max(r['span'] for r in report.values()):.2f} us "
              "(per-rank clocks are not comparable; compare spans, not stamps)")
    if a.json:
        json.dump({"meta": meta, "waits": waits, "ranks": report}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
