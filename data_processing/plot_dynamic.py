#!/usr/bin/env python3
"""Verify and visualize a PureMagic schedule of a dynamic circuit.

    python data_processing/plot_dynamic.py CIRCUIT.trans [--schedule CIRCUIT.schedule]
        [--sidecar CIRCUIT.trans.dyn.json] [--reaction-latency L] [--window A:B] [-o PREFIX]

Two parts:

1. CHECKS (independent of the Rust scheduler: they re-derive the constraints from the .trans file
   and read only the .schedule output). Exit status 1 if any fails.
     * every product is scheduled once (CX: two consecutive lcycles; S/SX: 1-4 consecutive, they
       take 3 normally, fewer or more when a T-failure correction is merged)
     * qubit order: a product starts after the previous product on each shared qubit finished
     * classical order: a product with `@k` starts at least L+1 lcycles after product k finished
     * no two products use the same qubit in one lcycle
     * sidecar: every conditional product's `needs` are among its `@` deps; frame_deps and needs only
       name earlier measurements; every `needs` target is a measurement
2. PLOTS (matplotlib), written as PREFIX.timeline.png and PREFIX.overview.png:
     * timeline: one row per qubit, one bar per product (colour = kind), measurement markers, and an
       arrow from each measurement to the conditional products that wait on it (the reaction gap
       is visible as the arrow's horizontal extent)
     * overview: per-qubit activity over the whole run, plus cumulative T and measurement count.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TRANS = re.compile(r"^([+-])([XYZ_]+)<(\w+)>(?:@([\d,]+))?\s*$")
SCHED = re.compile(r"^\s*(\d+):\s+(\S+)((?:\s+\d+<\w+>)*)\s*$")
ITEM = re.compile(r"(\d+)<(\w+)>")


def read_trans(path):
    prods = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = TRANS.match(line)
            if not m:
                raise SystemExit(f"{path}: cannot parse line {ln + 1}: {line!r}")
            sign, ops, kind, after = m.groups()
            prods.append(
                dict(
                    sign=sign,
                    qubits=[(q, c) for q, c in enumerate(ops) if c != "_"],
                    kind=kind,
                    after=[int(x) for x in after.split(",")] if after else [],
                )
            )
    return prods


def read_schedule(path):
    """Return {product id: [lcycles]} (a CX appears in two lcycles)."""
    when = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = ANSI.sub("", line.rstrip("\n"))
            if not line or line.startswith("#"):
                continue
            m = SCHED.match(line)
            if not m:
                raise SystemExit(f"{path}: cannot parse schedule line {line!r}")
            lc = int(m.group(1))
            for pid, _kind in ITEM.findall(m.group(3)):
                when[int(pid)].append(lc)
    return when


def latency_from_header(path):
    with open(path) as f:
        for line in f:
            m = re.search(r"reaction_latency: (\d+)", line)
            if m:
                return int(m.group(1))
            if not line.startswith("#"):
                break
    return 0


def check(prods, when, side, L):
    errs = []

    def err(msg):
        if len(errs) < 40:
            errs.append(msg)

    n = len(prods)
    start, end = {}, {}
    for i, p in enumerate(prods):
        ls = sorted(when.get(i, []))
        kind = p["kind"]
        lo, hi = {"CX": (2, 2), "S": (1, 4), "SX": (1, 4), "Sdg": (1, 4), "SXdg": (1, 4)}.get(kind, (1, 1))
        if not lo <= len(ls) <= hi:
            err(f"product {i} <{kind}> scheduled {len(ls)} time(s), expected {lo}-{hi}")
            continue
        if ls != list(range(ls[0], ls[0] + len(ls))):
            err(f"product {i} <{kind}> lcycles {ls} are not consecutive")
        start[i], end[i] = ls[0], ls[-1]
    for i in when:
        if i >= n:
            err(f"schedule mentions product {i}, but the trans file has only {n}")

    # qubit order + classical order
    last = {}
    qdep = defaultdict(set)
    for i, p in enumerate(prods):
        for q, _ in p["qubits"]:
            if q in last:
                qdep[i].add(last[q])
            last[q] = i
    for i, p in enumerate(prods):
        if i not in start:
            continue
        for j in qdep[i]:
            if j in end and start[i] <= end[j]:
                err(f"qubit order: {i} starts at {start[i]} but {j} (same qubit) ends at {end[j]}")
        for j in p["after"]:
            if j not in end:
                continue
            need = 1 if j in qdep[i] and L == 0 else L + 1
            if start[i] < end[j] + need:
                err(
                    f"classical order: {i} starts at {start[i]}, needs outcome of {j} "
                    f"(ends {end[j]}) + latency {L}"
                )

    # resource: one product per qubit per lcycle
    busy = {}
    for i, p in enumerate(prods):
        for lc in when.get(i, []):
            for q, _ in p["qubits"]:
                if (lc, q) in busy and busy[(lc, q)] != i:
                    err(f"lcycle {lc}: qubit {q} used by both {busy[(lc, q)]} and {i}")
                busy[(lc, q)] = i

    # sidecar consistency
    if side:
        # `needs` / `frame_deps` hold measurement ids (meas_id), not product indices.
        recs = {r["i"]: r for r in side["products"]}
        m2p = {r["meas_id"]: i for i, r in recs.items() if r.get("meas_id") is not None}
        derived = {d["id"]: [t[0] for t in d["terms"]] for d in side.get("derived", [])}

        def expand(mid):  # product indices of the real measurements behind an outcome id
            return [m2p.get(x) for x in derived.get(mid, [mid])]

        for i, r in recs.items():
            for mid in r.get("needs", []):
                j = m2p.get(mid)
                if j is None:
                    err(f"sidecar: product {i} needs outcome {mid}, which no measurement produces")
                    continue
                if j >= i:
                    err(f"sidecar: product {i} needs later product {j}")
                if j not in prods[i]["after"]:
                    err(f"sidecar: product {i} needs outcome {mid} (product {j}) but has no @{j}")
            for mid in r.get("frame_deps", []):
                for k in expand(mid):
                    if k is None or k >= i:
                        err(f"sidecar: frame_dep {mid} of product {i} is not an earlier measurement")
        for ev in side.get("events", []):
            for mid in ev.get("x_deps", []) + ev.get("z_deps", []):
                for k in expand(mid):
                    if k is None or k >= ev["before"]:
                        err(f"sidecar: {ev['op']} before {ev['before']} uses outcome {mid} "
                            f"that is not yet measured")
    return errs, start, end, qdep


KIND_STYLE = {  # name -> (label, colour, hatch)
    "T": ("T", "#c8782b", None),
    "M": ("measure", "#3f3f46", None),
    "R": ("reset/init", "#7a5ea8", None),
    "CX": ("CX", "#2f7d7a", None),
    "S": ("S/SX/Z/X", "#9aa7b3", None),
}


def style(kind):
    if kind in KIND_STYLE:
        return KIND_STYLE[kind]
    return KIND_STYLE["S"]


def plot(prods, start, end, qdep, side, L, prefix, window, nq):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    cond = {}
    if side:
        for r in side["products"]:
            if r.get("needs"):
                m2p = {q["meas_id"]: q["i"] for q in side["products"]
                       if q.get("meas_id") is not None}
                cond[r["i"]] = [m2p[m] for m in r["needs"] if m in m2p]
    ids = [i for i in start if window is None or (end[i] >= window[0] and start[i] <= window[1])]
    if window is None and len(ids) > 3000:
        # too much for a bar chart: use the first 3000 lcycles' worth and say so in the title
        cut = sorted(start[i] for i in ids)[2999]
        window = (0, cut)
        ids = [i for i in ids if start[i] <= cut]
    fig, ax = plt.subplots(figsize=(14, 0.55 * nq + 2.2))
    for i in ids:
        lab, col, _ = style(prods[i]["kind"])
        qs = [q for q, _ in prods[i]["qubits"]]
        is_cond = i in cond
        for q, b in prods[i]["qubits"]:
            ax.broken_barh(
                [(start[i] - 0.5, end[i] - start[i] + 1)],
                (q - 0.34, 0.68),
                facecolors=col,
                edgecolors="#b3261e" if is_cond else "white",
                linewidth=1.4 if is_cond else 0.5,
            )
        if len(qs) > 1:
            ax.plot([start[i]] * 2, [min(qs), max(qs)], color=col, lw=1.2, zorder=1)
        if prods[i]["kind"] == "M":
            q = qs[0]
            ax.plot(end[i] + 0.0, q, marker="D", ms=5, color="#111", zorder=5)
    drawn = set(ids)
    for i, needs in cond.items():
        if i not in drawn:
            continue
        for j in needs:
            if j in drawn and j in end:
                qj = prods[j]["qubits"][0][0]
                qi = prods[i]["qubits"][0][0]
                ax.annotate(
                    "",
                    xy=(start[i] - 0.5, qi),
                    xytext=(end[j] + 0.5, qj),
                    arrowprops=dict(arrowstyle="->", color="#b3261e", lw=0.8, alpha=0.7,
                                    shrinkA=0, shrinkB=0),
                    zorder=6,
                )
    ax.set_yticks(range(nq))
    ax.set_yticklabels([f"q{q}" for q in range(nq)])
    ax.invert_yaxis()
    ax.set_xlabel("logical cycle")
    span = "" if window is None else f"  (lcycles {window[0]}-{window[1]})"
    ax.set_title(f"Schedule timeline{span}; reaction latency {L}")
    ax.grid(axis="x", color="#ddd", lw=0.5)
    ax.set_axisbelow(True)
    h = [Patch(facecolor=c, label=l) for l, c, _ in KIND_STYLE.values()]
    h.append(Patch(facecolor="white", edgecolor="#b3261e", label="conditional (waits on outcome)"))
    h.append(Line2D([0], [0], color="#b3261e", lw=1, label="outcome -> dependent"))
    ax.legend(handles=h, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(f"{prefix}.timeline.png", dpi=140)
    plt.close(fig)

    # overview over the whole run
    import numpy as np

    T = max(end.values()) + 1
    nb = min(200, T)
    edges = np.linspace(0, T, nb + 1)
    act = np.zeros((nq, nb))
    ncum = np.zeros(nb)
    mcum = np.zeros(nb)
    for i in start:
        for lc in range(start[i], end[i] + 1):
            b = min(int(lc * nb / T), nb - 1)
            for q, _ in prods[i]["qubits"]:
                act[q, b] += 1
        b = min(int(end[i] * nb / T), nb - 1)
        if prods[i]["kind"] == "T":
            ncum[b] += 1
        if prods[i]["kind"] == "M":
            mcum[b] += 1
    width = T / nb
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 0.3 * nq + 5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[max(nq * 0.3, 1.5), 2]))
    im = a1.imshow(act / width, aspect="auto", extent=[0, T, nq - 0.5, -0.5],
                   cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    a1.set_yticks(range(nq))
    a1.set_yticklabels([f"q{q}" for q in range(nq)])
    a1.set_title("Qubit occupancy (fraction of lcycles in use)")
    fig.colorbar(im, ax=a1, pad=0.01, fraction=0.03)
    x = (edges[:-1] + edges[1:]) / 2
    a2.plot(x, np.cumsum(ncum), color="#c8782b", lw=2, label="T products done")
    a2.plot(x, np.cumsum(mcum), color="#3f3f46", lw=2, label="measurements done")
    a2.set_yscale("symlog")
    a2.set_xlabel("logical cycle")
    a2.set_ylabel("cumulative count")
    a2.legend(frameon=False)
    a2.grid(color="#ddd", lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{prefix}.overview.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("trans")
    ap.add_argument("--schedule")
    ap.add_argument("--sidecar")
    ap.add_argument("--reaction-latency", type=int, default=None,
                    help="default: read from the schedule header")
    ap.add_argument("--window", help="A:B lcycle range for the timeline plot")
    ap.add_argument("-o", "--prefix")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    stem = a.trans.rsplit(".", 1)[0] if "." in a.trans.rsplit("/", 1)[-1] else a.trans
    sched = a.schedule or stem + ".schedule"
    sidec = a.sidecar or a.trans + ".dyn.json"
    prefix = a.prefix or stem
    prods = read_trans(a.trans)
    when = read_schedule(sched)
    try:
        side = json.load(open(sidec))
    except FileNotFoundError:
        side = None
    L = a.reaction_latency if a.reaction_latency is not None else latency_from_header(sched)
    errs, start, end, qdep = check(prods, when, side, L)
    nq = max((q for p in prods for q, _ in p["qubits"]), default=-1) + 1
    nT = sum(p["kind"] == "T" for p in prods)
    nM = sum(p["kind"] == "M" for p in prods)
    nC = len(side and [r for r in side["products"] if r.get("needs")] or [])
    print(f"{len(prods)} products ({nT} T, {nM} M, {nC} conditional), {nq} qubits, "
          f"{max(end.values(), default=0)} lcycles, latency {L}"
          f"{'' if side else ', no sidecar found'}")
    if errs:
        print(f"FAIL: {len(errs)} violation(s) (first 40 shown)")
        for e in errs:
            print("  -", e)
    else:
        print("PASS: all schedule invariants hold")
    if not a.no_plot:
        w = tuple(int(x) for x in a.window.split(":")) if a.window else None
        plot(prods, start, end, qdep, side, L, prefix, w, nq)
        print(f"wrote {prefix}.timeline.png, {prefix}.overview.png")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
