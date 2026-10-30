#!/usr/bin/env python3
"""Check that `transpile` preserves the semantics of a dynamic Clifford+T circuit.

A dynamic circuit has mid-circuit measurement, `reset`, and classically conditioned Paulis.
`transpile` turns it into a `.trans` file (what the scheduler consumes) plus a classical sidecar
`<out>.dyn.json` (what an executor needs). This script is the oracle for that translation. It
simulates both sides exactly (statevector, branching on every measurement) and compares the joint
distribution of *all* measurement outcomes.

  reference : the input `.cliffordt.qasm`, run gate by gate.
  candidate : the `.trans` lines run in file order, with the sidecar's classical bookkeeping:

      T line   rotation exp(-i*s*pi/8*P), s = (+1 or -1 from the line's sign) * (-1)^(XOR of the
               true outcomes in frame_deps)
      Cliff    the physical Clifford named on the line
      M line   measure the unsigned Pauli P, physical outcome p;
               true outcome = p XOR [sign is '-'] XOR (XOR of true outcomes in frame_deps)
      R line   physical re-initialisation of one patch to |0> (only `--reset_mode init`)

  Pauli corrections (reset's X fix, `if(c==1) x ...`, even under `c0==1 && c1==0`) are never
  applied physically; they live in the frame, a Pauli conditioned on several bits using a
  `derived` AND outcome. If the frame logic is wrong the two distributions differ.

  Any other conditional gate (`if(c==1) h|s|t ...`) is physical. Its `.trans` lines carry a
  `cond` in the sidecar and run only if every `[outcome, value]` holds; a `fix` event just before
  it physically applies the Pauli the frame owed that qubit. The `@line,..` suffix of such a line
  must name exactly the M lines of the outcomes in its `needs`.

Usage:
  validate_dynamic.py circuit.cliffordt.qasm [--transpile PATH] [--modes frame,flush,init]
                      [--widths 0,1,2] [--defer-trailing]
  validate_dynamic.py --random N [--seed S] [--qubits 3] [--gates 14]   # fuzz

Exit status 0 iff every comparison matches (total variation distance < 1e-9).
Only numpy is required. Circuits are limited to ~10 qubits and ~10 measurements.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# The simulators recurse once per gate (and branch at every measurement). Circuits that contain
# synthesised rotations (dcsynth_to_cliffordt.py) have thousands of gates.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRANSPILE = ROOT / "target" / "release" / "transpile"

# --------------------------------------------------------------------------- gates
SQ2 = 1 / math.sqrt(2)
G1 = {
    "h": SQ2 * np.array([[1, 1], [1, -1]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex),
    "sx": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex),
    "sxdg": 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex),
}
PAULI = {"X": G1["x"], "Y": G1["y"], "Z": G1["z"]}
TRANS_1Q = {"S": "s", "Sdg": "sdg", "SX": "sx", "SXdg": "sxdg"}


class State:
    """n-qubit statevector as a rank-n tensor; axis q is qubit q."""

    def __init__(self, n: int, psi: np.ndarray | None = None):
        self.n = n
        if psi is None:
            psi = np.zeros((2,) * n, dtype=complex)
            psi[(0,) * n] = 1.0
        self.psi = psi

    def copy(self) -> "State":
        return State(self.n, self.psi.copy())

    def apply1(self, u: np.ndarray, q: int) -> None:
        self.psi = np.moveaxis(np.tensordot(u, self.psi, axes=([1], [q])), 0, q)

    def apply_cx(self, c: int, t: int) -> None:
        idx = [slice(None)] * self.n
        idx[c] = 1
        sub = self.psi[tuple(idx)]
        # after fixing axis c, the target axis shifts down by one when t > c
        ax = t - 1 if t > c else t
        self.psi[tuple(idx)] = np.flip(sub, axis=ax)

    def apply_cz(self, a: int, b: int) -> None:
        idx = [slice(None)] * self.n
        idx[a] = 1
        idx[b] = 1
        self.psi[tuple(idx)] *= -1

    def apply_pauli_string(self, ops: dict[int, str]) -> "State":
        out = self.copy()
        for q, p in ops.items():
            out.apply1(PAULI[p], q)
        return out

    def rotate(self, ops: dict[int, str], theta: float) -> None:
        """psi <- exp(-i theta P) psi = cos(theta) psi - i sin(theta) P psi."""
        p_psi = self.apply_pauli_string(ops).psi
        self.psi = math.cos(theta) * self.psi - 1j * math.sin(theta) * p_psi

    def measure_pauli(self, ops: dict[int, str]):
        """Yield (bit, probability, post-state) for the +1 (bit 0) / -1 (bit 1) outcomes of P."""
        p_psi = self.apply_pauli_string(ops).psi
        for bit, sgn in ((0, 1.0), (1, -1.0)):
            post = 0.5 * (self.psi + sgn * p_psi)
            prob = float(np.vdot(post, post).real)
            if prob > 1e-12:
                yield bit, prob, State(self.n, post / math.sqrt(prob))


def _merge(dist: dict, key: tuple, prob: float) -> None:
    dist[key] += prob


# --------------------------------------------------------------------------- reference
def parse_qasm(path: str):
    """Parse the dynamic Clifford+T QASM subset into a list of op tuples."""
    n = 0
    cregs: dict[str, tuple[int, int]] = {}
    nclbits = 0
    ops = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("//")[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "barrier")):
            continue
        if line.startswith("qreg"):
            n = int(re.search(r"\[(\d+)\]", line).group(1))
        elif line.startswith("creg"):
            m = re.match(r"creg\s+(\w+)\[(\d+)\]", line)
            cregs[m.group(1)] = (nclbits, int(m.group(2)))
            nclbits += int(m.group(2))
        elif line.startswith("measure"):
            m = re.match(r"measure\s+q\[(\d+)\](?:\s*->\s*(\w+)(?:\[(\d+)\])?)?", line)
            q = int(m.group(1))
            cb = None
            if m.group(2):
                base, size = cregs[m.group(2)]
                cb = base + (int(m.group(3)) if m.group(3) is not None else 0)
            ops.append(("measure", q, cb))
        elif line.startswith("reset"):
            ops.append(("reset", int(re.search(r"\[(\d+)\]", line).group(1))))
        elif line.startswith("if"):
            m = re.match(r"if\s*\(([^)]*)\)\s*(\w+)\s+q\[(\d+)\]", line)
            terms = []
            for term in m.group(1).split("&&"):
                reg, val = (x.strip() for x in term.split("=="))
                base, size = cregs[reg]
                assert size == 1, "validator supports 1-bit condition registers only"
                terms.append((base, int(val)))
            ops.append(("cond", terms, m.group(2), int(m.group(3))))
        else:
            m = re.match(r"(\w+)\s+(.*);", line)
            name, args = m.group(1), re.findall(r"q\[(\d+)\]", m.group(2))
            ops.append((name, *map(int, args)))
    return n, ops


def run_reference(n: int, ops) -> dict[tuple, float]:
    """Exact joint distribution of all measurement outcomes, keyed by (qubit, occurrence)."""
    ops = list(ops)
    # Mirror `transpile`: every qubit whose last op is not a measure/reset is measured at the end.
    ended = [False] * n
    written: set[int] = set()
    for op in ops:
        if op[0] in ("measure", "reset"):
            ended[op[1]] = True
            if op[0] == "measure" and op[2] is not None:
                written.add(op[2])
        elif op[0] == "cond":
            _, terms, g, q = op
            if len(terms) == 1 and g in ("x", "y", "z"):
                continue  # a single-bit conditional Pauli is a pure frame update
            # Settle the terms like `resolve_static_conditions`: a bit nothing has written reads
            # as 0, so a term demanding 1 from it never holds and one demanding 0 always does.
            keep: dict[int, int] = {}
            never = False
            for cb, v in terms:
                if cb not in written:
                    never |= bool(v)
                elif cb in keep:
                    never |= keep[cb] != v
                else:
                    keep[cb] = v
            if never:
                continue  # the gate is dropped
            if keep and g in ("x", "y", "z"):
                continue  # still a frame update (on a derived AND outcome)
            ended[q] = False  # a physical gate: unconditional, or conditional non-Pauli
        else:
            for q in op[1:]:
                ended[q] = False
    for q in range(n):
        if not ended[q]:
            ops.append(("measure", q, None))

    dist: dict[tuple, float] = defaultdict(float)

    def rec(i, st, prob, clbits, events, occ):
        if i == len(ops):
            _merge(dist, tuple(sorted(events.items())), prob)
            return
        op = ops[i]
        kind = op[0]
        if kind in ("measure", "reset"):
            q = op[1]
            for bit, p, post in st.measure_pauli({q: "Z"}):
                ev = dict(events)
                k = (q, occ.get(q, 0))
                ev[k] = bit
                oc = dict(occ)
                oc[q] = occ.get(q, 0) + 1
                cb = dict(clbits)
                nxt = post
                if kind == "measure" and op[2] is not None:
                    cb[op[2]] = bit
                if kind == "reset" and bit == 1:
                    nxt.apply1(G1["x"], q)
                rec(i + 1, nxt, prob * p, cb, ev, oc)
            return
        if kind == "cond":
            _, terms, g, q = op
            if all(clbits.get(cb, 0) == val for cb, val in terms):
                st = st.copy()
                st.apply1(G1[g], q)
            rec(i + 1, st, prob, clbits, events, occ)
            return
        st = st.copy()
        if kind == "cx":
            st.apply_cx(op[1], op[2])
        elif kind == "cz":
            st.apply_cz(op[1], op[2])
        elif kind == "swap":
            a, b = op[1], op[2]
            st.apply_cx(a, b), st.apply_cx(b, a), st.apply_cx(a, b)
        else:
            st.apply1(G1[kind], op[1])
        rec(i + 1, st, prob, clbits, events, occ)

    rec(0, State(n), 1.0, {}, {}, {})
    return dist


# --------------------------------------------------------------------------- candidate
LINE = re.compile(r"^([+-])([_XYZ]+)<(\w+)>(?:@([\d,]+))?$")


def run_trans(n: int, trans_path: str, sidecar: dict | None) -> dict[tuple, float]:
    lines = [l.strip() for l in Path(trans_path).read_text().splitlines() if l.strip()]
    recs = sidecar["products"] if sidecar else [{} for _ in lines]
    assert len(recs) == len(lines), "sidecar and .trans disagree on length"
    derived = {d["id"]: d["terms"] for d in (sidecar or {}).get("derived", [])}
    events_at: dict[int, list] = defaultdict(list)
    for e in (sidecar or {}).get("events", []):
        events_at[e["before"]].append(e)

    # The `@line,..` suffix must name exactly the M lines of the outcomes a line `needs`.
    line_of_meas = {r["meas_id"]: k for k, r in enumerate(recs) if r.get("meas_id") is not None}
    for k, text in enumerate(lines):
        suffix = LINE.match(text).group(4)
        got = sorted(int(x) for x in suffix.split(",")) if suffix else []
        want = sorted({line_of_meas[m] for m in recs[k].get("needs", [])})
        assert got == want, f"line {k} `{text}`: @-suffix {got} but sidecar needs lines {want}"
        assert all(a < k for a in got), f"line {k}: classical dependency on a later line"

    dist: dict[tuple, float] = defaultdict(float)

    def outcome(true_out, d):
        """True outcome d, or the AND of its terms if d is a derived outcome."""
        if d in derived:
            return int(all(true_out[m] == v for m, v in derived[d]))
        return true_out[d]

    def fix_events(i, st, true_out):
        for e in events_at.get(i, []):
            assert e["op"] == "fix", e
            x, z = int(e["x_const"]), int(e["z_const"])
            for d in e["x_deps"]:
                x ^= outcome(true_out, d)
            for d in e["z_deps"]:
                z ^= outcome(true_out, d)
            if x or z:
                st = st.copy()
                if x:
                    st.apply1(G1["x"], e["qubit"])
                if z:
                    st.apply1(G1["z"], e["qubit"])
        return st

    def rec(i, st, prob, true_out, events, occ):
        st = fix_events(i, st, true_out)
        if i == len(lines):
            _merge(dist, tuple(sorted(events.items())), prob)
            return
        m = LINE.match(lines[i])
        sign, body, label = m.group(1), m.group(2), m.group(3)
        r = recs[i]
        ops = {q: c for q, c in enumerate(body) if c != "_"}
        if r.get("cond") and not all(true_out[mid] == v for mid, v in r["cond"]):
            rec(i + 1, st, prob, true_out, events, occ)  # conditional line whose condition fails
            return
        flip = 0
        for d in r.get("frame_deps", []):
            flip ^= outcome(true_out, d)
        if label == "T":
            s = (1 if sign == "+" else -1) * (-1 if flip else 1)
            st = st.copy()
            st.rotate(ops, s * math.pi / 8)
            rec(i + 1, st, prob, true_out, events, occ)
        elif label == "M":
            mid, q = r["meas_id"], r["logical_qubit"]
            for p_bit, pr, post in st.measure_pauli(ops):
                t = p_bit ^ (1 if sign == "-" else 0) ^ flip
                to = dict(true_out)
                to[mid] = t
                ev = dict(events)
                k = (q, occ.get(q, 0))
                ev[k] = t
                oc = dict(occ)
                oc[q] = occ.get(q, 0) + 1
                rec(i + 1, post, prob * pr, to, ev, oc)
        elif label == "R":
            (q,) = ops
            st = st.copy()
            # The qubit was just measured in Z and is unentangled, so this is a deterministic
            # reset: flip it if it is in |1>.
            for bit, _, post in st.measure_pauli({q: "Z"}):
                if bit == 1:
                    post.apply1(G1["x"], q)
                st = post
            rec(i + 1, st, prob, true_out, events, occ)
        elif label == "CX":
            (c,) = [q for q, ch in ops.items() if ch == "Z"]
            (t,) = [q for q, ch in ops.items() if ch == "X"]
            st = st.copy()
            st.apply_cx(c, t)
            rec(i + 1, st, prob, true_out, events, occ)
        elif label in TRANS_1Q:
            (q,) = ops
            st = st.copy()
            st.apply1(G1[TRANS_1Q[label]], q)
            rec(i + 1, st, prob, true_out, events, occ)
        else:
            raise ValueError(f"unhandled .trans label {label!r}: {lines[i]}")

    rec(0, State(n), 1.0, {}, {}, {})
    return dist


def tv_distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


# --------------------------------------------------------------------------- driver
def transpile_once(binary: Path, qasm: Path, out: Path, mode: str, width: int, defer: bool,
                   auto: bool) -> subprocess.CompletedProcess:
    cmd = [str(binary), "-i", str(qasm), "-o", str(out), "--reset_mode", mode, "--sidecar"]
    if auto:
        cmd.append("--auto")
    else:
        cmd += ["-m", str(width)]
        if defer:
            cmd.append("--defer_trailing")
    return subprocess.run(cmd, capture_output=True, text=True)


def check(qasm: Path, binary: Path, modes, widths, defer_opts, tol=1e-9, verbose=True) -> bool:
    n, ops = parse_qasm(str(qasm))
    ref = run_reference(n, ops)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        configs = [(m, w, d, False) for m in modes for w in widths for d in defer_opts]
        configs += [(m, 0, True, True) for m in modes]  # --auto
        for mode, width, defer, auto in configs:
            out = Path(td) / "c.trans"
            res = transpile_once(binary, qasm, out, mode, width, defer, auto)
            tag = f"mode={mode} width={'auto' if auto else width} defer={defer or auto}"
            if res.returncode != 0:
                print(f"  FAIL {tag}: transpile exited {res.returncode}\n{res.stderr[-400:]}")
                ok = False
                continue
            side = Path(str(out) + ".dyn.json")
            sidecar = json.loads(side.read_text()) if side.exists() else None
            cand = run_trans(n, str(out), sidecar)
            d = tv_distance(ref, cand)
            if d > tol:
                ok = False
                print(f"  FAIL {tag}: TV distance {d:.3e}")
            elif verbose:
                print(f"  ok   {tag}: TV distance {d:.1e} over {len(ref)} outcome strings")
    return ok


# --------------------------------------------------------------------------- fuzzing
def random_circuit(rng: random.Random, n: int, n_gates: int) -> str:
    """Random dynamic Clifford+T circuit, one 1-bit creg per measurement (as dcsynth emits)."""
    body: list[str] = []
    bits: list[str] = []
    unwritten = "u0"
    body.append(f"creg {unwritten}[1];")  # declared, never measured into: reads as 0
    for _ in range(n_gates):
        r = rng.random()
        q = rng.randrange(n)
        if r < 0.40:
            body.append(f"{rng.choice(['h', 's', 'sdg', 't', 'tdg', 'sx'])} q[{q}];")
        elif r < 0.55 and n > 1:
            a, b = rng.sample(range(n), 2)
            body.append(f"cx q[{a}], q[{b}];")
        elif r < 0.70 and len(bits) < 6:
            name = f"m{len(bits)}"
            bits.append(name)
            body.append(f"creg {name}[1];")
            body.append(f"measure q[{q}] -> {name}[0];")
        elif r < 0.80:
            body.append(f"reset q[{q}];")
        elif bits:
            # one or two terms; a Pauli, a Clifford, or a T; sometimes on a bit nothing wrote
            pool = bits + ([unwritten] if rng.random() < 0.15 else [])
            terms = [f"{b}=={rng.randrange(2)}"
                     for b in rng.sample(pool, min(len(pool), rng.choice([1, 1, 2])))]
            g = rng.choice(["x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg"])
            body.append(f"if({' && '.join(terms)}) {g} q[{q}];")
        else:
            body.append(f"t q[{q}];")
    # creg declarations must precede use; hoist them.
    decls = [l for l in body if l.startswith("creg")]
    rest = [l for l in body if not l.startswith("creg")]
    # `if(...) y q[i];` is spelled `y`, and Paulis are spelled in lower case in QASM
    head = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    return "\n".join(head + decls + rest) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("qasm", nargs="?", help="a .cliffordt.qasm file")
    ap.add_argument("--transpile", default=str(DEFAULT_TRANSPILE), help="path to the transpile binary")
    ap.add_argument("--modes", default="frame,flush,init")
    ap.add_argument("--widths", default="0,1,2")
    ap.add_argument("--defer-trailing", action="store_true", help="only test with --defer_trailing")
    ap.add_argument("--random", type=int, default=0, help="fuzz with N random circuits")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--qubits", type=int, default=3)
    ap.add_argument("--gates", type=int, default=14)
    a = ap.parse_args()

    binary = Path(a.transpile)
    if not binary.exists():
        print(f"transpile binary not found at {binary}; run `cargo build --release` first")
        return 2
    modes = a.modes.split(",")
    widths = [int(w) for w in a.widths.split(",")]
    defer_opts = [True] if a.defer_trailing else [False, True]

    if a.random:
        rng = random.Random(a.seed)
        bad = 0
        with tempfile.TemporaryDirectory() as td:
            for k in range(a.random):
                p = Path(td) / f"fuzz{k}.cliffordt.qasm"
                p.write_text(random_circuit(rng, a.qubits, a.gates))
                if not check(p, binary, modes, widths, defer_opts, verbose=False):
                    bad += 1
                    keep = Path(os.getcwd()) / f"validate_dynamic_fail_{a.seed}_{k}.cliffordt.qasm"
                    keep.write_text(p.read_text())
                    print(f"circuit {k} FAILED; saved to {keep}")
        print(f"{a.random - bad}/{a.random} random circuits match across all configurations")
        return 1 if bad else 0

    if not a.qasm:
        ap.error("give a .cliffordt.qasm file or --random N")
    ok = check(Path(a.qasm), binary, modes, widths, defer_opts)
    print("ALL MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
