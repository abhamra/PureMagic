#!/usr/bin/env python3
"""Lower a dcsynth dynamic circuit to the Clifford+T QASM that `transpile` reads.

dcsynth (the dynamic-circuit ancilla-reuse compiler) assembles a Qiskit `QuantumCircuit` that
contains `measure`, `reset`, and `if_else` blocks whose bodies are arbitrary single-qubit
`UnitaryGate`s. PureMagic's `transpile` wants a Clifford+T circuit in OpenQASM 2 where:

  * every classical bit is its own 1-bit register (`creg m0[1];`), so a condition names one bit;
  * `measure q[i] -> m3[0];` and `reset q[i];` appear as written;
  * a gate that depends on measurement outcomes is spelled `if(m3==1 && m5==0) t q[2];`, one
    single-qubit gate per line, from h, s, sdg, sx, sxdg, x, y, z, t, tdg.

This script does that lowering. The steps:

  1. Consecutive single-qubit gates on a qubit are merged into one 2x2 matrix before synthesis
     (fewer T gates), and flushed at every CX, measurement, reset or conditional.
  2. A 2x2 unitary becomes Clifford+T by: identity -> nothing; a Pauli -> one x/y/z gate (those
     are free Pauli-frame updates downstream); another Clifford -> a shortest h/s word; anything
     else -> ZYZ Euler angles with each Rz synthesised by gridsynth (`pygridsynth`) to `--epsilon`.
  3. An `if_else` is flattened to leaves "under these bit values, apply this unitary to qubit q".
     For each qubit the leaves are completed to all 2^k assignments of the k bits involved. Then
     the *cheapest* assignment's unitary U_ref is applied unconditionally and only the difference
     V_a = U_a U_ref^dagger is conditioned on the bits. dcsynth's corrections usually differ by a
     Pauli (or at worst a Clifford), so most of them become free frame updates instead of
     conditional T gates that the scheduler must wait on.

Anything `transpile` cannot express (a conditional CX, a measurement inside an `if_else`, a
classical expression as a condition) raises `NotImplementedError` rather than being dropped.

Usage:
  dcsynth_to_cliffordt.py circuit.qpy -o circuit.cliffordt.qasm [--epsilon 1e-8]
  # circuit.qpy: qiskit.qpy.dump(qc, open("circuit.qpy", "wb")) on the circuit dcsynth built
  # (also accepted: OpenQASM 3 `.qasm3`, loaded with qiskit_qasm3_import)

Python:
  from dcsynth_to_cliffordt import lower
  text, stats = lower(qc, epsilon=1e-8)
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import warnings
from collections import ChainMap, Counter
from pathlib import Path

import numpy as np

SQ2 = 1 / math.sqrt(2)
G = {
    "h": SQ2 * np.array([[1, 1], [1, -1]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "t": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
    "tdg": np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex),
}
I2 = np.eye(2, dtype=complex)
# Rz(k*pi/4) up to global phase, as gates in time order (T^k).
EXACT_RZ = {0: [], 1: ["t"], 2: ["s"], 3: ["s", "t"], 4: ["z"], 5: ["z", "t"], 6: ["sdg"], 7: ["tdg"]}
MAX_COND_BITS = 8  # beyond this a single if_else is emitted leaf by leaf instead of completed


# ----------------------------------------------------------------------------- matrices
def seq_matrix(seq: list[str]) -> np.ndarray:
    """The matrix of gates applied in time order (first gate first)."""
    m = I2
    for g in seq:
        m = G[g] @ m
    return m


def phase_distance(a: np.ndarray, b: np.ndarray) -> float:
    """min over global phase of the operator-norm distance between two 2x2 unitaries."""
    alpha = np.angle(np.vdot(b, a))
    return float(np.linalg.norm(a - np.exp(1j * alpha) * b, 2))


def _canon(u: np.ndarray, digits: int = 6) -> tuple:
    """Hashable form of a 2x2 matrix with its global phase removed."""
    flat = u.reshape(-1)
    k = int(np.argmax(np.abs(flat) > 1e-6))
    v = (u / (flat[k] / abs(flat[k]))).reshape(-1)
    return tuple(np.round(v, digits).tolist())


def _clifford_table() -> dict[tuple, list[str]]:
    """All 24 single-qubit Cliffords (mod phase) -> shortest word over h, s (time order)."""
    table: dict[tuple, list[str]] = {_canon(I2): []}
    frontier = [(I2, [])]
    while frontier:
        nxt = []
        for u, w in frontier:
            for g in ("h", "s"):
                v = G[g] @ u
                key = _canon(v)
                if key not in table:
                    table[key] = w + [g]
                    nxt.append((v, w + [g]))
        frontier = nxt
    assert len(table) == 24
    return table


CLIFFORDS = _clifford_table()


def _compress_s_runs(word: list[str]) -> list[str]:
    """s s -> z, s s s -> sdg (runs of s taken mod 4)."""
    out: list[str] = []
    run = 0

    def flush() -> None:
        nonlocal run
        out.extend([[], ["s"], ["z"], ["sdg"]][run % 4])
        run = 0

    for g in word:
        if g == "s":
            run += 1
        else:
            flush()
            out.append(g)
    flush()
    return out


def pauli_name(u: np.ndarray, tol: float) -> str | None:
    for name in ("x", "y", "z"):
        if phase_distance(u, G[name]) < tol:
            return name
    return None


def nearest_clifford(u: np.ndarray, tol: float) -> list[str] | None:
    """The gate word of the Clifford within `tol` of `u` (mod phase), if there is one."""
    for key, word in CLIFFORDS.items():
        if phase_distance(u, seq_matrix(word)) < tol:
            return _compress_s_runs(word)
    return None


# ----------------------------------------------------------------------------- synthesis
_GS_REVERSED: bool | None = None  # does pygridsynth's string read last-gate-first?


def _gridsynth_letters(theta: float, eps: float) -> str:
    import mpmath
    from pygridsynth.gridsynth import gridsynth_gates

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gridsynth_gates(theta=mpmath.mpf(theta), epsilon=mpmath.mpf(eps))


def _letters_to_gates(s: str, reverse: bool) -> list[str]:
    names = {"H": "h", "S": "s", "T": "t", "X": "x", "Y": "y", "Z": "z"}
    letters = [c for c in s if c in names]  # W (global phase) and I are dropped
    if reverse:
        letters = letters[::-1]
    return _compress_s_runs_keep_others(letters, names)


def _compress_s_runs_keep_others(letters: list[str], names: dict[str, str]) -> list[str]:
    return _compress_s_runs([names[c] for c in letters])


def rz_matrix(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]])


def _calibrate_gridsynth() -> bool:
    """Work out, once, whether the gridsynth string lists the last-applied gate first."""
    theta, eps = 0.3, 1e-3
    s = _gridsynth_letters(theta, eps)
    best = []
    for reverse in (False, True):
        m = seq_matrix(_letters_to_gates(s, reverse))
        best.append((phase_distance(m, rz_matrix(theta)), reverse))
    d, reverse = min(best)
    if d > 10 * eps:
        raise RuntimeError(f"cannot interpret pygridsynth output (distance {d:.2e})")
    return reverse


def synth_rz(theta: float, eps: float) -> list[str]:
    """Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2}) as Clifford+T gates (time order)."""
    global _GS_REVERSED
    k = round(theta / (math.pi / 4))
    if abs(theta - k * math.pi / 4) < 1e-9:
        return list(EXACT_RZ[k % 8])
    if _GS_REVERSED is None:
        _GS_REVERSED = _calibrate_gridsynth()
    return _letters_to_gates(_gridsynth_letters(theta, eps), _GS_REVERSED)


def synth_1q(u: np.ndarray, eps: float, cliff_tol: float = 1e-7) -> list[str]:
    """Clifford+T gates (time order) equal to `u` up to a global phase.

    Exact for identity, Paulis and Cliffords (within `cliff_tol`); otherwise a ZYZ decomposition
    with every Rz synthesised to `eps`, so the whole sequence is within about 3*eps of `u`.
    """
    if phase_distance(u, I2) < cliff_tol:
        return []
    p = pauli_name(u, cliff_tol)
    if p:
        return [p]
    word = nearest_clifford(u, cliff_tol)
    if word is not None:
        return word
    from qiskit.synthesis import OneQubitEulerDecomposer

    theta, phi, lam = OneQubitEulerDecomposer("ZYZ").angles(u)  # u ~ Rz(phi) Ry(theta) Rz(lam)
    out = synth_rz(lam, eps)
    if abs(theta) > 1e-12:
        # Ry(b) = S H Rz(b) H Sdg as a matrix product, so in time order: sdg, h, rz, h, s.
        out += ["sdg", "h"] + synth_rz(theta, eps) + ["h", "s"]
    out += synth_rz(phi, eps)
    out = _compress_s_runs(out)
    err = phase_distance(seq_matrix(out), u)
    if err > max(8 * eps, 1e-9):
        raise RuntimeError(f"synthesised sequence is {err:.2e} from the target (eps={eps:.1e})")
    return out


def t_count(seq: list[str]) -> int:
    return sum(g in ("t", "tdg") for g in seq)


# ----------------------------------------------------------------------------- lowering
class Lowerer:
    def __init__(self, qc, eps: float, cliff_tol: float) -> None:
        self.qc, self.eps, self.cliff_tol = qc, eps, cliff_tol
        self.lines: list[str] = []
        self.pending: dict[int, np.ndarray] = {}
        self.stats: Counter = Counter()
        self._cache: dict[tuple, list[str]] = {}

    # -- emission ----------------------------------------------------------------------
    def synth(self, u: np.ndarray) -> list[str]:
        key = _canon(u, 9)
        if key not in self._cache:
            self._cache[key] = synth_1q(u, self.eps, self.cliff_tol)
        return self._cache[key]

    def emit(self, gate: str, q: int, conds: tuple = ()) -> None:
        if conds:
            terms = " && ".join(f"m{b}=={int(v)}" for b, v in conds)
            self.lines.append(f"if({terms}) {gate} q[{q}];")
            self.stats["conditional_gates"] += 1
            self.stats["conditional_t"] += gate in ("t", "tdg")
        else:
            self.lines.append(f"{gate} q[{q}];")
            self.stats["gates"] += 1
            self.stats["t"] += gate in ("t", "tdg")

    def add_1q(self, u: np.ndarray, q: int) -> None:
        self.pending[q] = u @ self.pending.get(q, I2)

    def flush(self, q: int) -> None:
        u = self.pending.pop(q, None)
        if u is not None:
            for g in self.synth(u):
                self.emit(g, q)

    # -- classical conditions -----------------------------------------------------------
    @staticmethod
    def _terms(condition, cmap) -> list[tuple[int, bool]]:
        try:
            target, value = condition
        except TypeError:
            raise NotImplementedError(f"unsupported classical expression condition: {condition}")
        if hasattr(target, "__iter__"):  # a ClassicalRegister
            bits = list(target)
            return [(cmap[b], bool((int(value) >> i) & 1)) for i, b in enumerate(bits)]
        return [(cmap[target], bool(value))]

    def _flatten(self, circ, qmap, cmap, conds) -> list[tuple[tuple, int, np.ndarray]]:
        """Leaves `(conds, qubit, 2x2 unitary)` of a (nested) if_else body, in program order."""
        from qiskit.circuit import IfElseOp
        from qiskit.quantum_info import Operator

        leaves = []
        for inst in circ.data:
            op = inst.operation
            if op.name in ("barrier", "delay"):
                continue
            if isinstance(op, IfElseOp):
                sub_q = ChainMap(dict(zip(op.blocks[0].qubits, (qmap[b] for b in inst.qubits))), qmap)
                sub_c = ChainMap(
                    dict(zip(op.blocks[0].clbits, (cmap[b] for b in inst.clbits))), cmap
                )
                terms = self._terms(op.condition, sub_c)
                leaves += self._flatten(op.blocks[0], sub_q, sub_c, conds + tuple(terms))
                if len(op.blocks) > 1 and len(op.blocks[1].data):
                    if len(terms) != 1:
                        raise NotImplementedError("else-branch of a multi-bit condition")
                    (b, v), = terms
                    sub_q1 = ChainMap(
                        dict(zip(op.blocks[1].qubits, (qmap[x] for x in inst.qubits))), qmap
                    )
                    sub_c1 = ChainMap(
                        dict(zip(op.blocks[1].clbits, (cmap[x] for x in inst.clbits))), cmap
                    )
                    leaves += self._flatten(op.blocks[1], sub_q1, sub_c1, conds + ((b, not v),))
            elif op.num_qubits == 1 and not op.num_clbits:
                leaves.append((conds, qmap[inst.qubits[0]], Operator(op).data))
            else:
                raise NotImplementedError(
                    f"`{op.name}` inside an if_else cannot be expressed: PureMagic supports only "
                    "single-qubit conditional gates (use the default per-qubit dcsynth layout)"
                )
        return leaves

    @staticmethod
    def _cost(v: np.ndarray, tol: float) -> float:
        if phase_distance(v, I2) < tol:
            return 0.0
        if pauli_name(v, tol):
            return 0.1  # a classical frame update
        if nearest_clifford(v, tol) is not None:
            return 1.0
        return 10.0  # a conditional T-type rotation: the expensive case

    def _emit_if(self, inst, qmap, cmap) -> None:
        op = inst.operation
        sub_q = ChainMap(dict(zip(op.blocks[0].qubits, (qmap[b] for b in inst.qubits))), qmap)
        sub_c = ChainMap(dict(zip(op.blocks[0].clbits, (cmap[b] for b in inst.clbits))), cmap)
        terms = self._terms(op.condition, sub_c)
        leaves = self._flatten(op.blocks[0], sub_q, sub_c, tuple(terms))
        if len(op.blocks) > 1 and len(op.blocks[1].data):
            if len(terms) != 1:
                raise NotImplementedError("else-branch of a multi-bit condition")
            (b, v), = terms
            sub_q1 = ChainMap(dict(zip(op.blocks[1].qubits, (qmap[x] for x in inst.qubits))), qmap)
            sub_c1 = ChainMap(dict(zip(op.blocks[1].clbits, (cmap[x] for x in inst.clbits))), cmap)
            leaves += self._flatten(op.blocks[1], sub_q1, sub_c1, ((b, not v),))
        if not leaves:
            return
        bits = sorted({b for conds, _, _ in leaves for b, _ in conds})
        qubits = sorted({q for _, q, _ in leaves})
        for q in qubits:
            self.flush(q)
        self.stats["if_else"] += 1

        if len(bits) > MAX_COND_BITS:  # too many assignments to complete; emit leaf by leaf
            for conds, q, u in leaves:
                for g in self.synth(u):
                    self.emit(g, q, tuple(sorted(set(conds))))
            return

        assignments = list(itertools.product((False, True), repeat=len(bits)))
        for q in qubits:
            # U_a: the unitary qubit q gets under assignment a (leaves in program order).
            per_a = []
            for a in assignments:
                val = dict(zip(bits, a))
                u = I2
                for conds, lq, lu in leaves:
                    if lq == q and all(val[b] == v for b, v in conds):
                        u = lu @ u
                per_a.append(u)
            best = None
            for r, u_ref in enumerate(per_a):
                vs = [u @ u_ref.conj().T for u in per_a]
                cost = sum(self._cost(v, self.cliff_tol) for v in vs)
                if best is None or cost < best[0] - 1e-12:
                    best = (cost, r, vs)
            _, r, vs = best
            self.add_1q(per_a[r], q)
            self.flush(q)  # the reference runs now, unconditionally; only V_a waits for outcomes
            for a, v in zip(assignments, vs):
                if phase_distance(v, I2) < self.cliff_tol:
                    continue
                conds = tuple(zip(bits, a))
                for g in self.synth(v):
                    self.emit(g, q, conds)

    # -- main walk ---------------------------------------------------------------------
    def run(self) -> str:
        from qiskit.circuit import IfElseOp
        from qiskit.quantum_info import Operator

        qc = self.qc
        qmap = {b: i for i, b in enumerate(qc.qubits)}
        cmap = {b: i for i, b in enumerate(qc.clbits)}
        for inst in qc.data:
            op = inst.operation
            qs = [qmap[b] for b in inst.qubits]
            if op.name in ("barrier", "delay", "id"):
                continue
            if op.name == "measure":
                self.flush(qs[0])
                c = cmap[inst.clbits[0]]
                self.lines.append(f"measure q[{qs[0]}] -> m{c}[0];")
                self.stats["measure"] += 1
            elif op.name == "reset":
                self.flush(qs[0])
                self.lines.append(f"reset q[{qs[0]}];")
                self.stats["reset"] += 1
            elif isinstance(op, IfElseOp):
                self._emit_if(inst, qmap, cmap)
            elif op.num_qubits == 1 and not op.num_clbits:
                self.add_1q(Operator(op).data, qs[0])
            elif op.name == "cx":
                self.flush(qs[0]), self.flush(qs[1])
                self.lines.append(f"cx q[{qs[0]}], q[{qs[1]}];")
                self.stats["cx"] += 1
            elif op.name == "cz":
                a, b = qs
                self.flush(a), self.flush(b)
                self.emit("h", b)
                self.lines.append(f"cx q[{a}], q[{b}];")
                self.emit("h", b)
                self.stats["cx"] += 1
            elif op.name == "swap":
                a, b = qs
                self.flush(a), self.flush(b)
                for c_, t_ in ((a, b), (b, a), (a, b)):
                    self.lines.append(f"cx q[{c_}], q[{t_}];")
                self.stats["cx"] += 3
            else:
                raise NotImplementedError(
                    f"`{op.name}` is not supported; decompose to cx + single-qubit gates first "
                    "(e.g. qiskit.transpile(qc, basis_gates=['cx','u3'], optimization_level=0))"
                )
        for q in sorted(self.pending):
            self.flush(q)

        head = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "// lowered from a dcsynth dynamic circuit by dcsynth_to_cliffordt.py",
            f"qreg q[{qc.num_qubits}];",
        ]
        head += [f"creg m{i}[1];" for i in range(qc.num_clbits)]
        return "\n".join(head + self.lines) + "\n"


def lower(qc, epsilon: float = 1e-8, clifford_tol: float = 1e-7) -> tuple[str, Counter]:
    """Lower a dcsynth circuit; returns (OpenQASM 2 text, statistics)."""
    lw = Lowerer(qc, epsilon, clifford_tol)
    text = lw.run()
    return text, lw.stats


def load_circuit(path: Path):
    if path.suffix == ".qpy":
        from qiskit import qpy

        with open(path, "rb") as f:
            circuits = qpy.load(f)
        return circuits[0]
    if path.suffix in (".qasm3", ".qasm"):
        from qiskit import qasm3

        return qasm3.loads(path.read_text())
    raise SystemExit(f"unrecognised input `{path}`: expected .qpy or .qasm3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="default: <input stem>.cliffordt.qasm")
    ap.add_argument("--epsilon", type=float, default=1e-8, help="gridsynth precision per Rz")
    ap.add_argument(
        "--clifford-tol", type=float, default=1e-7,
        help="a 2x2 unitary this close to a Clifford is snapped to it exactly",
    )
    args = ap.parse_args()
    qc = load_circuit(args.input)
    text, st = lower(qc, args.epsilon, args.clifford_tol)
    out = args.output or args.input.with_suffix(".cliffordt.qasm")
    out.write_text(text)
    print(
        f"wrote {out}: {qc.num_qubits} qubits, {qc.num_clbits} classical bits, "
        f"{st['measure']} measure, {st['reset']} reset, {st['if_else']} if_else, "
        f"{st['cx']} cx, T count {st['t']} unconditional + {st['conditional_t']} conditional "
        f"({st['conditional_gates']} conditional gates in all)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
