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
     else -> ZYZ Euler angles synthesised to `--epsilon` by cyclosynth (jointly optimised U3; the
     default) or, with `--synthesizer gridsynth`, by gridsynth (Ross-Selinger, pygridsynth: one
     Rz for a diagonal target, else three at epsilon/3 each, as BQSKit's Clifford+T flow does).
  3. An `if_else` is flattened to leaves "under these bit values, apply this unitary to qubit q".
     For each qubit the leaves are completed to all 2^k assignments of the k bits involved. Then
     the *cheapest* assignment's unitary U_ref is applied unconditionally and only the difference
     V_a = U_a U_ref^dagger is conditioned on the bits. dcsynth's corrections usually differ by a
     Pauli (or at worst a Clifford), so most of them become free frame updates instead of
     conditional T gates that the scheduler must wait on. (dcsynth's ae_4 corrections are Paulis
     only to within ~1e-5; see `--clifford-tol`.)

With `--keep-cliffordt` the input is taken to be already (mostly) Clifford+T, e.g. a circuit
reassembled from blocks that BQSKit compiled: gates named h, s, sdg, t, tdg, x, y, z, sx, sxdg
are emitted exactly as written, never merged or resynthesised; only the remaining single-qubit
unitaries (dcsynth's correction branches) go through step 2.

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
#: Gates `--keep-cliffordt` emits as written (all are in `transpile`'s dialect).
KEPT_GATES = frozenset({"h", "s", "sdg", "t", "tdg", "x", "y", "z", "sx", "sxdg"})
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


def clifford_distance(u: np.ndarray) -> float:
    """Distance (phase-minimized operator norm) from `u` to the nearest single-qubit Clifford."""
    return min(phase_distance(u, seq_matrix(word)) for word in CLIFFORDS.values())


def nearest_clifford(u: np.ndarray, tol: float) -> list[str] | None:
    """The gate word of the Clifford within `tol` of `u` (mod phase), if there is one."""
    for key, word in CLIFFORDS.items():
        if phase_distance(u, seq_matrix(word)) < tol:
            return _compress_s_runs(word)
    return None


# ----------------------------------------------------------------------------- synthesis
# Rotations are synthesised by cyclosynth (the Rust extension built from the `cyclosynth/`
# submodule; `maturin build --release` there and pip-install the wheel). It returns a string over
# H S T X Z with lowercase s, t meaning the dagger.
_LETTERS = {
    "H": "h", "S": "s", "s": "sdg", "T": "t", "t": "tdg",
    "X": "x", "Y": "y", "Z": "z", "h": "h", "x": "x", "y": "y", "z": "z",
}
_IGNORED_LETTERS = set("WwI")  # global phase / identity
_CS_REVERSED: bool | None = None  # is the string in matrix order (last letter applied first)?
# "u3": jointly optimised U3 search (fewest T, ~seconds per rotation at eps 1e-8, needs eps >= 1e-10);
# "zyz": three native Rz syntheses at eps/3 each (about 2.5-3x the T, milliseconds, any eps).
SYNTH_MODE = "u3"


def _cyclosynth():
    try:
        import cyclosynth
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "cannot import cyclosynth. Build it from the cyclosynth/ submodule "
            "(cd cyclosynth && maturin build --release && pip install target/wheels/*.whl)"
        ) from e
    return cyclosynth


def _letters_to_gates(s: str, reverse: bool) -> list[str]:
    out = []
    for c in s:
        if c in _IGNORED_LETTERS:
            continue
        if c not in _LETTERS:
            raise RuntimeError(f"unexpected gate letter {c!r} in cyclosynth output {s[:40]!r}")
        out.append(_LETTERS[c])
    if reverse:
        out.reverse()
    return _compress_s_runs(out)


def _gates_of(result, what: str) -> str:
    if not result or result.gates is None:
        raise RuntimeError(f"cyclosynth found no circuit for {what}")
    return result.gates


def rz_matrix(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]])


def ry_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _calibrate_cyclosynth() -> bool:
    """Work out, once, whether cyclosynth's gate string is in matrix order (so the last letter is
    applied first). Its README says matrix order, its type stub says time order; trust neither."""
    cs, eps = _cyclosynth(), 1e-3
    cases = [
        (_gates_of(cs.synthesize_rz(0.3, eps), "calibration Rz"), rz_matrix(0.3)),
        (
            _gates_of(cs.synthesize_u3(1.0472, 2.7577, 5.3947, eps), "calibration U3"),
            rz_matrix(2.7577) @ ry_matrix(1.0472) @ rz_matrix(5.3947),
        ),
    ]
    best = []
    for reverse in (False, True):
        d = max(phase_distance(seq_matrix(_letters_to_gates(g, reverse)), m) for g, m in cases)
        best.append((d, reverse))
    d, reverse = min(best)
    if d > 10 * eps:
        raise RuntimeError(f"cannot interpret cyclosynth output (distance {d:.2e})")
    return reverse


def _to_gates(letters: str) -> list[str]:
    global _CS_REVERSED
    if _CS_REVERSED is None:
        _CS_REVERSED = _calibrate_cyclosynth()
    return _letters_to_gates(letters, _CS_REVERSED)


def synth_rz(theta: float, eps: float) -> list[str]:
    """Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2}) as Clifford+T gates (time order)."""
    k = round(theta / (math.pi / 4))
    if abs(theta - k * math.pi / 4) < 1e-9:
        return list(EXACT_RZ[k % 8])
    return _to_gates(_gates_of(_cyclosynth().synthesize_rz(theta, eps), f"Rz({theta})"))


def synth_1q(u: np.ndarray, eps: float, cliff_tol: float = 1e-7,
             synthesizer: str = "cyclosynth") -> list[str]:
    """Clifford+T gates (time order) equal to `u` up to a global phase.

    Exact for identity, Paulis and Cliffords (within `cliff_tol`). Otherwise `u` ~ Rz(phi) Ry(theta)
    Rz(lam) is synthesised. cyclosynth: a diagonal target as one Rz, a general one through its
    jointly optimised U3 route (valid for eps >= 1e-10), falling back to the rotation-by-rotation
    ZYZ route (eps/3 each) below that or if the joint search finds nothing. gridsynth: a diagonal
    target as one Rz, a general one as Rz(lam), Ry(theta) = S H Rz(theta) H Sdg, Rz(phi) at eps/3
    each.
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
    if synthesizer == "gridsynth":
        if abs(theta) < 1e-12:
            out = synth_rz_grid(phi + lam, eps)
        else:
            e3 = eps / 3
            out = (synth_rz_grid(lam, e3) + ["sdg", "h"] + synth_rz_grid(theta, e3)
                   + ["h", "s"] + synth_rz_grid(phi, e3))
    elif synthesizer != "cyclosynth":
        raise ValueError(f"unknown synthesizer {synthesizer!r}")
    elif abs(theta) < 1e-12:
        out = synth_rz(phi + lam, eps)
    else:
        cs = _cyclosynth()
        r = None
        if SYNTH_MODE == "u3" and eps >= 1e-10:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = cs.synthesize_u3(float(theta), float(phi), float(lam), eps)
        if not r or r.gates is None:
            r = cs.synthesize_zyz(float(phi), float(theta), float(lam), eps)
        out = _to_gates(_gates_of(r, "a general unitary"))
    out = _compress_s_runs(out)
    err = phase_distance(seq_matrix(out), u)
    if err > max(8 * eps, 1e-9):
        raise RuntimeError(f"synthesised sequence is {err:.2e} from the target (eps={eps:.1e})")
    return out


# ----------------------------------------------------------------------------- gridsynth
# The Ross-Selinger algorithm as BQSKit's Clifford+T flow runs it (bqskit-ft's GridSynthPass calls
# pygridsynth's gridsynth_gates). Its string is over H S T X W (W = global phase).
_GS_REVERSED: bool | None = None


def _gridsynth_gates():
    try:
        import mpmath
        from pygridsynth.gridsynth import gridsynth_gates
    except ImportError as e:  # pragma: no cover
        raise SystemExit("cannot import pygridsynth (pip install pygridsynth mpmath)") from e
    mpmath.mp.dps = max(mpmath.mp.dps, 128)  # as bqskit-ft sets it
    return mpmath, gridsynth_gates


def _grid_letters(theta: float, eps: float) -> str:
    mpmath, gridsynth_gates = _gridsynth_gates()
    return str(gridsynth_gates(mpmath.mpf(theta), mpmath.mpf(eps)))


def synth_rz_grid(theta: float, eps: float) -> list[str]:
    """Rz(theta) as Clifford+T gates (time order), by gridsynth."""
    global _GS_REVERSED
    k = round(theta / (math.pi / 4))
    if abs(theta - k * math.pi / 4) < 1e-9:
        return list(EXACT_RZ[k % 8])
    if _GS_REVERSED is None:  # bqskit-ft says matrix order; check, as for cyclosynth
        g = _grid_letters(0.3, 1e-3)
        d = [(phase_distance(seq_matrix(_letters_to_gates(g, rev)), rz_matrix(0.3)), rev)
             for rev in (False, True)]
        dist, _GS_REVERSED = min(d)
        if dist > 1e-2:
            raise RuntimeError(f"cannot interpret gridsynth output (distance {dist:.2e})")
    return _letters_to_gates(_grid_letters(theta, eps), _GS_REVERSED)


def t_count(seq: list[str]) -> int:
    return sum(g in ("t", "tdg") for g in seq)


# ----------------------------------------------------------------------------- lowering
class Lowerer:
    """Walks a dcsynth circuit and writes the Clifford+T dialect.

    Every approximation is recorded in ``ledger`` as (kind, error) in the phase-minimized
    operator norm, which bounds half the diamond distance of that gate: ``rot`` for a
    cyclosynth rotation (error <= eps), ``snap`` for a target replaced by a Clifford, Pauli or
    identity (error < cliff_tol). Errors add up over the circuit (see `lower`).

    With ``dry=True`` nothing is synthesized: each target is recorded with its distance to the
    nearest Clifford (kind ``item``), which is all `lower` needs to split an error budget."""

    def __init__(self, qc, eps: float, cliff_tol: float, dry: bool = False,
                 keep_cliffordt: bool = False, synthesizer: str = "cyclosynth") -> None:
        self.qc, self.eps, self.cliff_tol, self.dry = qc, eps, cliff_tol, dry
        self.keep_cliffordt, self.synthesizer = keep_cliffordt, synthesizer
        self.lines: list[str] = []
        self.pending: dict[int, np.ndarray] = {}
        self.stats: Counter = Counter()
        self._cache: dict[tuple, tuple[list[str], float]] = {}
        self.ledger: list[tuple[str, float]] = []

    # -- emission ----------------------------------------------------------------------
    def synth(self, u: np.ndarray) -> list[str]:
        key = _canon(u, 9)
        if self.dry:
            d = clifford_distance(u)
            self.ledger.append(("item", d))
            return nearest_clifford(u, self.cliff_tol) or ["t"]  # a placeholder rotation
        if key not in self._cache:
            word = synth_1q(u, self.eps, self.cliff_tol, self.synthesizer)
            self._cache[key] = (word, phase_distance(seq_matrix(word), u))
        word, err = self._cache[key]
        if err > 1e-15 or t_count(word):
            self.ledger.append(("rot" if t_count(word) else "snap", err))
        return word

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
                    d = phase_distance(v, I2)  # this branch's correction is dropped
                    if d > 1e-15:
                        self.ledger.append(("item" if self.dry else "snap", d))
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
            elif self.keep_cliffordt and op.name in KEPT_GATES and not op.num_clbits:
                self.flush(qs[0])  # an earlier unitary runs first, on its own
                self.emit(op.name, qs[0])
                self.stats["kept_gates"] += 1
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


def snap_threshold(eps: float) -> float:
    """Snap a target to a Clifford when it is within this distance, given rotation accuracy eps.

    Synthesizing a rotation to accuracy eps costs about c*log2(1/eps) T gates, so tightening all
    rotations to absorb extra error x costs about c*x/(eps*ln 2) T gates. Snapping a target at
    distance d costs nothing in T and d - eps more error than synthesizing it; it pays when
    c*log2(1/eps) > c*(d - eps)/(eps*ln 2), i.e. d < eps*(1 + ln(1/eps)), whatever c is."""
    return eps * (1.0 + math.log(1.0 / eps))


def allocate(distances: list[float], budget: float) -> tuple[float, float]:
    """Largest rotation accuracy eps whose total error fits `budget`, and its snap threshold.

    `distances` holds each target's distance to the nearest Clifford. A target within
    tau = snap_threshold(eps) is snapped (error = its distance); every other one is a rotation
    synthesized to eps. Every rotation gets the same eps, which minimizes the total T count
    when all rotations have the same cost curve (equal marginal cost per unit of error)."""
    d = np.sort(np.asarray(distances, dtype=float))
    prefix = np.concatenate([[0.0], np.cumsum(d)])

    def used(eps: float) -> float:
        k = int(np.searchsorted(d, snap_threshold(eps), side="left"))
        return prefix[k] + eps * (len(d) - k)

    if len(d) == 0:
        return budget, snap_threshold(budget)
    lo, hi = 1e-15, budget
    if used(hi) <= budget:
        return hi, snap_threshold(hi)
    for _ in range(200):  # bisection in log space
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if used(mid) <= budget else (lo, mid)
    return lo, snap_threshold(lo)


def lower(qc, epsilon: float = 1e-8, clifford_tol: float = 1e-7,
          synth_mode: str = "u3", error_budget: float | None = None,
          block_error: float = 0.0, keep_cliffordt: bool = False,
          synthesizer: str = "cyclosynth") -> tuple[str, Counter]:
    """Lower a dcsynth circuit; returns (OpenQASM 2 text, statistics).

    `synthesizer` (cyclosynth or gridsynth) synthesises the non-Clifford unitaries; with
    `keep_cliffordt`, gates already in Clifford+T are emitted as written (see the module docs).

    Error accounting. Every error is measured as half the diamond distance of the implemented
    operation from the ideal one, and these add up along the circuit. The total is at most
    block_error (dcsynth's realized block errors, from stage A) + the sum of the lowering's
    rotation errors (each <= eps) + its snap errors (each < clifford_tol); the statistics report
    the lowering's realized parts and the total.

    With `error_budget`, `epsilon` and `clifford_tol` are chosen instead: the part of the budget
    left after `block_error` is split by `allocate` (dry passes find the targets; they depend on
    the snap threshold, so the split is repeated until it is stable), and the circuit is lowered
    once with the result."""
    global SYNTH_MODE
    SYNTH_MODE = synth_mode
    if error_budget is not None:
        rest = error_budget - block_error
        if rest <= 0:
            raise ValueError(f"error budget {error_budget:.3g} is used up by the blocks "
                             f"({block_error:.3g}); compile the blocks more accurately")
        tau = clifford_tol
        for _ in range(6):
            dry = Lowerer(qc, rest, tau, dry=True, keep_cliffordt=keep_cliffordt,
                          synthesizer=synthesizer)
            dry.run()
            epsilon, new_tau = allocate([e for _, e in dry.ledger], rest)
            if abs(new_tau - tau) <= 1e-3 * new_tau:
                tau = new_tau
                break
            tau = new_tau
        clifford_tol = tau
    lw = Lowerer(qc, epsilon, clifford_tol, keep_cliffordt=keep_cliffordt,
                 synthesizer=synthesizer)
    text = lw.run()
    rot = [e for k, e in lw.ledger if k == "rot"]
    snap = [e for k, e in lw.ledger if k == "snap"]
    lw.stats.update({
        "epsilon": epsilon, "clifford_tol": clifford_tol,
        "n_rotations": len(rot), "n_snaps": len(snap),
        "error_rotations": float(sum(rot)), "error_snaps": float(sum(snap)),
        "error_lowering": float(sum(rot) + sum(snap)), "error_blocks": block_error,
        "error_total": float(sum(rot) + sum(snap) + block_error),
    })
    if error_budget is not None:
        lw.stats["error_budget"] = error_budget
        if lw.stats["error_total"] > error_budget * (1 + 1e-9):
            raise RuntimeError(f"realized error {lw.stats['error_total']:.3g} exceeds the budget "
                               f"{error_budget:.3g}")
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
    ap.add_argument("--epsilon", type=float, default=1e-8,
                    help="cyclosynth precision per 2x2 unitary (phase-minimized operator norm, "
                         ">= half the diamond distance); ignored with --error-budget")
    ap.add_argument(
        "--clifford-tol", type=float, default=1e-7,
        help="a 2x2 unitary this close to a Clifford is snapped to it exactly",
    )
    ap.add_argument(
        "--synth", choices=("u3", "zyz"), default="u3",
        help="u3: jointly optimised U3 (fewest T, slow at eps 1e-8); zyz: three fast Rz syntheses",
    )
    ap.add_argument(
        "--synthesizer", choices=("cyclosynth", "gridsynth"), default="cyclosynth",
        help="what synthesises non-Clifford unitaries: cyclosynth (default) or gridsynth "
             "(Ross-Selinger, via pygridsynth, as BQSKit's Clifford+T flow)",
    )
    ap.add_argument(
        "--keep-cliffordt", action="store_true",
        help="emit gates already in Clifford+T (h s sdg t tdg x y z sx sxdg) as written; only "
             "the other single-qubit unitaries are synthesised (for reassembled Clifford+T blocks)",
    )
    ap.add_argument(
        "--error-budget", type=float, default=None,
        help="total error allowed for the whole circuit (half diamond distance); chooses "
             "--epsilon and --clifford-tol so that block error + lowering error fits it",
    )
    ap.add_argument(
        "--block-error", type=float, default=0.0,
        help="realized error of dcsynth's blocks (stage A's variants.json `block_error`)",
    )
    args = ap.parse_args()
    qc = load_circuit(args.input)
    text, st = lower(qc, args.epsilon, args.clifford_tol, args.synth,
                     args.error_budget, args.block_error, args.keep_cliffordt, args.synthesizer)
    out = args.output or args.input.with_suffix(".cliffordt.qasm")
    out.write_text(text)
    print(
        f"wrote {out}: {qc.num_qubits} qubits, {qc.num_clbits} classical bits, "
        f"{st['measure']} measure, {st['reset']} reset, {st['if_else']} if_else, "
        f"{st['cx']} cx, T count {st['t']} unconditional + {st['conditional_t']} conditional "
        f"({st['conditional_gates']} conditional gates in all)",
        file=sys.stderr,
    )
    print(
        f"error (half diamond distance): blocks {st['error_blocks']:.3g} + rotations "
        f"{st['error_rotations']:.3g} ({st['n_rotations']} at eps {st['epsilon']:.3g}) + snaps "
        f"{st['error_snaps']:.3g} ({st['n_snaps']}, tol {st['clifford_tol']:.3g}) = "
        f"{st['error_total']:.3g}"
        + (f" of budget {st['error_budget']:.3g}" if "error_budget" in st else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
