#!/usr/bin/env python3
"""Tests for dcsynth_to_cliffordt.py, and the dcsynth -> transpile -> oracle pipeline.

Run (needs numpy, qiskit, pygridsynth; the `transpile` binary must be built):

  cargo build --release --bin transpile
  python data_processing/test_dcsynth_lowering.py            # or: pytest data_processing/

What is checked:
  * synthesis: identity/Pauli/Clifford are exact, a random unitary is within the requested epsilon;
  * lowering: the exact joint distribution of all measurement outcomes of the original Qiskit
    circuit (measure / reset / nested if_else / UnitaryGate) equals that of the lowered QASM;
  * the lowered QASM then goes through `transpile` in every reset mode and width, and
    validate_dynamic.py's oracle confirms the .trans file + sidecar still match it;
  * what cannot be expressed is rejected, not silently dropped.

The if_else structures are built with dcsynth's own `_conditional` when DCSYNTH_SRC points at its
`src/dcsynth` directory (dcsynth needs bqskit, which is stubbed out here); otherwise with a copy
of that function, so the tests always run.
"""
from __future__ import annotations

import importlib
import math
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcsynth_to_cliffordt as lowering  # noqa: E402
import validate_dynamic as vd  # noqa: E402

from qiskit import ClassicalRegister, QuantumCircuit  # noqa: E402
from qiskit.circuit.library import UnitaryGate  # noqa: E402
from qiskit.quantum_info import random_unitary  # noqa: E402

ROOT = HERE.parent
TRANSPILE = next(
    (p for p in (ROOT / "target" / "release" / "transpile", ROOT / "target" / "debug" / "transpile")
     if p.exists()),
    ROOT / "target" / "release" / "transpile",
)
EPS = 1e-7


# ----------------------------------------------------------------------------- dcsynth's `_conditional`
def _load_dcsynth_conditional():
    src = os.environ.get("DCSYNTH_SRC")
    if not src or not Path(src).is_dir():
        return None

    class Stub(types.ModuleType):
        def __getattr__(self, n):
            if n.startswith("__"):
                raise AttributeError(n)
            return object

    pkg = types.ModuleType("dcsynth")
    pkg.__path__ = [src]
    sys.modules["dcsynth"] = pkg
    for _ in range(10):
        try:
            return importlib.import_module("dcsynth.assembly.schedule")._conditional
        except ModuleNotFoundError as e:
            sys.modules[e.name] = Stub(e.name)
    return None


def _conditional_copy(qc, bits, bodies, qubits):
    """A copy of dcsynth.assembly.schedule._conditional: bodies[v] runs when `bits` read v."""

    def with_bits(body):
        out = QuantumCircuit(len(qubits))
        out.add_bits(list(bits))
        out.compose(body, qubits=list(range(len(qubits))), inplace=True)
        return out

    if len(bits) == 1:
        qc.if_else((bits[0], False), with_bits(bodies[0]), with_bits(bodies[1]), qubits, list(bits))
        return
    half = len(bodies) // 2
    outer = []
    for m in (0, 1):
        body = QuantumCircuit(len(qubits))
        body.add_bits(list(bits))
        _conditional_copy(body, list(bits[1:]), bodies[m * half:(m + 1) * half], list(range(len(qubits))))
        outer.append(body)
    qc.if_else((bits[0], False), outer[0], outer[1], qubits, list(bits))


CONDITIONAL = _load_dcsynth_conditional() or _conditional_copy
USING_REAL_DCSYNTH = CONDITIONAL is not _conditional_copy


# ----------------------------------------------------------------------------- reference simulator
def _tasks(circ, qmap, cmap):
    """Flatten a Qiskit circuit to ('u'|'cx'|'measure'|'reset'|'if', ...) tasks."""
    from qiskit.circuit import IfElseOp
    from qiskit.quantum_info import Operator

    out = []
    for inst in circ.data:
        op = inst.operation
        qs = [qmap[b] for b in inst.qubits]
        if op.name == "barrier":
            continue
        if op.name == "measure":
            out.append(("measure", qs[0], cmap[inst.clbits[0]]))
        elif op.name == "reset":
            out.append(("reset", qs[0]))
        elif isinstance(op, IfElseOp):
            blocks = []
            for blk in op.blocks:
                bq = {b: qmap[o] for b, o in zip(blk.qubits, inst.qubits)}
                bc = {b: cmap[o] for b, o in zip(blk.clbits, inst.clbits)}
                blocks.append(_tasks(blk, {**qmap, **bq}, {**cmap, **bc}))
            cond_t, cond_v = op.condition
            bits = list(cond_t) if hasattr(cond_t, "__iter__") else [cond_t]
            idx = [cmap[b] if b in cmap else None for b in bits]
            if any(i is None for i in idx):  # a body-local bit: resolve through the blocks' maps
                idx = [({**cmap, **bc}).get(b) for b in bits]
            if hasattr(cond_t, "__iter__"):
                want = [(int(cond_v) >> k) & 1 for k in range(len(bits))]
            else:
                want = [int(bool(cond_v))]
            out.append(("if", list(zip(idx, want)), blocks[0], blocks[1] if len(blocks) > 1 else []))
        elif op.name == "cx":
            out.append(("cx", qs[0], qs[1]))
        else:
            out.append(("u", qs[0], Operator(op).data))
    return out


def run_qiskit(qc) -> dict:
    """Exact joint distribution of all measure/reset outcomes, keyed like validate_dynamic's."""
    qmap = {b: i for i, b in enumerate(qc.qubits)}
    cmap = {b: i for i, b in enumerate(qc.clbits)}
    dist: dict = defaultdict(float)

    def rec(tasks, st, prob, clbits, events, occ):
        while tasks:
            t, tasks = tasks[0], tasks[1:]
            kind = t[0]
            if kind == "u":
                st = st.copy()
                st.apply1(t[2], t[1])
            elif kind == "cx":
                st = st.copy()
                st.apply_cx(t[1], t[2])
            elif kind == "if":
                taken = all(clbits.get(i, 0) == w for i, w in t[1])
                tasks = (t[2] if taken else t[3]) + tasks
            else:  # measure / reset
                q = t[1]
                for bit, p, post in st.measure_pauli({q: "Z"}):
                    ev = dict(events)
                    ev[(q, occ.get(q, 0))] = bit
                    oc = dict(occ)
                    oc[q] = occ.get(q, 0) + 1
                    cb = dict(clbits)
                    if kind == "measure":
                        cb[t[2]] = bit
                    elif bit == 1:
                        post.apply1(vd.G1["x"], q)
                    rec(tasks, post, prob * p, cb, ev, oc)
                return
        dist[tuple(sorted(events.items()))] += prob

    rec(_tasks(qc, qmap, cmap), vd.State(qc.num_qubits), 1.0, {}, {}, {})
    return dist


def run_lowered(text: str, tmp_path: Path, name: str):
    path = tmp_path / f"{name}.cliffordt.qasm"
    path.write_text(text)
    n, ops = vd.parse_qasm(str(path))
    return path, vd.run_reference(n, ops)


# ----------------------------------------------------------------------------- circuits
def finish(qc: QuantumCircuit) -> QuantumCircuit:
    """Measure every qubit explicitly so both simulators key the same outcomes."""
    tail = ClassicalRegister(qc.num_qubits, "fin")
    qc.add_register(tail)
    for q in range(qc.num_qubits):
        qc.measure(q, tail[q])
    return qc


def ancilla_block(qc, data: list[int], anc: int, bits_reg, gates) -> None:
    """compute onto the ancilla, measure it, reset it: the shape of one dynamic block."""
    for kind, *a in gates:
        getattr(qc, kind)(*a)
    qc.measure(anc, bits_reg)
    qc.reset(anc)


def perqubit_corrections(qc, bits, per_qubit_unitaries: dict[int, list[np.ndarray]]) -> None:
    """dcsynth's 'perqubit' layout: one `_conditional` per dependent data qubit."""
    for q, us in per_qubit_unitaries.items():
        bodies = []
        for u in us:
            body = QuantumCircuit(1)
            body.append(UnitaryGate(u), [0])
            bodies.append(body)
        CONDITIONAL(qc, list(bits), bodies, [q])


def circ_pauli_corrections() -> QuantumCircuit:
    """Corrections that differ by a Pauli: must lower to frame updates, no conditional T."""
    qc = QuantumCircuit(3)
    c = ClassicalRegister(1, "c0")
    qc.add_register(c)
    qc.h(0), qc.t(0), qc.cx(0, 2), qc.h(2), qc.t(2), qc.cx(2, 1)
    qc.measure(2, c[0]), qc.reset(2)
    qc.t(1), qc.h(1)
    perqubit_corrections(
        qc, [c[0]],
        {0: [np.eye(2), vd.G1["z"]], 1: [vd.G1["x"] @ vd.G1["h"], vd.G1["h"]]},
    )
    qc.t(0)
    return finish(qc)


def circ_clifford_corrections() -> QuantumCircuit:
    """Figure 1 of the paper: H and S conditioned on mid-circuit outcomes (non-Pauli Cliffords)."""
    qc = QuantumCircuit(3)  # data 0, 1; ancilla 2
    m0, m1 = ClassicalRegister(1, "m0"), ClassicalRegister(1, "m1")
    qc.add_register(m0), qc.add_register(m1)
    qc.h(0), qc.t(0), qc.cx(0, 1), qc.cx(0, 2), qc.t(1), qc.cx(1, 2), qc.s(2)
    qc.measure(2, m0[0]), qc.reset(2)
    with qc.if_test((m0[0], 1)):
        qc.h(0)
    qc.h(2), qc.cx(1, 2)
    qc.measure(2, m1[0]), qc.reset(2)
    with qc.if_test((m1[0], 1)):
        qc.x(1)
        qc.s(0)
    qc.t(1)
    return finish(qc)


def circ_generic_corrections() -> QuantumCircuit:
    """Two bits (nested if_else), one correction that is genuinely non-Clifford relative to the rest.

    U_a = base, Z.base, S.base, base.Z: the first three differ from `base` by I, Z, S (free or
    Clifford), but base.Z.base^dagger is a pi-rotation about a generic axis, so one conditional
    T-type rotation cannot be avoided.
    """
    qc = QuantumCircuit(3)
    c = ClassicalRegister(2, "c")
    qc.add_register(c)
    qc.h(0), qc.cx(0, 2), qc.t(0), qc.cx(2, 1)
    qc.measure(2, c[0]), qc.reset(2)
    qc.h(2), qc.t(2), qc.cx(1, 2), qc.cx(0, 2)
    qc.measure(2, c[1]), qc.reset(2)
    base = random_unitary(2, seed=3).data
    perqubit_corrections(
        qc, [c[0], c[1]],
        {0: [base, vd.G1["z"] @ base, vd.G1["s"] @ base, base @ vd.G1["z"]]},
    )
    qc.cx(0, 1)
    return finish(qc)


def circ_correction_after_block() -> QuantumCircuit:
    """What dcsynth's corrections look like: U_a = P_a.U, a Pauli (or Clifford) *after* a generic U.

    Every difference U_a U_ref^dagger is then that Pauli/Clifford, so no conditional T is needed
    however non-Clifford U is.
    """
    qc = QuantumCircuit(4)  # data 0..2, ancilla 3
    c = ClassicalRegister(2, "c")
    qc.add_register(c)
    qc.h(0), qc.h(1), qc.t(0), qc.cx(0, 3), qc.cx(1, 3), qc.t(3), qc.cx(2, 3)
    qc.measure(3, c[0]), qc.reset(3)
    qc.h(3), qc.cx(0, 3), qc.t(3), qc.cx(2, 3), qc.h(3)
    qc.measure(3, c[1]), qc.reset(3)
    u0 = random_unitary(2, seed=11).data
    u1 = random_unitary(2, seed=12).data
    z, x, s = vd.G1["z"], vd.G1["x"], vd.G1["s"]
    perqubit_corrections(qc, [c[0], c[1]], {
        0: [u0, z @ u0, x @ u0, z @ x @ u0],       # Pauli-after: free frame updates
        1: [u1, s @ u1, z @ u1, s @ z @ u1],       # Clifford-after: a few physical gates
    })
    qc.cx(0, 1), qc.cx(1, 2)
    return finish(qc)


def circ_shared_layout() -> QuantumCircuit:
    """dcsynth's SHARED layout: one if_else whose leaves touch several data qubits."""
    qc = QuantumCircuit(3)
    c = ClassicalRegister(1, "c")
    qc.add_register(c)
    qc.h(0), qc.h(1), qc.cx(0, 2), qc.cx(1, 2), qc.t(2)
    qc.measure(2, c[0]), qc.reset(2)
    b0, b1 = QuantumCircuit(2), QuantumCircuit(2)
    b0.append(UnitaryGate(vd.G1["h"]), [0]), b0.append(UnitaryGate(vd.G1["z"]), [1])
    b1.append(UnitaryGate(vd.G1["h"] @ vd.G1["s"]), [0]), b1.append(UnitaryGate(vd.G1["z"]), [1])
    CONDITIONAL(qc, [c[0]], [b0, b1], [0, 1])
    qc.t(0), qc.t(1)
    return finish(qc)


CIRCUITS = {
    "pauli": circ_pauli_corrections,
    "fig1_cliffords": circ_clifford_corrections,
    "generic": circ_generic_corrections,
    "after_block": circ_correction_after_block,
    "shared": circ_shared_layout,
}


# ----------------------------------------------------------------------------- tests
def test_clifford_table_has_24_and_snaps():
    assert len(lowering.CLIFFORDS) == 24
    for key, word in lowering.CLIFFORDS.items():
        u = lowering.seq_matrix(word)
        assert lowering.phase_distance(u, lowering.seq_matrix(lowering.nearest_clifford(u, 1e-9))) < 1e-9


def test_synthesis_is_exact_for_cliffords_and_close_otherwise():
    for name in ("x", "y", "z", "h", "s", "sdg"):
        seq = lowering.synth_1q(lowering.G[name], EPS)
        assert lowering.phase_distance(lowering.seq_matrix(seq), lowering.G[name]) < 1e-12
        assert lowering.t_count(seq) == 0
    assert lowering.synth_1q(np.eye(2), EPS) == []
    assert lowering.synth_1q(lowering.G["z"], EPS) == ["z"]
    for seed in range(4):
        u = random_unitary(2, seed=seed).data
        seq = lowering.synth_1q(u, EPS)
        assert lowering.phase_distance(lowering.seq_matrix(seq), u) < 8 * EPS, seed
        assert lowering.t_count(seq) > 0


def _lower_and_compare(name: str, tmp_path: Path) -> Path:
    qc = CIRCUITS[name]()
    text, stats = lowering.lower(qc, epsilon=EPS)
    path, got = run_lowered(text, tmp_path, name)
    want = run_qiskit(qc)
    tv = vd.tv_distance(want, got)
    assert tv < 5e-5, f"{name}: lowered QASM differs from the Qiskit circuit (TV {tv:.2e})"
    return path, stats, text


def test_lowering_matches_the_qiskit_circuit(tmp_path: Path):
    for name in CIRCUITS:
        _lower_and_compare(name, tmp_path)


def test_pauli_and_clifford_corrections_cost_no_conditional_t(tmp_path: Path):
    for name in ("pauli", "fig1_cliffords", "shared", "after_block"):
        _, stats, text = _lower_and_compare(name, tmp_path)
        assert stats["conditional_t"] == 0, f"{name}: a conditional T was emitted\n{text}"
    _, stats, text = _lower_and_compare("pauli", tmp_path)
    cond = [l for l in text.splitlines() if l.startswith("if(")]
    assert cond and all(l.split()[-2] in ("x", "y", "z") for l in cond), cond


def test_reference_is_chosen_to_make_the_difference_cheap(tmp_path: Path):
    # `generic` has one correction (base.Z) that is a generic rotation away from `base`: the
    # reference must be `base` so that exactly that one assignment pays for T gates, not three.
    _, stats, text = _lower_and_compare("generic", tmp_path)
    cond = [l for l in text.splitlines() if l.startswith("if(")]
    patterns = {l[: l.index(")") + 1] for l in cond if l.split()[-2] in ("t", "tdg", "h")}
    assert len(patterns) == 1, f"T-type gates under {len(patterns)} different conditions: {patterns}"
    assert stats["conditional_t"] > 0


def test_through_transpile_and_the_oracle(tmp_path: Path):
    assert TRANSPILE.exists(), f"build it first: cargo build --release --bin transpile ({TRANSPILE})"
    for name in CIRCUITS:
        path, _, _ = _lower_and_compare(name, tmp_path)
        ok = vd.check(path, TRANSPILE, ["frame", "flush", "init"], [0, 1, 2], [False, True],
                      verbose=False)
        assert ok, f"{name}: transpile output no longer matches the circuit"


def test_unsupported_constructs_raise():
    qc = QuantumCircuit(2)
    c = ClassicalRegister(1, "c")
    qc.add_register(c)
    qc.h(0), qc.measure(0, c[0])
    with qc.if_test((c[0], 1)):
        qc.cx(0, 1)
    try:
        lowering.lower(qc)
    except NotImplementedError as e:
        assert "single-qubit" in str(e)
    else:
        raise AssertionError("a conditional cx must be rejected")
    qc = QuantumCircuit(2)
    qc.add_register(c)
    qc.h(0), qc.measure(0, c[0])
    with qc.if_test((c[0], 1)):
        qc.measure(1, c[0])
    try:
        lowering.lower(qc)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("a measurement inside an if_else must be rejected")


def main() -> int:
    import tempfile

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    print(f"if_else built with {'the real dcsynth _conditional' if USING_REAL_DCSYNTH else 'a copy of it'}")
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d)) if fn.__code__.co_argcount else fn()
                print(f"  ok   {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print("ALL PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
