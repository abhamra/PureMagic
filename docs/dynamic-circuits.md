# Dynamic circuits: reset, mid-circuit measurement, feed-forward

PureMagic's original input is a static Clifford+T circuit: every measurement is at the end, and
Cliffords are commuted there too. A *dynamic* circuit (for example the output of the `dcsynth`
ancilla-reuse compiler) measures a qubit in the middle, resets it, reuses it, and applies gates
that depend on the measurement outcomes. This document describes how that is supported.

Nothing here changes how a static circuit is transpiled or scheduled. Without `reset`, `if(...)`
or a mid-circuit `measure`, the outputs are identical to before (the `--sidecar` file is the only
new artifact and is not written unless asked for).

## Pipeline

```
dcsynth circuit (Qiskit)        dcsynth_to_cliffordt.py      transpile                    puremagic
  measure/reset/if_else   --->   Clifford+T QASM2 dialect --->  .trans + .trans.dyn.json ---> schedule
  UnitaryGate bodies             (1-bit cregs, if(m3==1) g)    (+ `@id` classical deps)      (--reaction-latency L)
```

```bash
cargo build --release --bins
# in dcsynth's environment, on the circuit it assembled (see "What dcsynth emits"):
#   from qiskit import qpy; qpy.dump(qc, open("circuit.qpy", "wb"))
python data_processing/dcsynth_to_cliffordt.py circuit.qpy -o circuit.cliffordt.qasm --epsilon 1e-8
./target/release/transpile -i circuit.cliffordt.qasm -o circuit.trans --defer_trailing   # --reset_mode frame|flush|init
./target/release/puremagic -c circuit.trans --reaction-latency 0
```

## What dcsynth emits, and why not plain QASM

`dcsynth.assembly.schedule.build_and_check` (and `assemble`) return a Qiskit `QuantumCircuit`:

* `qc.measure(ancilla, bit)` followed immediately by `qc.reset(ancilla)` ends every dynamic block;
* the data qubits' corrections are `if_else((bit, False), body0, body1, ...)` where each body is a
  single-qubit `UnitaryGate` (default "perqubit" layout), nested one level per extra bit;
* the blocks themselves (`block*_main.qasm`) are BQSKit output: `cx` and `u3`.

OpenQASM 2 can say `measure`, `reset` and `if(creg==n) gate q[i];`, but not nested blocks, and the
`UnitaryGate`s are arbitrary 2x2 matrices that have to become Clifford+T anyway. OpenQASM 3 has
`if (c[0] == 1) { ... }` blocks, but still no Clifford+T synthesis, and `compile_cliffordt`'s loader
only skips `measure` statements; it has no reset or conditional support. So the bridge is a
Python lowering (`data_processing/dcsynth_to_cliffordt.py`) that writes a restricted dialect:

* one 1-bit register per classical bit (`creg m3[1];`);
* `measure q[i] -> m3[0];` and `reset q[i];` as written;
* a gate that depends on outcomes: `if(m3==1 && m5==0) t q[2];`, one single-qubit gate per line,
  gate in h, s, sdg, sx, sxdg, x, y, z, t, tdg. (`&&` is an extension; with one term the line is
  plain OpenQASM 2.)

The lowering applies the cheapest reference unitary unconditionally and conditions only the
difference. dcsynth's corrections are `P_a . U` (a Pauli or Clifford after a block's unitary), so
the differences are Paulis, which are free frame updates, instead of conditional T rotations.

## The three ideas in `transpile`

**1. Measurement needs nothing special.** In the Heisenberg picture a measurement of `Z_q` after
Cliffords `C` is the product `C^dagger Z_q C`, emitted in program order, like a final measurement.

**2. Reset is a measurement plus a classical correction.** After `M_Z -> m` the qubit is `|m>`;
resetting to `|0>` is `X^m`, a Pauli. Pauli corrections are never executed on the lattice. They
join a classical *Pauli frame*: per qubit, the XOR of outcomes that owe an `X` and a `Z`. A later
product `P` is measured as `+-P` depending on whether it anticommutes with the frame, so the
product's sign becomes `sign XOR (XOR of outcomes)`. The scheduler never reads signs, so none of
this changes the schedule. A conditional Pauli (`if(m==1) x q[i];`) is the same frame update, and
a Pauli on several bits uses a derived AND outcome so the frame stays linear.

**3. A conditional non-Pauli gate is physical.** `if(m==1) h q[i];` cannot be a frame update. The
transpiler flushes pending Cliffords, pays off the qubit's owed Pauli (an `Event::Fix`), and emits
the gate as ordinary products tagged with their condition and a classical dependency `@id,..` on
the measurement lines whose outcomes decide the condition. The dependency makes a DAG edge
even though no qubit is shared.

### Reset modes (`transpile --reset_mode`)

| mode | what the reset becomes | cost |
|------|------------------------|------|
| `frame` (default) | `<M>` on the qubit, then a frame update | no extra product |
| `flush` | like `frame`, but flush pending Cliffords first so the measurement is weight 1 and later products start from a clean frame | extra Cliffords possible |
| `init` | like `flush`, plus an `<R>` product occupying the patch for one lcycle | models a physical re-initialisation |

`frame` is the faithful model if re-preparing a patch is free or overlaps with other work. `init`
is the conservative one.

## Files in `.trans`

* A line may end in `@3,5`: ids (0-based line numbers) of earlier products whose *outcome* it
  needs. `<R>` is a one-qubit re-initialisation. Both are parsed in `pauliproduct.rs` and become
  edges in `Circuit::gen_deps`.
* `<output>.dyn.json` (written for any dynamic circuit): everything an executor needs to
  recover true outcomes: per measurement `meas_id`, `clbit`, `is_reset`, `frame_deps`, derived AND
  outcomes, `fix` events. With `p` the physical outcome of product `k`:
  `true_outcome(k) = p XOR sign(k) XOR (XOR of true_outcome(j) for j in frame_deps(k))`.

## Scheduling and "real time"

PureMagic is dynamic because magic-state cultivation times are random: every lcycle MINFIT picks
among ready products and the pool of ready magic states changes. The dependency graph is static.
Feed-forward adds two things:

* **Edges.** `@` dependencies are ordinary parents, so nothing starts before the outcome it needs.
  Pauli/Clifford corrections that stay in the frame add no edges at all.
* **Reaction latency.** A logical outcome is usable only after its syndrome rounds are decoded.
  `puremagic --reaction-latency L` delays a product with an `@` edge until `L` lcycles after the
  product it names completes (default 0 = ideal). Qubit-overlap edges are not delayed; if a child
  has both kinds on the same parent the whole edge is delayed (conservative). The scheduler loop
  keeps running while delayed releases are outstanding.

Latency is paid only where a measurement is on the critical path. For dcsynth-style circuits the
corrections are mostly Paulis, so almost nothing depends on an outcome and the latency is hidden;
`tests/fixtures/dyn_pingpong.cliffordt.qasm` is the opposite case (three hops, so `L` costs `3L`
lcycles).

## Verifying

```bash
cargo test                                                     # unit + integration tests
python data_processing/validate_dynamic.py circuit.cliffordt.qasm            # one circuit
python data_processing/validate_dynamic.py --random 150 --seed 1             # fuzz
python data_processing/test_dcsynth_lowering.py                # converter, end to end
# optional: DCSYNTH_SRC=/path/to/Artifact/src/dcsynth uses dcsynth's own if_else builder
```

`validate_dynamic.py` is the oracle: it simulates the input QASM and the `.trans` file plus sidecar
exactly (statevector, branching at every measurement) and compares the joint distribution of all
outcomes (total variation < 1e-9) in every reset mode and width.

## Visualizing and checking a schedule

```bash
./target/release/puremagic -c circuit.trans --reaction-latency 5
python data_processing/plot_dynamic.py circuit.trans            # reads circuit.schedule, circuit.trans.dyn.json
python data_processing/plot_dynamic.py circuit.trans --window 0:600 -o out/zoom
```

`plot_dynamic.py` re-derives the constraints from the `.trans` file and checks the `.schedule`
output against them, without using the scheduler's code: every product scheduled once (CX two
consecutive lcycles, S/SX a short consecutive run), a product starts after the previous product on
each of its qubits, a product with `@k` starts at least `L+1` lcycles after product `k` ended, no qubit
is used twice in one lcycle, and the sidecar is consistent (every `needs` has its `@` edge; frame
dependencies and fix events only use outcomes already measured). It exits 1 on any violation. It writes
`<prefix>.timeline.png` (one row per qubit; conditional products outlined in red; an arrow from each
measurement to what waits on it, so the reaction gap is visible) and `<prefix>.overview.png`
(occupancy per qubit over the whole run plus cumulative T and measurement counts). The timeline
shows the first ~3000 products unless `--window` is given. Outcome-correctness (not just ordering) is
what `validate_dynamic.py` checks.

## Limits

* Conditional CX/CZ, a measurement inside an `if`, and classical expressions are rejected, not
  dropped. dcsynth's default "perqubit" layout never needs them.
* Rz synthesis is approximate (`--epsilon`, per rotation); Pauli and Clifford corrections are exact.
* Several conditions on one `if_else` are completed to all `2^k` assignments (k <= 8).
* `compile_cliffordt` does not handle dynamic circuits; use the converter, which synthesises
  gates itself with `pygridsynth`.
* The model has no notion of the ancilla being a different kind of patch: every qubit, ancilla
  included, is a logical data patch in the layout.
