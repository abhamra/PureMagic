# Mid-circuit measurement in PureMagic: what changed, and how it works

This is the specification of everything the MCM work adds to PureMagic: the model, the file
formats, each component's changes, the timing model, what is verified, what it costs and what
is not done yet. `docs/dynamic-circuits.md` is the user guide (how to run it); this document is
the reference for how it works. Branch: `mcm-ancilla`, which contains `mcm-support` and
`ancilla-patch`.

| Commit | Branch | What |
|---|---|---|
| 6dd5b41, 80535d0 | mcm-support | Dynamic-circuit transpilation, `.trans` `@` dependencies, sidecar, reaction latency, validator, checker, dcsynth lowering |
| 7613cbc | mcm-support | Lowering synthesizes rotations with cyclosynth |
| 16ed103 | mcm-support | Checker accepts the scheduler's own S corrections |
| 2a0d3a7 | ancilla-patch | `--ancilla-qubits`: ancillas hosted on injection patches |
| 3481138, e8659f3 | mcm-ancilla | Merge; hosted patches sleep between uses |
| 7a1fdda | mcm-ancilla | Conditions in `.trans` (`?`); scheduler skips products whose condition fails (`--cond-outcomes`) |
| d697a79 | mcm-ancilla | Conditional CX/CZ/SWAP; flushed CZ/SWAP no longer dropped |

## 1. Model

PureMagic works in the Heisenberg picture (Pauli-based computation): every Clifford is pushed to
the end of the circuit, so the program is a sequence of Pauli product rotations (each T becomes a
π/8 rotation on the conjugated Pauli `C†Z_qC`) and Pauli product measurements. A dynamic circuit
adds three things. Each is handled as follows.

1. **Ordinary gates.** Unchanged: Cliffords go into the tableau, and each T becomes a π/8 product.
2. **Mid-circuit measurement.** A product measurement of the conjugated `Z`, `C†Z_qC`, emitted in
   program order exactly like a final measurement. It is often multi-qubit (in grover_5 DC-opt,
   `measure q[5]` becomes `+YZ_Z_X_<M>`). It gets an outcome id.
   **Reset** is that measurement plus a correction: the qubit holds `|m⟩`, so it is owed `X^m`.
   The correction goes into a classical **Pauli frame** (per qubit, the XOR of the outcomes that
   owe an X and a Z), not onto the lattice. A later product whose Pauli anticommutes with the
   frame only changes sign, which is recorded as `frame_deps`.
3. **Conditional operations**, by gate type:

   | Conditional gate | Treatment | Why |
   |---|---|---|
   | Pauli (`if(m) x q`) | Frame update only; no product | A Pauli commutes through everything up to a sign |
   | Pauli under several bits | Frame update on a *derived* AND outcome | Keeps the frame linear in the outcomes |
   | Clifford (`if(m) h/s/sx q`) | **Physical**: flush pending Cliffords, pay off the qubit's frame (`fix` event), emit `<S>`/`<SX>` products carrying the condition | Absorbing it into the tableau would make every later product depend on the outcome |
   | T (`if(m) t q`) | **Physical**: same flush and frame payment, then a weight-1 conditional `<T>` | Strictly only the condition is needed; see §8 |
   | CX / CZ / SWAP (`if(m) cx q[a],q[b]`) | **Physical**, like a conditional Clifford: flush, pay off the frame on *both* qubits, emit conditional `<CX>` (CZ as H·CX·H, SWAP as three CX) | Same reason; both qubits must be frame-free (d697a79) |

   A measurement or reset inside an `if`, gates on three or more qubits, and classical
   expressions are rejected with an error, not dropped.

A conditional product carries two things:
- **when**: the `@` list, the measurements whose outcome it waits for. This is a DAG edge plus
  reaction latency.
- **if**: the `?` condition, the outcome values for which it runs.

## 2. File formats

### `.trans` (read by the scheduler)

One product per line; the 0-based line number is the product id.

```
line   := sign pauli "<" kind ">" [ "@" ids ] [ "?" terms ]
sign   := "+" | "-"
pauli  := one of "_XYZ" per qubit
kind   := T | M | R | CX | S | Sdg | SX | SXdg
ids    := id ("," id)*          earlier products whose outcome this one needs
terms  := id "=" ("0"|"1") ("," id "=" ("0"|"1"))*
```

- `<R>` (new): re-initialize one patch to `|0⟩` (only with `--reset_mode init`).
- `@3,5` (new): needs the outcomes of products 3 and 5. Each named product must be earlier.
- `?3=1,5=0` (new, 7a1fdda): runs only if the true outcome of product 3 is 1 and of product 5
  is 0. Every id in `?` must also be in `@` and must be a `<M>` (checked when the circuit is
  loaded). `@` can name more products than `?`: the outcomes of a paid-off frame are needed
  even though they do not decide whether the product runs.

Example (grover_5 DC-opt):

```
+YZ_Z_X_<M>                 line 1304: mid-circuit measurement (meas_id 0)
+___Z___<S>@1304?1304=1     an S on q3, after outcome 1304 is known, only if it is 1
+___X___<SX>@1304?1304=1
```

A file without `@`/`?` is an ordinary static circuit; nothing about static circuits changed.

### `.trans.dyn.json` (sidecar; read by executors, the validator and the checker, not the scheduler)

Written for every dynamic circuit (`transpile --sidecar` forces it for static ones). Version 2:

- `products`: one record per `.trans` line, in order.
  - Every record: `i`, `label`.
  - T/M/R: `sign`, `frame_deps` (outcome ids whose XOR flips the sign).
  - M: `meas_id`, `logical_qubit`, `clbit`, `is_reset`.
  - Conditional records: `cond` (`[outcome id, value]` pairs) and `needs` (outcome ids = the `@` list).
- `events`: `fix` (before product `before`, apply `X^(…)Z^(…)` to a qubit to pay off its frame)
  and `derived` (an AND outcome's definition).
- `derived`: AND outcomes and their terms.

`true_outcome(k) = physical(k) XOR sign(k) XOR (XOR of true_outcome(j) for j in frame_deps(k))`.

The sidecar uses **outcome ids** (`meas_id`); `.trans` uses **product (line) ids**. `cond` in the
sidecar and `?` in `.trans` are the same condition in the two numberings; the validator checks
that they agree.

### `.schedule` header (new lines, only when the circuit has conditions)

```
# cond_outcomes: random p1=0.5
# outcomes: 1304=1 8217=0 11126=1      sampled true outcome per measurement product id
# skipped: 1359 1360 ...               conditional products not run
```

## 3. Transpiler (`src/transpile.rs`, binary `transpile`)

- **Parsing**: 1-bit classical registers (`creg m3[1];`), `measure q -> m[0]`, `reset q`, and
  `if(r==v [&& r2==v2 ...]) g q[i];` (`parse_conditional`). Conditions whose bit no measurement
  has written yet are settled statically (`resolve_static_conditions`: an unwritten bit reads 0).
- **Frame** (`PauliFrame`): per-qubit sets of outcome ids owing X and Z, kept in the physical
  coordinates the tableau defines. It is conjugated whenever Cliffords are flushed, and gains a
  constant term for any Pauli the flushed gates drop (`dropped_pauli`).
- **Measurement**: product `C†Z_qC`, outcome id, `frame_deps`. A qubit whose last operation is
  not a measurement or reset gets a final measurement; conditional Paulis do not count as an
  operation for this.
- **Reset** (`--reset_mode`): `frame` (default) adds `X^m` to the frame; `flush` first flushes
  pending Cliffords, so the measurement is weight 1 and later products start from a clean frame;
  `init` also emits `<R>` and clears the qubit's frame.
- **Conditional gates**: see §1. Physical ones: `flush_clifford_queue` (the whole queue, every
  qubit), `pay_frame` → `Event::Fix` for every qubit the gate touches, then products with
  `meta.cond` and `meta.needs` (condition outcomes ∪ the fixes' outcomes). A conditional
  Clifford's dropped Pauli joins the frame on a derived outcome.
- **CZ and SWAP** have no `.trans` form; `expand_for_emit` rewrites them into CX (+H) wherever
  Cliffords are emitted. Before d697a79 a CZ or SWAP pending at a flush was silently dropped
  (wrong output; the dcsynth lowering only emits `cx`, so no benchmark was affected).
- **Output**: `write_trans` writes `@` (needs) and `?` (cond), mapping outcome ids to
  measurement lines. `write_sidecar` writes the sidecar. CLI: `--reset_mode`, `--sidecar`.

## 4. Circuit loading (`src/pauliproduct.rs`, `src/circuit.rs`)

- `PauliProduct` gains `after` (`@`), `cond` (`?`) and the `R` kind.
- `Circuit::gen_deps` adds a DAG edge for every `@` id (it must be earlier), and rejects a `?` term
  that is not in `@` or does not name a measurement.
- `estimate_num_layers` (the layer/volume estimate) includes the classical edges.

## 5. Scheduler (`src/scheduler.rs`, binary `puremagic`)

### Timing: `@` is a release time, not a routing constraint

Each lcycle the scheduler (1) takes the products whose parents are all done, (2) picks among them
with MINFIT given the magic states available, and (3) routes the picks with A*/Steiner-tree
packing. `@` edges act only in step 1:

- A product is ready when `remaining_parents` reaches 0.
- With `--reaction-latency L`, completing a measurement does not count down its `@` children
  immediately: they go into `delayed_unlocks` and are released `L` lcycles later
  (`release_delayed_unlocks`). Qubit-overlap edges are not delayed.
- The router never sees `@`, so it doesn't plan ahead for latency: there's no prioritization of
  measurements others wait on, and no reservation of paths for products about to be released.
  The constraint always holds (`plot_dynamic.py` checks start ≥ end + L + 1), but it is met
  greedily, not optimized.

### Conditions: `--cond-outcomes random|worst`, `--cond-p1 P` (7a1fdda)

PureMagic already reveals randomness lcycle by lcycle (cultivation times, T-gate failures);
measurement outcomes are handled the same way.

- `random` (default): when a conditional product becomes ready, each measurement its condition
  names gets a sampled true outcome (1 with probability `--cond-p1`, default 0.5), once per
  measurement. A product whose condition fails is **skipped** (`Scheduler::admit`):
  - It completes at once: no lcycle, no routing, no magic state, no T-failure draw, no S
    correction.
  - Its children are unlocked immediately and admitted in turn (a skipped product has no outcome,
    so nothing waits on it through the latency).
  - A skipped product still waits for its own `@` release: deciding to skip needs the outcome as
    much as running does.
  - On a hosted qubit it advances the qubit's chain without waking the patch.
- `worst`: every conditional product runs (the previous behaviour; an upper bound on the cost).
- Outcomes come from a separate random stream (seeded from `-r`), so the cultivation and T-failure
  draws, and the schedule of a circuit without conditions, are identical in both modes.
- The schedule header logs mode, outcomes and skipped products (§2). Stdout reports:
  `Conditional products: N, skipped K (k T) on M sampled outcome(s), p1=P`.
- Outcome model: `random` gives the cost under independent outcomes with P(1) = p1. That matches
  Exp 7's "mean of both branches" at p1 = 0.5, but it is an assumption: real outcome statistics
  depend on the quantum state. Report `worst` alongside it when a bound is needed.

Measured effect (dcsynth DC-opt/Mix circuits, Clifford tol 1e-4, reset mode flush, latency 5,
seeds 1–4, data-patch layout; patches are unchanged, so volume changes by the same percentage;
112 schedules, all pass the checker; `bench_results/cond_level1/`):

| Circuit | Conditional / all products | Skipped (mean) | lcycles worst → random |
|---|---|---|---|
| heis_20 DC-opt | 28,754 / 149,520 | 13,785 (2,532 T) | 53,861 → 48,750 (−9.5%) |
| heis_20 Mix | 28,699 / 147,608 | 14,058 (2,587 T) | 53,812 → 47,751 (−11.3%) |
| qaoa_12 DC-opt | 2,416 / 20,069 | 1,214 (224 T) | 8,556 → 7,781 (−9.1%) |
| grover_6 DC-opt | 9,017 / 52,116 | 4,200 (776 T) | 52,248 → 48,670 (−6.8%) |
| vqe_4 DC-opt | 4,484 / 27,879 | 1,723 (316 T) | 33,126 → 31,197 (−5.8%) |
| tfim_20 DC-opt | 6,627 / 49,884 | 2,595 (474 T) | 20,496 → 20,144 (−1.7%) |
| grover_5 DC-opt | 1,249 / 17,049 | 423 (75 T) | 17,418 → 17,272 (−0.8%) |
| qnn_6 DC-opt | 851 / 9,739 | 312 (58 T) | 7,906 → 7,904 (−0.0%) |

The saving tracks how much conditional work lies on the critical path. Skipping does not remove
the flushes that conditional gates cause at transpile time (§8), which is Level 2's target.

### Hosted ancillas: `--ancilla-qubits a,b,...` (2a0d3a7, e8659f3; needs `-u`)

- A hosted qubit's patch is an injection patch: **Data** while live, **Magic** (cultivating)
  while dormant. Only its X half-node cultivates; the Z half-node is route-only.
- Routing roots exclude hosted nodes, and precomputation runs while hosted nodes are still Data,
  so no precomputed tree depends on a hosted node.
- **Dormant → live**: when one of the qubit's products becomes ready (including after a delayed
  release); any cultivation in progress is discarded.
- **Live → dormant**: after the qubit's last product, or after a product that leaves the patch in
  a state a fresh patch can reproduce (a weight-1 X or Z `<M>`, or `<R>`) when the next product on
  the qubit is not ready yet. Pending S corrections are cleared.
- Stdout reports per qubit: `Hosted ancilla qN: live L / T lcycles, went live K time(s)`.

### Other scheduler details that matter for dynamic circuits

- **T failures** are a 50% coin flip per executed T. The S correction is tracked per qubit
  (`correction_power`) and applied lazily: it is merged into the next Clifford on that qubit, or
  later T bases are conjugated (`t_conjugated_bases`).
- **Termination**: the loop runs while products are pending, Cliffords are in flight, or delayed
  releases are outstanding.

## 6. dcsynth lowering (`data_processing/dcsynth_to_cliffordt.py`)

- Input: dcsynth's assembled Qiskit circuit (`measure`+`reset` per dynamic block, `if_else`
  corrections with `UnitaryGate` bodies, BQSKit `cx`/`u3` blocks).
- Output: the restricted QASM dialect: 1-bit registers, `if(m3==1 && m5==0) g q[i];`, one
  single-qubit gate per line.
- Corrections: for each correction it applies the cheapest branch's unitary unconditionally and
  conditions only the difference `D = U₁U₀†`. If `D` is a Pauli (within `--clifford-tol`), it is
  a free frame update; otherwise it is synthesized into a conditional Clifford+T sequence, each
  gate becoming its own conditional line with the same condition.
- Rotations are synthesized with cyclosynth (`--synth u3`, the default, falls back to zyz).
  Rotations within `--clifford-tol` of a Clifford are snapped to it.

## 7. Verification

| Tool | Checks |
|---|---|
| `data_processing/validate_dynamic.py` | **Exact oracle.** Simulates the input QASM and the `.trans` + sidecar exactly (statevector, branching at each measurement) in every reset mode, and compares the joint outcome distribution (TV < 1e-9). Also checks that `@` matches the sidecar's `needs` and `?` matches its `cond`. `--random N` fuzzes. |
| `data_processing/plot_dynamic.py` | **Schedule checker**, independent of the scheduler. Every product is scheduled once, unless skipped (CX 2 consecutive lcycles, S/SX 1–4). A skipped product's condition is false under the logged outcomes; in `random` mode every conditional product that ran has a true condition. Qubit order is checked against the previous product that ran on each qubit; classical order is start ≥ end + L + 1. No qubit is used twice in an lcycle. The sidecar is consistent. Also draws timeline and overview plots. |
| `cargo test` | Parser (`?`, `@`, `<R>`), transpiler (frame, conditional gates, reset modes, sidecar), scheduler (latency, hosting, dormancy, conditional skipping: both modes, p1 extremes, multi-term conditions, determinism, hosting with skipped products, rejection of bad conditions), integration (end to end on `tests/fixtures/dyn_*.qasm`) |
| `test_dcsynth_lowering.py` | The lowering end to end |

Status: `cargo test` passes in debug. Release has one failure, `node::add_nb_beyond_max`, which
already fails on `main`. The oracle passes on 60 random dynamic circuits and the 3 fixtures. The
end-to-end pipeline was verified exactly on grover_3, ae_3 and ae_4. On larger circuits, only the
schedule checker runs, and it has passed on every schedule.

## 8. Costs and known limitations

- **Flushes make Cliffords physical.** Every physical conditional gate flushes the whole Clifford
  queue on every qubit, and so does every reset with `--reset_mode flush|init`. Static circuits
  emit no physical Cliffords at all; the DC-opt circuits emit many (tol 1e-4):

  | Circuit | T | Conditional T | Unconditional S/SX | Conditional S/SX | CX |
  |---|---|---|---|---|---|
  | Orig (any) | – | 0 | 0 | 0 | 0 |
  | grover_5 DC-opt | 13,118 | 227 | 2,806 | 1,022 | 79 |
  | tfim_20 DC-opt | 19,283 | 1,211 | 25,047 | 5,416 | 90 |
  | heis_20 DC-opt | 41,398 | 5,287 | 84,382 | 23,467 | 175 |

  An S/SX occupies 3 lcycles. This is a likely contributor to DC volume growing faster than T
  count, but it has not been isolated.
- **Conditional T does not need the flush.** A conditional π/8 rotation on the conjugated Pauli
  commutes with the Clifford pushing like an unconditional one; only the condition gates it.
- **Timing assumptions the scheduler does not enforce.** A T whose sign depends on the frame, and
  each T's own injection correction, both need decoded outcomes before their S correction is
  resolved. Corrections are applied lazily, so this is usually hidden, but no latency is charged
  for it.
- **To verify.** A failed T on a multi-qubit Pauli is corrected with per-qubit S gates
  (`correction_power`); the exact correction is a π/4 rotation on the whole product.
- **Unsupported.** Measurement or reset inside `if`, conditional gates on three or more qubits,
  classical expressions, loops.
- **Modelling.** Every qubit, ancilla included, is a full logical patch, unless hosted. Unused
  declared qubits still get patches: strip them before scheduling (the benchmark harness does).
- **A pending T correction is ignored by measurements.** After a failed T, later T products on
  the qubit are re-conjugated (`apply_t_conjugation`), but a measurement is scheduled in its
  original basis and the pending correction carries on past it (it is cleared only when a hosted
  patch goes dormant). Physically the correction must come first or rotate the measurement
  basis, e.g. X becomes Y, which routes differently and affects the hosting release rule. Mid-circuit
  measurements make this far more common than in static circuits, where it only touches the final
  measurements.
- **Hosting relies on `--reset_mode flush|init`.** A hosted patch is released only after a
  weight-1 X/Z measurement or `<R>`; with `frame`, ancilla measurements are usually multi-qubit,
  so the patch stays live until its last product.
- **Estimates ignore skipping.** The printed volume estimate and "normalized scheduling
  efficiency" count every product, including skipped conditional ones.
- **No circuit-level error budget.** Synthesis error (ε per rotation) and Clifford snapping (up
  to `--clifford-tol` per rotation or correction) are not summed into a reported total.
- **Exactness at scale.** The oracle is exponential in width; above ~14 qubits only the schedule
  checker (ordering and resources, not outcomes) runs.

## 9. Planned

- **Level 2: absorb conditional Cliffords at run time.** The transpiler would emit a conditional
  Clifford as a classical event. Once the outcome is known, the scheduler would conjugate later
  products by it, using a run-time Clifford tableau (generalizing `correction_power`). This
  removes the flushes. Cost: product supports change at release, so precomputed trees must be
  recomputed, and the oracle must model the run-time conjugation.
- **Cheaper variants first**: no flush for conditional T; flush only the conditional qubit's
  backward light cone; one flush and frame payment per run of conditional gates with the same
  condition.
- **`--cond-outcomes trace FILE`**: real outcomes from simulation or hardware (the online
  executor case).
- **Latency-aware selection**: prefer measurements that conditional products wait on.
- **Repeat-until-success (RUS) circuits**: see the design sketch below.

### Repeat-until-success: what supporting it would take

An RUS gadget runs a block, measures, and repeats the block until the measurement reports
success. That needs three things the stack does not have: loops, measurements under a
condition, and per-measurement success probabilities.

1. **Input and lowering.** Accept a loop (Qiskit `WhileLoopOp`, or an OpenQASM 3 `while`) in
   `dcsynth_to_cliffordt.py`, and lower it by **bounded unrolling**: K copies of the body, where
   copy i runs only if attempts 1..i-1 all failed. Each copy's operations get the AND condition
   (`?m1=0,m2=0,..`), which `?` already expresses. K is chosen so that P(not done after K) is
   below the error budget. True unbounded loops would need the scheduler to instantiate
   iterations itself (see 5).
2. **Measurement and reset under a condition** (the transpiler rejects them today).
   - The value a later `if` reads becomes a multiplexer: the new outcome if the measurement ran,
     else the bit's old value. Express it as derived outcomes (ANDs of terms), the way
     `derived_outcome` handles multi-bit conditions.
   - A conditional reset puts a multiplexed term into the frame. Avoid that by requiring
     `flush`/`init` resets inside loop bodies, so each attempt starts frame-free.
   - The tableau must be the same on both branches. Flush at every iteration boundary, so a
     skipped iteration leaves the same Clifford frame as an executed one.
3. **Scheduler.**
   - Allow conditional `<M>`/`<R>`. A skipped measurement has no outcome, so products listing
     it in `@` must take the outcome of whichever attempt actually ran: release them when the
     loop's last executed measurement is decoded.
   - Sample outcomes with per-measurement success probabilities rather than the global
     `--cond-p1`: an annotation in `.trans` (e.g. `<M>%0.75`) or the sidecar.
   - Report expected cost over seeds; the number of iterations is now random.
4. **Formats and checks.** Sidecar records for conditional measurements and multiplexed
   outcomes. The checker must accept skipped measurements. The oracle must branch over
   iterations, which is feasible for small K.
5. **Unbounded / online alternative.** Keep the body as a template and let the scheduler
   re-instantiate it on failure. This needs dynamic product ids, Steiner trees and terminals
   computed at run time, and a hosting chain that grows. It is closer to a real online executor,
   but a much larger change than unrolling.
