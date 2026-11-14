# Mid-circuit measurement in PureMagic: what changed, and how it works

This is the specification of everything the MCM work adds to PureMagic: the model, the file
formats, each component's changes, the timing model, what is verified, what it costs and what
is not done yet. `docs/dynamic-circuits.md` is the user guide (how to run it); this document is
the reference for how it works. Branch: `mcm-support`, which now contains everything below
(`ancilla-patch`, `mcm-ancilla`, `t-autocorrect-litinski` and `t-correction-frame` were merged
into it).

### Where everything is

In this repository (`PureMagic/`, branch `mcm-support`):

| File | What it holds |
|---|---|
| `docs/mcm-changes.md` | This file: change log (commit table below), MCM specification, scheduler notes, known limitations, plans |
| `docs/t-gate-corrections.md` | Failed-T corrections: the two bugs in the old model, the six `--t-correction` models, their comparisons, approximations, choosing `--reaction-latency` |
| `docs/dynamic-circuits.md` | User guide for dynamic circuits (how to run them) |
| `README.md` | Every command-line flag |
| `notebooks/dynamic_circuit_tutorial.ipynb` | Walkthrough of a small dynamic circuit end to end |
| `src/frame.rs`, `src/scheduler.rs` | Clifford frame; the T-correction models live in the scheduler |

Next to it, in `../bench_results/` (benchmark data, not in the repository):

| Path | What it holds |
|---|---|
| `REPORT.md` | All benchmark results and findings, including the corrected-T rerun and the frame/hybrid comparisons |
| `ERROR_LOG.md` | Every bug found, with evidence and status |
| `scripts/CHANGES.md` | Change log of the benchmark harness; the scripts are in `scripts/` |
| `t_correction_bug/` | Reproduction of the T-correction bug, exact checker (`check.py`), frame/hybrid fuzzer (`fuzz_frame.py`) |
| `t_correction_cmp/` | Model comparisons: `results_*.csv`, `summary_*.txt`, `frame_detail.txt`, frozen binaries |
| `out_*_tfix/` | The report's result sets rescheduled with the corrected T model |
| `patches/README.md` | How the branches were consolidated into `mcm-support` |
| `notes/t-correction-proof.html` | Typeset proof that the old T correction is wrong (also published as a private page) |

| Commit | Branch | What |
|---|---|---|
| 6dd5b41, 80535d0 | mcm-support | Dynamic-circuit transpilation, `.trans` `@` dependencies, sidecar, reaction latency, validator, checker, dcsynth lowering |
| 7613cbc | mcm-support | Lowering synthesizes rotations with cyclosynth |
| 16ed103 | mcm-support | Checker accepts the scheduler's own S corrections |
| 2a0d3a7 | ancilla-patch | `--ancilla-qubits`: ancillas hosted on injection patches |
| 3481138, e8659f3 | mcm-ancilla | Merge; hosted patches sleep between uses |
| 7a1fdda | mcm-ancilla | Conditions in `.trans` (`?`); scheduler skips products whose condition fails (`--cond-outcomes`) |
| d697a79 | mcm-ancilla | Conditional CX/CZ/SWAP; flushed CZ/SWAP no longer dropped |
| 351d2c8 | mcm-ancilla | Measurements honor a pending T-failure correction |
| 53ab131 | mcm-ancilla | Estimates leave skipped conditional products out |
| 86e858c | mcm-ancilla | Exact check skipped above 10 qubits |
| bcc0cfc | mcm-ancilla | Lowering: error ledger and `--error-budget` |
| cc57acb | mcm-ancilla | An S/SX holds its nodes for all 3 lcycles (schedules no longer depend on `--plot paths`) |
| d15d91a | mcm-ancilla | `notebooks/dynamic_circuit_tutorial.ipynb` |
| d94a740 | mcm-ancilla | A T whose sign depends on measurement outcomes waits for them (`@` needs) |
| 2f84ec0, a673091 | t-autocorrect-litinski | Failed T injections auto-corrected (Litinski), default `--t-correction litinski`; see `docs/t-gate-corrections.md` |
| 563a4ef | t-autocorrect-litinski | `--t-correction inplace-y` (Gidney in-place Y) and `--y-meas-lcycles` |
| a76ea98 | mcm-support | Merge of t-autocorrect-litinski: `mcm-support` now holds all of the above |
| e563b71, 4c53671 | t-correction-frame | `--t-correction frame`: failed T corrections folded into an exact Clifford frame (opt-in; loses to `litinski`, see §5 and `docs/t-gate-corrections.md` §3a) |
| 5634c69 | t-correction-frame | `--t-correction hybrid`: local frame for one-qubit T gates, cleared by rotations before CX; helper for multi-qubit T gates (`docs/t-gate-corrections.md` §3b) |
| a91db93 | t-correction-fix | `litinski` split into `litinski-y` (default; the old `litinski`: never waste a ready state, fall back to an in-place X/Y measurement) and `litinski-ready` (a ready magic state may be wasted as the helper; +0.8% over the 24 circuits); `docs/t-gate-corrections.md` §2.1–2.2 |

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
   frame only changes sign, which is recorded as `frame_deps`. For a T, that sign decides T vs T†,
   so it also waits for those outcomes (`@`).
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
  - T/M/R: `sign`, `frame_deps` (outcome ids whose XOR flips the sign). A T with `frame_deps`
    also has `needs` (it waits for those measurements: its sign decides T vs T†).
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

- **T failures** are a 50% coin flip per executed T. By default (`--t-correction litinski-y`, the
  former `litinski`) the correction is applied on a borrowed helper patch, so nothing is owed on
  the data; the helper is held until the correction is decided (reaction latency, chained through
  earlier anticommuting T gates). When no bus or cultivating neighbour is free, `litinski-y`
  waits and then measures the magic patch in place in X or Y, while `litinski-ready` wastes a
  ready magic state as the helper. See `docs/t-gate-corrections.md`. The two bullets below describe only
  `--t-correction legacy`, which is kept to compare against.
- **Exact-frame model** (`--t-correction frame`, e563b71, opt-in). No correction is ever applied.
  The scheduler keeps an exact Clifford frame W (`src/frame.rs`, a GF(2) tableau without signs)
  with hardware state = W·(ideal state):
  - **Updates.** A failed T about P sets W ← W·R_P(−π/4). A physical S/SX/CX line runs as written
    and sets W ← G·W·G†.
  - **Rewriting.** Every T and measurement about Q runs about W·Q·W†, whose support can be larger
    or smaller than Q's. It is routed on the rewritten terminals, and a multi-qubit measurement
    leaves its precomputed tree.
  - **Waiting.** A T or measurement whose Pauli anticommutes with a T injected in the last L lcycles
    waits until that outcome is decoded, because only then is its rewritten Pauli known. This is
    the reaction limit, paid on the data path instead of on a helper.
  - **Not supported with `--ancilla-qubits`** (a hosted patch could be released while the frame
    still entangles it). `<R>` stays correct without a physical reset: after the preceding
    measurement it is a Pauli, which the frame absorbs.
  - **Verified** against exact unitaries on 900 random circuits, including that no product ever ran
    on a Pauli that depended on an undecoded outcome.
  - **Cost** (24 circuits, ancillas as data patches): geomean +9.6% / +23.2% lcycles over legacy at
    L = 0 / 1, against +4.1% / +6.7% for `litinski`. Physical Cliffords spread the frame (tfim_20
    DC-opt: mean T/M weight 1.24 → 7.31). Long runs of non-commuting T gates stall the data from
    L = 1. Because DC circuits carry the physical Cliffords, it penalizes DC more than Orig. It
    wins only on circuits without physical Cliffords at low latency (9 of 24 at L = 0).
- **Hybrid model** (`--t-correction hybrid`, 5634c69, opt-in). Why it exists: the exact frame
  spreads through every physical CX (W ← CX·W·CX† turns a one-qubit S into an entangling π/4
  rotation), so at `-m 1` a one-qubit T ended up running on 45–90 qubits. The hybrid keeps the
  frame *local*, one one-qubit Clifford per qubit (one of 24, up to Paulis), so no product grows.
  What it does, step by step:
  1. **A failed one-qubit T** (say about Z₃) is folded into the frame: nothing physical happens
     and no helper is used. The frame now records "S about Z pending on qubit 3".
  2. **A failed multi-qubit T** (say about Z₀Z₁) would put an entangling correction in the frame,
     so it is auto-corrected on a borrowed helper patch exactly as in `litinski-y` instead.
  3. **Every T and measurement about Q** runs about W·Q·W†. A local frame only changes letters:
     with S_Z pending on qubit 3, a T about X₃ runs as a T about Y₃, on the same patch (a Y
     uses both of the qubit's data nodes).
  4. **Physical S and SX lines** run as written and the frame is conjugated by them; it stays
     local.
  5. **Physical CX lines** are where a frame would spread. A CX commutes with the frame (up to
     Paulis) when the control's pending Clifford fixes Z and the target's fixes X; then it just
     runs. Otherwise the scheduler first applies, physically, one π/4 one-qubit rotation (when the
     pending Clifford swaps two axes) or two (when it cycles X, Y, Z) on that qubit, about X, Y or
     Z, which brings its frame back to the identity. These are scheduler-added S-type products
     (an S, an SX, or a rotation about Y on both data nodes, routed like a Y measurement), and
     the CX runs after them.
  6. **Waiting.** A one-qubit T's outcome is decoded L lcycles after injection
     (`--reaction-latency`). Until then, a T or measurement that anticommutes with it waits (its
     letters depend on the outcome), and so does a CX on that qubit (the outcome decides whether a
     clearing rotation is needed). Multi-qubit T gates only hold their helper, as in `litinski`.

  Verified against exact unitaries on 960 random circuits (including that no CX ever runs while
  the frame does not commute with it, whatever an undecoded outcome turns out to be). Cost against
  the old model, geomean lcycles: at L = 0 it ties Litinski (+4.1% vs +4.1%; +2.7% vs +3.2% at
  `-m 1`) and runs as fast, with no helper patch for one-qubit T gates. From L = 1 it loses (+14.1%
  vs +6.7%; +45.9% vs +4.8% on the large `-m 1` benchmarks): synthesized one-qubit rotations are
  runs of T gates about alternating axes, and each one waits L lcycles for the previous outcome on
  the data path, where Litinski only holds helpers. Not supported with `--ancilla-qubits`.
- **Legacy model.** The S correction is tracked per qubit (`correction_power`) and applied
  lazily: it is merged into the next Clifford on that qubit, or later T bases are conjugated
  (`t_conjugated_bases`).
- **Measurements under a pending correction** (351d2c8, legacy model) are conjugated the same way: measuring P
  before a pending S is measuring S†PS, so with an odd correction an X measurement is a Y
  measurement (both half-nodes). The correction stays pending afterwards. A multi-qubit
  measurement whose basis changes is routed live on its conjugated terminals instead of its
  precomputed tree. The hosting release rule uses the effective basis. A re-initialization `<R>`
  voids the qubit's pending correction.
- **Estimates** (parallelism, volume, normalized efficiency, S-correction ratio) are computed on
  `Circuit::without(skipped)`, the circuit that actually ran (53ab131).
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
- Every approximation is recorded (a ledger) and the realized error is reported; with
  `--error-budget`, ε and the snap threshold are chosen to fit a total budget (§6a).

## 6a. Error budget (bcc0cfc)

**Unit.** Every error is measured as e = ½‖Φ − 𝒱‖◇: the worst-case trace distance between the
outputs of the implemented channel Φ and the ideal unitary V, over all inputs including
entangled ones. **It adds up**: the diamond norm is subadditive under composition and unchanged
by tensoring with an identity, and a classically controlled choice is bounded by its worst
branch. So for the whole circuit,

    e_total ≤ Σ_blocks e_block + Σ_rotations e_rot + Σ_snaps e_snap.

**Each tool in that unit.**

| Source | Tool's own metric | In e |
|---|---|---|
| cyclosynth rotation | phase-minimized operator norm `min_φ ‖U − e^{iφ}V‖`, ≤ ε | e ≤ that norm (for unitaries e = sin(θ/2) ≤ 2 sin(θ/4) = the norm) |
| Clifford snap (lowering) | same norm, < `--clifford-tol` | same |
| BQSKit block (`bq`) | HS distance `√(1 − |Tr U†V|²/D²)` ≤ its threshold | computed exactly from the circuits: e ≤ `min_φ ‖U − e^{iφ}V‖`; a priori ≤ √(2D)·HS distance |
| dcsynth dynamic block (`dc`) | `--epsilon` bounds **C = 1 − Σⱼ|Tr V†Kⱼ|²/D²**, an infidelity (the square of a distance), not a distance | computed from the Kraus operators Kⱼ = Bⱼ⟨j|U_main|0⟩: e ≤ δ + δ²/2, δ = √‖Σⱼ Eⱼ†Eⱼ‖ with Eⱼ = Kⱼ − cⱼV. In practice e ≈ 1.15–1.5·√C |

So a library compiled at dcsynth ε = 1e-6 means up to ~1e-3 *per block* in e, not 1e-6. Measured
on qnn_4/grover_4 (fresh libraries), the realized block errors are 5e-6 to 1.4e-4, against 1e-8
per cyclosynth rotation.

**Where block errors come from.** `bench_results/scripts/block_errors.py` computes e for every
block an assembly uses, from the actual circuits. Stage A writes the sum as `block_error` in
`variants.json`.

**Distributing a budget E in the lowering** (`--error-budget E --block-error B`):

1. The blocks are already compiled, so their error B is fixed; the lowering gets R = E − B (an
   error if R ≤ 0: the library must be compiled more accurately).
2. **Equal ε for every rotation.** A rotation to accuracy ε costs about c·log₂(1/ε) T gates.
   Minimizing Σ c·log₂(1/εᵢ) subject to Σ εᵢ = const gives equal marginal costs c/(εᵢ ln 2), i.e.
   equal εᵢ.
3. **Snap threshold τ = ε(1 + ln(1/ε)).** Snapping a target at distance d saves about
   c·log₂(1/ε) T and spends d − ε more error. Absorbing that error by tightening the rotations
   costs about c(d − ε)/(ε ln 2) T. So snapping pays exactly when d < ε(1 + ln(1/ε)), for any c.
   That is about 12ε at ε = 1e-5, so snapping should be far more aggressive than ε.
4. A dry pass collects each target's distance to the nearest Clifford. ε is the largest value
   with Σ_{d<τ(ε)} d + ε·#{d ≥ τ(ε)} ≤ R (bisection; repeated while τ moves the targets). The
   circuit is lowered once with (ε, τ), and the realized total is checked against E.

On qnn_4 DC-opt (B = 3.8e-5):

| Setting | ε | T count | Realized e |
|---|---|---|---|
| fixed ε 1e-8, tol 1e-4 | 1e-8 | 3,838 | 4.7e-5 + B (snaps alone 4.6e-5) |
| `--error-budget 1e-3` | 1.8e-5 | 2,263 | 8.0e-4 |
| `--error-budget 1e-2` | 2.0e-4 | 1,731 | 7.2e-3 |

**Not covered.** The budget for *compiling* the blocks (dcsynth's ε) is chosen before this step;
splitting E between block compilation and lowering needs dcsynth's depth-vs-ε trade-off.
Exp 7's BQSKit-FT path (`full_rz_compile`) uses HS distances and is not part of this pipeline.

## 7. Verification

| Tool | Checks |
|---|---|
| `data_processing/validate_dynamic.py` | **Exact oracle.** Simulates the input QASM and the `.trans` + sidecar exactly (statevector, branching at each measurement) in every reset mode, and compares the joint outcome distribution (TV < 1e-9). Also checks that `@` matches the sidecar's `needs` and `?` matches its `cond`. `--random N` fuzzes. |
| `data_processing/plot_dynamic.py` | **Schedule checker**, independent of the scheduler. Every product is scheduled once, unless skipped (CX 2 consecutive lcycles, S/SX 1–4). A skipped product's condition is false under the logged outcomes; in `random` mode every conditional product that ran has a true condition. Qubit order is checked against the previous product that ran on each qubit; classical order is start ≥ end + L + 1. No qubit is used twice in an lcycle. The sidecar is consistent. Also draws timeline and overview plots. |
| `cargo test` | Parser (`?`, `@`, `<R>`), transpiler (frame, conditional gates, reset modes, sidecar), scheduler (latency, hosting, dormancy, conditional skipping: both modes, p1 extremes, multi-term conditions, determinism, hosting with skipped products, rejection of bad conditions), integration (end to end on `tests/fixtures/dyn_*.qasm`) |
| `test_dcsynth_lowering.py` | The lowering end to end, including the error budget (fits E, more budget never costs more T, ε is the largest that fits) |

Status: `cargo test` passes in debug. Release has one failure, `node::add_nb_beyond_max`, which
already fails on `main`. The oracle passes on 460 random dynamic circuits (with CZ, SWAP and
conditional two-qubit gates) and the fixtures. It is run only up to **10 qubits** (86e858c;
stage A's sampled fidelity has the same cap); wider circuits get the schedule checker only, which
checks ordering and resources, not outcomes. The end-to-end pipeline was verified exactly on
grover_3, ae_3 and ae_4.

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
  count, but it has not been isolated. In the tutorial's 3-qubit example, the conditional gates'
  flushes create about 175 of 203 unconditional S/SX gates (with the conditional lines removed,
  27 remain, all from the reset flush), and they form a serial tail of about 500 lcycles.
- **Fixed in cc57acb: S/SX nodes were held for 1 of their 3 lcycles unless paths were plotted.**
  Schedules of circuits with physical S/SX were therefore optimistic: DC-opt lcycles rise by
  +1.6% to +14.8% with the fix (tfim_20 +15%, grover_5 and qaoa_12 +3–4%). Static circuits emit
  no physical S/SX and were unaffected.
- **Conditional T does not need the flush.** A conditional π/8 rotation on the conjugated Pauli
  commutes with the Clifford pushing like an unconditional one; only the condition gates it.
- **A T whose sign depends on outcomes waits for them.** If a T's Pauli anticommutes with a
  Pauli the frame owes, it is T or T† depending on those outcomes; the transpiler gives it those
  measurements as `needs` (an `@` edge, with reaction latency). A measurement's sign only
  relabels its outcome and gets no edge.
- **Fixed on t-autocorrect-litinski: T-failure corrections.** The two items below describe the
  legacy model (`--t-correction legacy`). The default now auto-corrects each T on an ancilla and
  charges the reaction latency for deciding it; see `docs/t-gate-corrections.md`.
- **Timing assumption the legacy model does not enforce.** Each T's own injection correction needs
  the injection outcome decoded before it is applied. Corrections are applied lazily, so this is
  usually hidden, but no latency is charged for it.
- **Multi-qubit T failures are modelled per qubit (not exact; legacy model).** A failed π/8 rotation on
  P = P₁⊗…⊗P_w needs the π/4 rotation e^{−iπ/4·P}, a single weight-w Clifford. PureMagic
  instead raises each qubit's `correction_power` and later applies, or conjugates by, single-qubit
  S gates. The two differ:
  - **Cost.** One multi-patch π/4 product vs w separate 3-lcycle S gates.
  - **Effect on later products.** e^{−iπ/4·P} maps an anticommuting Q to ∝ PQ, which can change
    the support (P = Z₁Z₂, Q = X₁ → Y₁Z₂). The per-qubit swap gives Y₁. Later products may
    therefore be routed on the wrong qubits.

  This predates the MCM work and affects static circuits too: 61% of grover_5 DC-opt's T
  products and 90% of its Orig's have weight > 1. The standard fix is Litinski's auto-corrected
  π/8 rotation. The P⊗Z injection measurement is followed by an X or Y measurement of the magic
  ancilla, chosen by the outcome, so the data never needs a correction; the cost is an ancilla
  measurement that waits for decoding, which is where reaction latency belongs.
- **Unsupported.** Measurement or reset inside `if`, conditional gates on three or more qubits,
  classical expressions, loops.
- **Modelling.** Every qubit, ancilla included, is a full logical patch, unless hosted. Unused
  declared qubits still get patches: strip them before scheduling (the benchmark harness does;
  `docs/dynamic-circuits.md`, "Pipeline", shows how).
- **Hosting relies on `--reset_mode flush|init`.** A hosted patch is released only after a
  weight-1 X/Z measurement or `<R>`; with `frame`, ancilla measurements are usually multi-qubit,
  so the patch stays live until its last product.
- **Exactness at scale.** The oracle is exponential in width, so above 10 qubits only the schedule
  checker runs (ordering and resources, not outcomes).

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
