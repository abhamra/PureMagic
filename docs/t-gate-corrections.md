# Correcting failed T injections

A T gate in the Pauli-based circuit is a π/8 rotation R_P(π/8) = e^{−iπ/8·P} about a Pauli
product P. It is injected by measuring P⊗Z_m jointly with a magic-state patch m. Half the time
the rotation comes out as R_P(−π/8), and the circuit then owes the Clifford correction
R_P(π/4). This document covers how PureMagic used to account for that correction, why that was
wrong, and the models that replace it. Litinski's auto-corrected rotation comes in two variants
that differ only in which patch may be the helper:

- `litinski-ready` (default): when no bus or cultivating neighbour is free, a ready magic state
  is borrowed as the helper and its state is wasted (§2.1);
- `litinski-y`: a ready state is never borrowed; an injection without a helper waits, then has
  its magic patch measured in place in X or Y, Gidney's technique (§2.2). This is what
  `litinski` meant before the split, and every result below labelled `litinski` is this variant.

| Flag | Values | Default |
|---|---|---|
| `--t-correction` | `litinski-ready`, `litinski-y`, `inplace-y`, `frame`, `hybrid`, `legacy` | `litinski-ready` |
| `--y-meas-lcycles` | extra lcycles for an in-place Y measurement | 1 |
| `--reaction-latency` | lcycles to decode an outcome and act on it (also used for `@` edges) | 0 |

`-F` (no T failures) skips all of this: nothing is ever owed.

## 1. The old model and why it is wrong (`--t-correction legacy`)

From 76a66fa (July 2026) PureMagic recorded a failed T as a per-qubit counter: for each qubit q
of P, `correction_power[q] += 1 (mod 4)` and `correction_basis[q]` = the qubit's axis. Later T
gates and measurements were relabelled qubit by qubit (X↔Y about Z, and so on), power 2 was
treated as a free Pauli, and a pending odd power was paid as a one-qubit S gate before the next
Clifford on that qubit. The paper (arXiv 2512.06484, §III-E) describes the same approach. It is
wrong in two independent ways.

1. **Multi-qubit P.** R_{Z⊗Z}(π/4) = e^{iπ/4}·CZ·(S⊗S) is entangling, so no product of one-qubit
   corrections equals it. Deferred past a later operation Q that anticommutes with P, it turns Q
   into ∝ QP, which can add qubits: P = Z₀Z₁ and Q = X₀ give Y₀Z₁, a weight-2 operation. The
   per-qubit relabelling gives Y₀, on the wrong qubits.
2. **Two failures about different axes on one qubit**, even for weight-1 T gates. The counter adds
   powers and overwrites the axis. S_Z followed by S_{−Y} is recorded as "power 2" and dropped as a
   Pauli, but it is a non-Pauli Clifford that cycles the axes (X→−Y, Z→X). A (power, axis)
   record can represent only 10 of the 24 one-qubit Cliffords.

Smallest example, one qubit, both T gates failing: `+Z<T> +X<T> +Z<S> +X<M>`. Exactly, the
operations run about Z, −Y, X, and the measurement is of −Y. The legacy model gets the first two
right up to sign, but then runs the S about Z and measures X. The two failed corrections combine
into a non-Pauli Clifford, and no correction gate is emitted for it. The
model also charged no decoding time before a correction is decided.

`legacy` is kept only for comparison. It reproduces the old schedules line for line.

## 2. Auto-corrected π/8 rotation (`litinski-ready`, default, and `litinski-y`)

Litinski, *A Game of Surface Codes* (Quantum 3, 128, 2019), auto-corrected π/8 rotations. With a
|0⟩ helper patch h next to the magic patch m:

1. measure P⊗Z_m (the injection) and Z_m⊗Y_h in the same lcycle, then measure m in X. The magic
   patch is free again from the next lcycle;
2. once the outcome is decoded, measure h in **X** to apply the correction R_P(π/4), or in **Z** not
   to.

The correction is a choice of measurement basis on an ancilla. Every outcome only adds Paulis to
the frame, so **no later product is ever changed and no S gate is emitted.** Only the helper
waits.

What the scheduler does in both variants:

- **Helper.** A free neighbour of the magic patch, borrowed in the injection lcycle. A bus
  patch comes first, then the cultivating magic patch with the least progress (its cultivation
  restarts). Never a data patch or a hosted patch. With magic routing (`-u`) there are no bus
  patches, so the helper is always a magic patch. No patch is dedicated to helpers, so injection
  stays single-patch.
- **The helper need not be adjacent in principle.** Z_m⊗Y_h could be routed through free tiles to
  a patch farther away. In this tile model that never helps: any free tile a route could start
  through can host the helper itself, so the nearest free candidate is always adjacent. A routed
  search was implemented (on `t-correction-fix`) and found no non-adjacent helper on any
  benchmark.
- **Hold.** The helper stays occupied until the correction is decided (the decision lcycle below).
  Data-side products are never delayed: the helper's measurement acts on a different patch, so
  it commutes with everything that follows on the data.

The variants differ when no bus or cultivating neighbour is free. The free neighbours left are
then usually ready magic states (on most failed searches in the PureMagic benchmarks).

### 2.1 `--t-correction litinski-ready` (default): waste a ready state

A ready magic state is borrowed as the helper, last in the order above; its state is lost and the
patch starts cultivating again. Fewer injections wait for a helper, and the in-place fallback of
§2.2 is kept only as a safety net (a magic patch with no free neighbour at all for 2 lcycles).

Measured against `litinski-y` (geomean lcycles; seeds 0–1):

| Set | L = 0 | L = 1 | Better / worse on |
|---|---|---|---|
| 5 PureMagic benchmarks at `-m 0` and `-m 1` (10 circuits, no hosting) | −0.3 points vs the old model (+2.7% vs +3.0%) | −0.3 points (+5.1% vs +5.4%) | 6 better at both L, 3 worse, 1 mixed |
| 7 dcsynth benchmarks, Orig and DC-opt (14 circuits, ancillas hosted) | +1.6% | +1.6% | 3–4 better, 10–11 worse |

On the PureMagic benchmarks, waits for a helper drop from 28.3% to 6.1% of injections, the
in-place fallback (0.47%) disappears, and 16.5% of helpers are wasted ready states. On the
dcsynth benchmarks the fallback also disappears (6,828 injections over the runs, to 0) and waits
halve (421,224 to 122,409), but the 214,889 wasted ready states cost more: these circuits are
T-dense and limited by magic-state supply. `litinski-ready` is the default; for the dcsynth
comparisons in this repository, `litinski-y` is the cheaper choice and matches the results
reported below.

### 2.2 `--t-correction litinski-y`: never waste a ready state; in-place Y fallback

A ready magic state is never a helper. An injection with no other free neighbour waits for one,
holding its magic patch. After 2 lcycles, or at once if no neighbour could ever be a helper, the
magic patch is measured in place instead (section 3, Gidney): it is held until the decision and
measured in X (no correction) or Y (applies it). Waiting longer could deadlock two magic patches
that each occupy the other's only free neighbour: without the 2-lcycle limit, 4 of 20 PureMagic
benchmark runs took 1–4 million lcycles instead of 1,500–14,000.

### When is the correction decided?

A T's own outcome is known L = `--reaction-latency` lcycles after injection. Whether a correction
is owed also depends on the frame: if an earlier T's Pauli byproduct anticommutes with this P,
it flips this rotation's sign. That byproduct comes from the earlier T's final ancilla
measurement, made at that T's decision and decoded L later. So decisions chain:

    decision(T_k) = max( inject(T_k) + L,
                         max over earlier anticommuting T_j sharing a qubit: decision(T_j) + L )

(with the in-place Y measurement time added to decision(T_j) when T_j's patch is measured in Y).
With L = 0 everything is decided at once. A run of anticommuting T gates, such as a synthesized
one-qubit rotation, therefore needs about L lcycles per T. This is Litinski's reaction limit; the
old model did not charge it.

## 3. In-place Y measurement (`--t-correction inplace-y`)

Gidney, *Inplace Access to the Surface Code Y Basis* (Quantum 8, 1310, 2024). No helper. After
P⊗Z_m, the magic patch waits for the decision and is then measured in **X** (no correction) or
**Y** (applies it: S_Z†·X·S_Z = −Y). The Y measurement takes ⌊d/2⌋+2 rounds, modelled as
`--y-meas-lcycles` (default 1, rounded up to whole lcycles) of extra hold, and its frame bit
arrives that much later, which moves the decisions of later anticommuting T gates too.

The trade-off: it needs no helper, but the magic patch cannot cultivate while it waits, and Y
measurements lengthen the decision chain.

## 3a. Exact Clifford frame (`--t-correction frame`)

Never apply the correction at all. R_P(π/4) is a Clifford, so it can be folded into a frame W
(`src/frame.rs`): the hardware state is W·(ideal state), and every later T or measurement about
Q runs about W·Q·W†. A failed T about P sets W ← W·R_P(−π/4). Physical S, SX and CX lines in the
`.trans` run unchanged and set W ← G·W·G†. The frame merges into measurements for free (they
stay Pauli measurements), but not into T gates:

- **Supports change.** W·Q·W† can act on more qubits than Q, or on fewer: after a failed T about
  Z₀Z₁, X₀ runs as Y₀Z₁. A rewritten product gets new terminals, and a multi-qubit measurement
  leaves its precomputed tree.
- **Data-side waits.** Whether a later Q is rewritten depends on whether the earlier T failed, and
  that is known only L lcycles after injection. So a T or measurement that anticommutes with a T
  injected in the last L lcycles waits. (With Litinski only the helper waits.)
- **No helper and no hold.** At L = 0 nothing waits at all.
- **No hosted ancillas.** A hosted patch is released after a single-patch X/Z measurement as
  written; under the frame that measurement can run on several patches, leaving the ancilla
  entangled, and products of other qubits can be rewritten onto it. Supporting this needs the
  hosting bookkeeping to follow the executed supports; `frame` rejects `--ancilla-qubits`.
- **Resets** (`<R>`) need no physical re-initialization: the measurement before them fixes the
  ideal qubit, so the reset is a Pauli, which the frame absorbs.

The frame keeps signs out (only supports and letters affect scheduling). It is a GF(2) tableau on
64-qubit words, so rewriting a product costs O(weight · n/64) word operations.

## 3b. Local frame with flushes (`--t-correction hybrid`)

The exact frame grows products because physical CX lines carry it onto more qubits. The hybrid
keeps the frame *local*, one one-qubit Clifford per qubit, so a product keeps its support and only
its letters change:

- **One-qubit T** that fails: folded into the frame (no helper, no hold).
- **Multi-qubit T** that fails: auto-corrected on a helper as in `litinski-y` (never a ready state). Its correction is
  entangling, so it never enters the frame.
- **Physical S/SX:** pass through the frame (it stays local).
- **Physical CX:** runs only if the control's frame fixes Z and the target's frame fixes X (then
  the CX commutes with the frame up to Paulis). Otherwise one or two π/4 one-qubit rotations about
  X, Y or Z are applied physically first, bringing that qubit's frame to the identity. A rotation
  about Y uses both of the qubit's data nodes, as a Y measurement does.
- **Waits:** a T or measurement that anticommutes with a one-qubit T injected in the last L
  lcycles waits for that outcome, as in `frame`. A CX also waits for undecoded one-qubit T
  outcomes on its qubits, because they decide whether a rotation is needed.

Like `frame`, it does not support `--ancilla-qubits` yet. With a local frame this would only need
the release rule to use the executed letter, and the frame cleared when a patch is re-prepared.

## 3c. Choosing `--reaction-latency`

L is the time to decode an outcome and act on it, in logical cycles, rounded up. A logical cycle
is about d syndrome-extraction rounds, so

    L = ceil( t_reaction / (d * t_round) )

| Setting | Round time | Reaction time | L at d = 25 | Source |
|---|---|---|---|---|
| Superconducting, fault-tolerance assumption | 1 µs | 10 µs | 1 (0.4, rounded up) | Gidney & Ekerå 2019 (RSA-2048 estimate); reused in later resource estimates |
| Superconducting, measured today | 1.1 µs | 63 µs mean decoder latency (d = 5, real time) | 3 (2.3); 12 at d = 5 | Google, "Quantum error correction below the surface code threshold" (Nature 2025) |
| Trapped ions, neutral atoms | ~100 µs to ~1 ms (mostly projections) | small next to a logical cycle | 0–1 | e.g. trapped-ion lattice-surgery and neutral-atom architecture estimates |

Litinski's *Game of Surface Codes* assumes the 1 µs code cycle; the 10 µs reaction time is Gidney
and Ekerå's. Google's 63 µs is a mean measured at d = 5; a larger-distance decoder would be slower,
but the logical cycle grows with d too. Use **L = 1** as the main setting (the standard
fault-tolerance assumption) and L = 2–5 as a sensitivity range for today's decoders.

## 4. Approximations

- **Correction needed = injection failed.** The true condition is (injection outcome) XOR (frame
  parity), which is also a fair coin. The scheduler uses the failure coin. This matters only for
  `inplace-y`, where it decides which patches pay the Y time.
- **Z_m⊗Y_h takes one lcycle**, in parallel with P⊗Z_m. Helper X/Z measurements and the magic X
  measurement are instantaneous (transversal).
- **The chain runs only through T gates.** Frame bits from mid-circuit measurements are handled by
  the transpiler, which gives such a T an `@` dependency, so the T itself waits (d94a740).
- **No decoder backlog**: every outcome takes exactly L lcycles.

## 5. Results

The circuits are 24 in total, all run with `-u`:

- 5 PureMagic benchmarks (`data/circuits`), transpiled at `-m 1` and `-m 0`;
- Orig and DC-opt variants of 7 dcsynth benchmarks (tol 1e-4, ancillas hosted).

Each was scheduled under every model, at reaction latency L ∈ {0, 1, 2, 4}, with seeds 0–4: 1,920
runs, every one passing the checker. `inplace-y0` is in-place Y with `--y-meas-lcycles 0`, to
separate the cost of holding the magic patch from the cost of the Y measurement. Volume changes
exactly as lcycles do (the patch count is fixed).

Geomean lcycles relative to `legacy` at the same L, with the range over circuits:

| L | litinski | inplace-y | inplace-y0 |
|---|---|---|---|
| 0 | +4.1% [+0, +10] | +3.4% [+1, +9] | +1.1% [−1, +8] |
| 1 | +6.7% [+1, +11] | +14.8% [+3, +54] | +5.5% [+2, +10] |
| 2 | +25.4% [+7, +103] | +37.2% [+7, +154] | +25.2% [+5, +104] |
| 4 | +75.3% [+15, +303] | +90.8% [+16, +352] | +76.4% [+14, +303] |

- **The helper itself is cheap.** At L = 0 Litinski costs +4%: it borrows a cultivating
  neighbour for one lcycle, which restarts that patch's cultivation.
- **From L = 1, Litinski is the better model.** In-place Y pays the Y measurement on half the T
  gates, and that delays every later anticommuting decision. On T-dense, chain-heavy circuits
  (dnn, qugan) it costs +32% to +54% where Litinski costs +1% to +6%. With a free Y measurement
  (`inplace-y0`) the two models are within 1 point of each other, so the Y time is in-place Y's
  whole disadvantage.
- **At L ≥ 2 the reaction limit dominates every model.** Runs of anticommuting T gates need about
  L lcycles per T, which `legacy` never charged. Letting a T wait rather than hold a helper
  through a long chain was tried; it changed lcycles by at most 2%.

L = 1, per circuit (lcycle change relative to `legacy`; T is the number of T gates):

| circuit | T | legacy lcycles | litinski | inplace-y | inplace-y0 |
|---|---|---|---|---|---|
| ae_5.dc_opt | 3404 | 5406 | +8.6% | +10.0% | +7.2% |
| ae_5.orig | 1577 | 2332 | +4.7% | +12.7% | +4.4% |
| cdkm_ripple_carry_adder.m0 | 792 | 893 | +6.9% | +19.2% | +6.0% |
| cdkm_ripple_carry_adder.m1 | 792 | 6358 | +6.5% | +6.5% | +5.4% |
| dnn_n51.m0 | 15360 | 8712 | +6.4% | +32.4% | +5.4% |
| dnn_n51.m1 | 15360 | 9003 | +3.0% | +33.3% | +3.2% |
| grover_6.dc_opt | 23129 | 34421 | +8.2% | +12.0% | +7.0% |
| grover_6.orig | 7005 | 10797 | +6.8% | +14.8% | +5.1% |
| ising_n98.m0 | 15358 | 2078 | +5.3% | +5.5% | +3.4% |
| ising_n98.m1 | 15358 | 2101 | +3.7% | +4.6% | +2.1% |
| knn_n129.m0 | 10525 | 1360 | +10.9% | +13.4% | +7.5% |
| knn_n129.m1 | 10525 | 2034 | +5.3% | +2.9% | +2.3% |
| qaoa_16.dc_opt | 9583 | 9948 | +9.7% | +10.1% | +8.2% |
| qaoa_16.orig | 5024 | 3282 | +5.2% | +6.3% | +3.7% |
| qaoa_8.dc_opt | 2552 | 2098 | +8.5% | +11.7% | +5.9% |
| qaoa_8.orig | 2552 | 3221 | +6.2% | +8.3% | +5.1% |
| qnn_4.dc_opt | 3840 | 4772 | +7.2% | +9.7% | +5.7% |
| qnn_4.orig | 2209 | 4293 | +7.1% | +9.1% | +6.0% |
| qugan_n111.m0 | 26156 | 13372 | +3.9% | +54.3% | +4.3% |
| qugan_n111.m1 | 26156 | 16829 | +1.2% | +34.6% | +2.1% |
| tfim_20.dc_opt | 19283 | 23515 | +10.8% | +12.9% | +9.9% |
| tfim_20.orig | 6198 | 5088 | +9.2% | +16.8% | +8.3% |
| vqe_6.dc_opt | 255347 | 533023 | +9.9% | +10.1% | +8.3% |
| vqe_6.orig | 11862 | 18762 | +6.8% | +16.9% | +5.6% |

### The frame model

Same 24 circuits, with every ancilla as a data patch (the frame model cannot host them) for all
models: 1,920 runs, all passing the checker. Geomean lcycles relative to `legacy`, with the range:

| L | litinski | inplace-y | frame | frame beats litinski on |
|---|---|---|---|---|
| 0 | +4.1% [+0, +10] | +3.3% [+1, +9] | +9.6% [−4, +72] | 9 of 24 |
| 1 | +6.7% [+1, +11] | +14.7% [+3, +54] | +23.2% [−2, +97] | 7 of 24 |
| 2 | +25.3% [+7, +103] | +37.0% [+7, +154] | +40.4% [−1, +194] | 8 of 24 |
| 4 | +74.3% [+15, +303] | +89.9% [+16, +352] | +79.8% [+3, +389] | 10 of 24 |

Per circuit at L = 0 and L = 1 (relative to `legacy`), with the frame's mean T/measurement weight
as written and as executed, and the largest executed weight (seed 0, L = 0):

| circuit | L0 litinski | L0 frame | L1 litinski | L1 frame | weight written → executed | max |
|---|---|---|---|---|---|---|
| cdkm_ripple_carry_adder.m1 | +5.0% | +5.7% | +6.5% | +5.7% | 1.00 → 17.92 | 143 |
| cdkm_ripple_carry_adder.m0 | +6.8% | +10.5% | +6.9% | +10.9% | 81.44 → 78.56 | 200 |
| dnn_n51.m1 | +1.2% | +33.1% | +3.0% | +76.1% | 1.00 → 2.94 | 38 |
| dnn_n51.m0 | +3.2% | +7.1% | +6.4% | +58.6% | 2.47 → 2.70 | 42 |
| ising_n98.m1 | +1.2% | +3.0% | +3.7% | +4.2% | 1.00 → 1.72 | 6 |
| ising_n98.m0 | +2.1% | +0.7% | +5.3% | +0.7% | 1.75 → 1.78 | 4 |
| knn_n129.m1 | +0.5% | +2.3% | +5.3% | +2.6% | 1.00 → 1.05 | 90 |
| knn_n129.m0 | +8.3% | +2.2% | +10.9% | +2.3% | 1.05 → 1.08 | 99 |
| qugan_n111.m1 | +0.0% | +5.2% | +1.2% | +76.7% | 1.00 → 2.80 | 86 |
| qugan_n111.m0 | +2.3% | +0.7% | +3.9% | +97.3% | 2.47 → 2.87 | 90 |
| grover_6.orig | +3.4% | +1.8% | +6.8% | +21.8% | 2.90 → 3.80 | 6 |
| grover_6.dc_opt | +5.6% | +15.5% | +8.1% | +21.1% | 2.29 → 4.79 | 7 |
| vqe_6.orig | +3.0% | +0.0% | +6.8% | +30.2% | 4.14 → 4.19 | 6 |
| vqe_6.dc_opt | +8.6% | +24.8% | +9.8% | +25.4% | 1.55 → 5.00 | 7 |
| qaoa_8.orig | +2.8% | -3.9% | +6.2% | -2.5% | 2.83 → 2.34 | 6 |
| qaoa_8.dc_opt | +3.8% | +0.7% | +7.7% | +2.2% | 2.83 → 2.32 | 5 |
| qnn_4.orig | +4.7% | -0.4% | +7.1% | +2.2% | 2.63 → 2.49 | 4 |
| qnn_4.dc_opt | +3.8% | +6.9% | +6.5% | +9.6% | 1.93 → 2.65 | 5 |
| ae_5.orig | +2.1% | -0.8% | +4.7% | +21.1% | 2.51 → 2.26 | 4 |
| ae_5.dc_opt | +6.4% | +17.9% | +9.2% | +20.0% | 1.86 → 3.54 | 6 |
| qaoa_16.orig | +2.3% | +9.9% | +5.2% | +12.9% | 1.34 → 2.16 | 5 |
| qaoa_16.dc_opt | +8.7% | +30.3% | +9.7% | +29.2% | 1.32 → 2.66 | 7 |
| tfim_20.orig | +4.5% | +8.0% | +9.2% | +17.9% | 6.45 → 2.94 | 9 |
| tfim_20.dc_opt | +9.8% | +72.4% | +10.6% | +76.3% | 1.24 → 7.31 | 20 |

- **Physical Cliffords spread the frame.** Every S/SX/CX line conjugates W, so circuits with many
  of them see weights grow: DC-opt variants (which have physical Cliffords from the transpiler's
  flushes) and `-m 1` transpilations. tfim_20 DC-opt goes from weight 1.24 to 7.31 (+72% lcycles
  at L = 0), dnn_n51 `-m 1` from 1 to 2.94, cdkm `-m 1` from 1 to 17.9 (max 143).
- **Without physical Cliffords the frame can win.** Folding P into an overlapping Q can lower the
  weight (tfim_20 Orig: 6.45 → 2.94), and there is no helper. At L = 0 it beats Litinski on 9
  circuits, by up to 7 points (knn_n129 `-m 0`: +2.2% vs +8.3%; qaoa_8 Orig: −3.9%).
- **Data-side waits dominate from L = 1.** Long runs of non-commuting T gates stall the data:
  dnn_n51 and qugan_n111 cost +59% to +97% at L = 1, against +1% to +6% with Litinski.
- **It would bias the DC/Orig comparison.** DC circuits have the physical Cliffords, so the frame
  penalizes them more than Orig (vqe_6 at L = 0: Orig +0%, DC-opt +25%). Litinski charges both
  alike.

### The hybrid model

Added to the same comparisons (all ancillas as data patches), every run passing the checker.
Geomean lcycles relative to `legacy`:

| Set | L | litinski | frame | hybrid |
|---|---|---|---|---|
| 24 circuits as transpiled (w unlimited except 5 at `-m 1`) | 0 | +4.1% | +9.6% | +4.1% |
| | 1 | +6.7% | +23.2% | +14.1% |
| | 2 | +25.3% | +40.4% | +29.0% |
| | 4 | +74.3% | +79.8% | +69.1% |
| 19 circuits, all at `-m 1` | 0 | +3.2% | +19.5% | +2.7% |
| | 1 | +5.7% | +30.1% | +17.5% |
| | 4 | +67.4% | +79.9% | +77.3% |
| 4 large PureMagic benchmarks at `-m 1` | 0 | +1.9% | +465% (3 circuits) | +4.2% |
| | 1 | +4.8% | +457% (3 circuits) | +45.9% |
| | 4 | +183.5% | +497% (3 circuits) | +184.9% |

(The large set is qaoa N149, qft n160, square_heisenberg N225 and grover n100; frame mode did not
finish on grover n100, and qv N064 and fermi_hubbard 128q were run for the hybrid only.)

- **The hybrid removes the growth.** Products keep their weight, so it runs as fast as Litinski
  (qft n160: 5.7 s per schedule vs 5.6 s, frame 40.8 s), and at L = 0 it matches or slightly beats
  Litinski (+2.7% vs +3.2% at `-m 1`), with no helper patches at all for one-qubit T gates.
- **From L = 1 it loses to Litinski.** Runs of one-qubit T gates about alternating axes (the
  output of single-qubit rotation synthesis) still wait L lcycles per T on the data path, and CXs
  wait too. Litinski moves those waits onto helper patches: +6.7% vs +14.1% (w unlimited), +4.8%
  vs +45.9% (large `-m 1`).
- **At L = 4 the reaction limit dominates every model,** and the hybrid is even with or slightly
  ahead of Litinski on some sets (+69.1% vs +74.3%; qft n160 +145% vs +225%).

`litinski` (now `litinski-y`) was the best of these models at every latency from 1 up, and within
a point of the best at 0; `litinski-ready` (§2.1) is cheaper still on the PureMagic benchmarks
and is the default. `hybrid` is the better choice at zero reaction latency; `frame` is kept as the
reference for an exact frame.

### DC-opt vs Orig

DC-opt has more T gates per lcycle than Orig, so it pays slightly more. At L = 1 with Litinski,
the DC-opt/Orig lcycle ratio rises by +0.1% (qnn_4) to +4.3% (qaoa_16). Absolute lcycle counts
from before this change are optimistic by the amounts above.

## 6. Verification

- Unit tests (`src/scheduler.rs`):
  - `litinski_leaves_every_later_product_unchanged`: no relabelling and no added products, over 8
    seeds;
  - `litinski_helpers_are_free_neighbours_and_holds_are_respected`: each helper is adjacent and not
    a data patch; held patches are never used as a magic state; the hold totals add up;
  - `litinski_decisions_chain_only_through_anticommuting_t_gates`;
  - `inplace_y_holds_the_magic_patch_and_adds_y_time_for_failures`;
  - `inplace_y_failure_delays_the_next_anticommuting_decision`;
  - `legacy_mode_still_tracks_per_qubit_corrections`;
  - frame mode: `frame_mode_runs_every_product_once_and_rewrites_after_failures`,
    `frame_mode_waits_for_an_anticommuting_outcome`, `frame_mode_without_failures_matches_legacy`,
    `frame_mode_rejects_hosted_ancillas`; `src/frame.rs` checks the tableau against a slow
    event-list reference on random histories (up to 130 qubits) and its gate rules against the
    transpiler's signed tableau.
- Frame mode against exact unitaries: 900 random circuits × seeds × L ∈ {0, 2} of 2–4 qubits
  (T gates and measurements of weight 1–3, physical S/Sdg/SX/SXdg/CX), replayed from the trace in
  execution order (`bench_results/t_correction_bug/fuzz_frame.py`, `check.py`). Every executed T
  and measurement matches, and flipping the outcome of any T not yet decoded never changes a
  product that already ran. The hybrid is fuzzed the same way (960 runs over 80 circuits in
  both modes) and additionally checked never to run a CX that the frame does not commute with,
  also with any undecoded outcome flipped. Unit tests: `hybrid_keeps_the_frame_local_and_supports_unchanged`,
  `hybrid_runs_a_cx_only_after_its_flush`, `hybrid_without_failures_matches_legacy`.
- The schedule checker (`data_processing/plot_dynamic.py`) passes on every comparison run.
- A two-qubit circuit that shows both problems, and the one-qubit circuit above, were run with
  every model and compared with an exact Clifford-tableau calculation. `legacy` executes 4 and 2
  operations on the wrong Pauli; `litinski` and `inplace-y` execute every operation as written.
