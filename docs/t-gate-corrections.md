# Correcting failed T injections

A T gate in the Pauli-based circuit is a π/8 rotation R_P(π/8) = e^{−iπ/8·P} about a Pauli
product P. It is injected by measuring P⊗Z_m jointly with a magic-state patch m. Half the time
the rotation comes out as R_P(−π/8), and the circuit then owes the Clifford correction
R_P(π/4). This document covers how PureMagic used to account for that correction, why that was
wrong, and the two models that replace it.

| Flag | Values | Default |
|---|---|---|
| `--t-correction` | `litinski`, `inplace-y`, `legacy` | `litinski` |
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

## 2. Auto-corrected π/8 rotation (`--t-correction litinski`, default)

Litinski, *A Game of Surface Codes* (Quantum 3, 128, 2019), auto-corrected π/8 rotations. With a
|0⟩ helper patch h next to the magic patch m:

1. measure P⊗Z_m (the injection) and Z_m⊗Y_h in the same lcycle, then measure m in X. The magic
   patch is free again from the next lcycle;
2. once the outcome is decoded, measure h in **X** to apply the correction R_P(π/4), or in **Z** not
   to.

The correction is a choice of measurement basis on an ancilla. Every outcome only adds Paulis to
the frame, so **no later product is ever changed and no S gate is emitted.** Only the helper
waits.

What the scheduler does:

- **Helper.** A free neighbour of the magic patch, borrowed in the injection lcycle. A bus
  patch comes first, then the cultivating magic patch with the least progress (its cultivation
  restarts). Never a data patch, a hosted patch, or a ready magic state. With magic routing
  (`-u`) there are no bus patches, so the helper is always a cultivating magic patch. No patch
  is dedicated to helpers, so injection stays single-patch.
- **No free neighbour yet.** The injection waits up to 2 lcycles for one, holding the magic
  patch. After that, or if no neighbour could ever be a helper, the magic patch is measured in
  place (section 3). Waiting longer could deadlock two magic patches that wait on each other.
- **Hold.** The helper stays occupied until the correction is decided (the decision lcycle below).
  Data-side products are never delayed: the helper's measurement acts on a different patch, so
  it commutes with everything that follows on the data.

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
  - `legacy_mode_still_tracks_per_qubit_corrections`.
- The schedule checker (`data_processing/plot_dynamic.py`) passes on every comparison run.
- A two-qubit circuit that shows both problems, and the one-qubit circuit above, were run with
  every model and compared with an exact Clifford-tableau calculation. `legacy` executes 4 and 2
  operations on the wrong Pauli; `litinski` and `inplace-y` execute every operation as written.
