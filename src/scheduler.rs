use crate::accum_start;
use crate::astar::{AStar, PathResult};
use crate::circuit::Circuit;
use crate::cultivation::{CultivationManager, NOT_CULTIVATING};
use crate::debug_sched;
use crate::fn_timer;
use crate::frame::{CliffordFrame, FrameGate, SPauli};
use crate::info_sched;
use crate::node::NodeType;
use crate::pauliproduct::{GateType, Operator, PauliProduct};
use crate::steinertree::SteinerTree;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use crate::utils::AccumTimers;
use colored::{Color, Colorize};

use indexmap::{IndexMap, IndexSet};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

/// Per-lcycle and cumulative qubit utilisation counters.
pub(crate) struct ScheduleStats {
    data_qubits: usize,
    bus_qubits: usize,
    magic_qubits: usize,
    /// Cumulative totals across all lcycles (reset only at construction).
    tot_data_used: usize,
    tot_bus_used: usize,
    tot_magic_used: usize,
    /// Per-lcycle counters; reset to 0 at the end of each `update()` call.
    bus_used: usize,
    data_used: usize,
    magic_used: usize,
    /// T products that consumed a magic node this lcycle (first attempt only).
    t_scheduled: usize,
    /// Ready magic nodes used in any path this lcycle (routing or T-terminal).
    magic_ready_used: usize,
    tot_magic_unused: usize,
    plot_info: String,
}

impl ScheduleStats {
    pub(crate) fn new(data_qubits: usize, bus_qubits: usize, magic_qubits: usize) -> Self {
        ScheduleStats {
            data_qubits,
            bus_qubits,
            magic_qubits,
            tot_data_used: 0,
            tot_bus_used: 0,
            tot_magic_used: 0,
            bus_used: 0,
            data_used: 0,
            magic_used: 0,
            t_scheduled: 0,
            magic_ready_used: 0,
            tot_magic_unused: 0,
            plot_info: String::new(),
        }
    }

    pub(crate) fn summarize(&self, n_lcycles: usize) {
        let data_frac = self.tot_data_used as f64 / (self.data_qubits * n_lcycles) as f64;
        let bus_frac = self.tot_bus_used as f64 / (self.bus_qubits * n_lcycles) as f64;
        let magic_frac = self.tot_magic_used as f64 / (self.magic_qubits * n_lcycles) as f64;
        let magic_unused_frac =
            self.tot_magic_unused as f64 / (self.magic_qubits * n_lcycles) as f64;
        println!("Qubit fractions used:");
        println!("  data:        {:.3}", data_frac);
        println!("  bus:         {:.3}", bus_frac);
        println!("  magic:       {:.3}", magic_frac);
        println!("Magic unused {:.3}", magic_unused_frac);
    }

    pub(crate) fn update(
        &mut self, lcycle_i: usize, pp_paths_len: usize, tot_available: usize, magic_ready: usize,
        magic_unused: usize, plotting: bool,
    ) {
        self.tot_data_used += self.data_used;
        self.tot_bus_used += self.bus_used;
        self.tot_magic_used += self.magic_used;
        self.tot_magic_unused += magic_unused;

        let tot_qubits = self.data_qubits + self.bus_qubits + self.magic_qubits;
        let tot_qubits_used = self.data_used + self.bus_used + self.magic_used;

        info_sched!("Scheduling results:");
        let frac_paths =
            if tot_available == 0 { 1.0 } else { pp_paths_len as f64 / tot_available as f64 };
        let frac_qubits =
            if tot_qubits == 0 { 0.0 } else { tot_qubits_used as f64 / tot_qubits as f64 };
        // magic_denom excludes nodes consumed purely as routing intermediaries (magic-routing mode).
        let magic_ready_routing = self.magic_ready_used.saturating_sub(self.t_scheduled);
        let magic_denom = magic_ready.saturating_sub(magic_ready_routing);
        let frac_magic =
            if magic_denom == 0 { 0.0 } else { self.t_scheduled as f64 / magic_denom as f64 };
        info_sched!("  products:    {}/{} ({:.2})", pp_paths_len, tot_available, frac_paths);
        info_sched!("  qubits:      {}/{} ({:.2})", tot_qubits_used, tot_qubits, frac_qubits);
        info_sched!("  magic:       {}/{} ({:.2})", self.t_scheduled, magic_denom, frac_magic);
        if plotting {
            self.plot_info = format!(
                "lcycle {}: products scheduled {}/{} ({:.2}), qubits {}/{} ({:.2}), magic {}/{} ({:.2})",
                lcycle_i,
                pp_paths_len,
                tot_available,
                frac_paths,
                tot_qubits_used,
                tot_qubits,
                frac_qubits,
                self.t_scheduled,
                magic_denom,
                frac_magic,
            );
        }

        self.data_used = 0;
        self.bus_used = 0;
        self.magic_used = 0;
        self.t_scheduled = 0;
        self.magic_ready_used = 0;
    }

    pub(crate) fn inc(&mut self, node_type: NodeType) {
        match node_type {
            NodeType::Bus => self.bus_used += 1,
            NodeType::Magic => self.magic_used += 1,
            NodeType::Data => self.data_used += 1,
        }
    }

    pub(crate) fn inc_with_cultivation(&mut self, node_type: NodeType, cultivation_time: i32) {
        self.inc(node_type);
        if node_type == NodeType::Magic && cultivation_time == 0 {
            self.magic_ready_used += 1;
        }
    }

    pub(crate) fn inc_t(&mut self) {
        self.t_scheduled += 1;
    }

    pub(crate) fn plot_info_str(&self) -> &str {
        &self.plot_info
    }
}

/// Read-only inputs grouped so methods can borrow `input` and mutable state
/// fields simultaneously without conflicting borrows.
pub(crate) struct SchedulerInput {
    pub circuit: Circuit,
    pub topo: TopoGraph,
}

impl SchedulerInput {
    pub(crate) fn circuit_stem(&self) -> &str {
        use std::path::Path;
        Path::new(&self.circuit.circuit_fname)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("circuit")
    }
}

/// Lcycles a Litinski injection waits for a free neighbour to borrow as its |0> helper before
/// its magic patch is measured in place instead.
const HELPER_MAX_WAIT: usize = 2;

/// How a failed T injection is corrected (see docs/t-gate-corrections.md).
///
/// A T is a pi/8 rotation about a Pauli P, done by measuring P (x) Z_m with a magic patch m. Half
/// the time it comes out as -pi/8 and owes a pi/4 rotation about P.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum TCorrection {
    /// Auto-corrected pi/8 rotation (Litinski, Game of Surface Codes). In the injection lcycle a
    /// free neighbour of the magic patch is borrowed as a |0> helper; Z_m (x) Y_helper is measured
    /// alongside P (x) Z_m, then the magic patch in X, which frees it. Once the injection outcome
    /// is decoded the helper is measured in X or Z, which applies the correction or not. No
    /// Clifford ever reaches the data: later products are never changed and no S gate is
    /// emitted. The helper is held until that decision (reaction latency, chained through
    /// earlier T gates whose Paulis anticommute). The helper is a bus patch, else a cultivating
    /// magic patch, else a ready magic state, whose state is wasted. An injection that finds no
    /// helper within `HELPER_MAX_WAIT` lcycles has its magic patch measured in place instead (in
    /// X, or in Y for a failed T); with ready states allowed this is rare. The default.
    LitinskiReady,
    /// As `litinski-ready`, but a ready magic state is never borrowed as the helper. Injections
    /// whose free neighbours are all ready states wait for a helper, and after `HELPER_MAX_WAIT`
    /// lcycles fall back to measuring the magic patch in place in X or Y (Gidney). This was
    /// `litinski` before the two were split.
    LitinskiY,
    /// In-place Y measurement (Gidney, "Inplace Access to the Surface Code Y Basis", Quantum
    /// 2024): no helper. After P (x) Z_m the magic patch itself is held until the correction is
    /// decided, then measured in X (no correction) or Y (applies it). A Y measurement takes
    /// floor(d/2)+2 rounds, modelled as `--y-meas-lcycles` extra lcycles on the magic patch, and
    /// the frame bit it yields is known only after that. Single-patch injection, but the magic
    /// patch cannot cultivate while it waits.
    InplaceY,
    /// Exact Clifford frame (src/frame.rs): a failed T's correction R_P(pi/4) is never applied.
    /// It is folded into a frame W, and every later T and measurement about Q runs about
    /// W.Q.W^dag, possibly on more qubits than Q. Physical S/SX/CX run unchanged and
    /// conjugate the frame. No helper and no hold, but a T or measurement whose rewritten
    /// Pauli depends on an undecoded injection outcome (it anticommutes with a T injected in
    /// the last L lcycles) waits until that outcome is known. Not supported with hosted
    /// ancillas.
    Frame,
    /// Hybrid of `frame` and `litinski` that keeps the frame local (one one-qubit Clifford per
    /// qubit), so products never grow. A failed one-qubit T is folded into the frame; a failed
    /// multi-qubit T, whose correction would entangle, is auto-corrected on a helper as in
    /// `litinski`. Physical S/SX pass through the frame. Before a physical CX whose qubits' frame
    /// does not commute with it, one or two pi/4 one-qubit rotations are applied physically to
    /// bring that qubit's frame to the identity. A CX also waits for undecoded one-qubit T
    /// outcomes on its qubits (they decide whether a rotation is needed). Not supported with
    /// hosted ancillas.
    Hybrid,
    /// The per-qubit S-correction record PureMagic used before (correction_power). Incorrect: it
    /// splits the correction of a multi-qubit T into independent one-qubit corrections and
    /// merges corrections about different axes into a Pauli. Kept to compare against.
    Legacy,
}

/// How the scheduler treats conditional products (a `?id=v,..` suffix in the `.trans` file).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum CondOutcomes {
    /// Sample each measurement's true outcome when a condition first needs it (1 with
    /// probability `cond_p1`), and skip every conditional product whose condition then fails:
    /// the schedule is that of the circuit the sampled outcomes actually call for.
    Random,
    /// Ignore conditions and run every conditional product: an upper bound on the cost.
    Worst,
}

/// Assigns Pauli products to lcycles and routes them through the topology.
pub(crate) struct Scheduler {
    pub(crate) input: SchedulerInput,
    rng_uniform: StdRng,
    magic_state_lambda: f64,
    plot_option: String,
    pub(crate) cultivation: CultivationManager,
    pub(crate) stats: ScheduleStats,
    /// (lcycle_index, product_ids) for every lcycle that scheduled at least one product.
    pub(crate) lcycle_scheduled: Vec<(usize, Vec<i32>)>,
    pub(crate) scheduled_products: IndexSet<i32>,
    /// Per-node occupancy bitmap for the current lcycle; reset to all-false each lcycle.
    used: Vec<bool>,
    /// In-progress multi-lcycle Clifford gates: maps product_id →
    /// (remaining_lcycles, product, node_ids, opt_tree).
    /// CX occupies 2 lcycles; S/SX occupy 3.
    clifford_paths: IndexMap<i32, (usize, PauliProduct, Vec<u16>, Option<Rc<TreeGraph>>)>,
    pub(crate) t_gate_failures: usize,
    pub(crate) correction_gates_emitted: usize,
    pub(crate) stree_computation: SteinerTree,
    pub(crate) astar: AStar,
    no_t_failures: bool,
    record_cultivation_dist: bool,
    /// Reusable buffers to avoid per-lcycle allocations.
    terminals_buf: Vec<u16>,
    scheduled_ids_buf: Vec<i32>,
    children_buf: Vec<i32>,
    precomputed_clifford_trees: HashMap<i32, Rc<TreeGraph>>,
    remaining_ids_buf: Vec<i32>,
    /// Indexed by product_id; built once before scheduling begins.
    precomputed_terminals: Vec<Vec<u16>>,
    /// (is_y_paired, preferred_roots, side_roots) per terminal per product.
    precomputed_root_info: Vec<Vec<(bool, Vec<u16>, Vec<u16>)>>,
    timers: AccumTimers,
    loop_timer: usize,
    other_timer: usize,
    pps_pending: Vec<PauliProduct>,
    /// Remaining unresolved parent count per product; decremented as parents complete.
    remaining_parents: Vec<usize>,
    pub(crate) pp_paths: Vec<(i32, Option<Rc<TreeGraph>>)>,
    current_lcycle: usize,
    /// Per-qubit accumulated S^k correction power (0=none, 1=S, 2=Z, 3=S†).
    /// Incremented on T gate failure; cleared when a Clifford is encountered or power cycles to 0.
    correction_power: Vec<u8>,
    /// Pauli basis of the most-recent conjugated T-gate operator per qubit.
    /// Used as the operator basis when emitting an S correction before a Clifford.
    correction_basis: Vec<char>,
    /// Next ID to assign when creating a dynamic S correction product.
    next_correction_id: i32,
    /// Maps product_id → (qubit, new_basis) pairs for T gates whose operators were
    /// conjugated this lcycle. Populated in sched_remaining; consumed in process_t_gate_outcomes.
    t_conjugated_bases: HashMap<i32, Vec<(u16, char)>>,
    /// T gate product IDs whose precomputed_terminals/root_info currently reflect a
    /// conjugated basis; triggers a restore via apply_t_conjugation once correction_power hits zero.
    conjugated_t_products: HashSet<i32>,
    /// S product IDs fully discharged for free by `try_merge_s_correction`: a pending
    /// correction merged with this product's own S to a net power of 0 or 2 (identity or
    /// classical Z), so zero physical gates were scheduled. Consulted by `unlock_children`/
    /// `advance_clifford_state`/`check_clifford_repetitions` to treat these as complete,
    /// single-lcycle, zero-node events instead of ordinary 3-lcycle S/SX Cliffords.
    free_merged_products: HashSet<i32>,
    /// Classical reaction latency, in lcycles. A product with a classical dependency (`after`, the
    /// `@id,..` suffix of a `.trans` line) may start only this many lcycles later than it would
    /// from the dependency's completion alone: the outcome of a logical measurement is not
    /// usable until its syndrome rounds have been decoded. 0 (the default) keeps the scheduler
    /// exactly as it was.
    reaction_latency: usize,
    /// `(due_lcycle, child_id)`: parent completions on classical edges whose release is held back
    /// by `reaction_latency`. Released at the start of the first lcycle after `due_lcycle`.
    delayed_unlocks: Vec<(usize, i32)>,
    /// Conditional products: how they are decided, and P(outcome = 1) for `Random`.
    cond_outcomes: CondOutcomes,
    cond_p1: f64,
    /// Separate stream for measurement outcomes, so sampling them leaves the cultivation and
    /// T-failure draws (and hence an unconditional circuit's schedule) unchanged.
    rng_outcomes: StdRng,
    /// Sampled true outcome per measurement product id, in the order they were sampled.
    pub(crate) outcomes: IndexMap<i32, bool>,
    /// Conditional products skipped because their condition failed, in the order skipped.
    pub(crate) skipped: Vec<i32>,
    skipped_set: HashSet<i32>,
    n_skipped_t: usize,
    /// Hosted-ancilla state (`--ancilla-qubits`), indexed by logical qubit; empty when
    /// nothing is hosted. A hosted patch is dormant (cultivating) until one of its products
    /// is ready, and again after a product that leaves it in a known single-patch state
    /// (see `releases_patch`) until its next product is ready, and after its last product.
    hosted: HostedAncillas,
    /// How failed T injections are corrected.
    t_correction: TCorrection,
    /// Patches held after a T injection until its correction is decided: (last lcycle, node).
    held: Vec<(usize, u16)>,
    /// Litinski injections whose magic patch had no free neighbour for the |0> helper yet:
    /// (T product, magic node, decision lcycle, give-up lcycle). The magic patch is held until
    /// a helper is found.
    helper_wanted: Vec<(i32, u16, usize, usize)>,
    /// Magic patches measured in place, by T product: (magic node, decision lcycle). If the T
    /// failed, the patch is measured in Y, which takes `y_meas_lcycles` more.
    inplace_pending: HashMap<i32, (u16, usize)>,
    /// Failed T products still waiting for a helper (their measurement basis is not known yet).
    failed_waiting: HashSet<i32>,
    /// Extra lcycles an in-place Y measurement of a magic patch takes.
    y_meas_lcycles: usize,
    /// True if `hold_t_patches` occupied any patch in the current lcycle.
    holding_now: bool,
    /// Per qubit, recent T products and the lcycle their correction is decided (decision chain).
    t_decisions: Vec<Vec<(i32, usize)>>,
    pub(crate) t_stats: TCorrectionStats,
    /// Frame mode: the Clifford frame (None in other modes and with -F).
    frame: Option<CliffordFrame>,
    /// Frame mode: T injections whose outcome is not decoded yet, as (last undecided lcycle,
    /// ideal axis conjugated by the physical Cliffords run since).
    frame_undecided: Vec<(usize, SPauli)>,
    /// Frame mode: rewritten Paulis of products currently routed on them.
    frame_exec: HashMap<i32, Vec<Operator>>,
    /// Frame mode: physical Cliffords already folded into the frame (they span several lcycles).
    frame_applied: HashSet<i32>,
    pub(crate) frame_stats: FrameStats,
}

/// Counters for `--t-correction frame`, reported after scheduling.
#[derive(Default, Debug, Clone)]
pub(crate) struct FrameStats {
    /// Failed T gates folded into the frame, and physical Cliffords that conjugated it.
    pub(crate) failures: usize,
    pub(crate) cliffords: usize,
    /// Product-lcycles T gates and measurements waited for an undecoded outcome.
    pub(crate) waits: usize,
    /// Hybrid: pi/4 one-qubit rotations applied physically to clear the frame before a CX.
    pub(crate) flushes: usize,
    /// Executed T gates and measurements, how many ran rewritten, their total weight as written
    /// and as executed, and the largest executed weight.
    pub(crate) products: usize,
    pub(crate) rewritten: usize,
    pub(crate) ideal_weight: usize,
    pub(crate) exec_weight: usize,
    pub(crate) max_exec_weight: usize,
    /// One entry per executed T or measurement: (lcycle, product, executed operators). Kept in
    /// test and debug builds only.
    pub(crate) log: Vec<(usize, i32, Vec<Operator>)>,
}

/// Counters for the T-correction model, reported after scheduling.
#[derive(Default, Debug, Clone)]
pub(crate) struct TCorrectionStats {
    /// Injections handled by the auto-correction model.
    pub(crate) injections: usize,
    /// |0> helper patches borrowed (Litinski), and how many had to wait for a free neighbour.
    pub(crate) helpers: usize,
    pub(crate) helpers_deferred: usize,
    /// Helpers that were ready magic states (`litinski-ready`; the state is wasted).
    pub(crate) helpers_ready: usize,
    /// Injections that found no helper within `HELPER_MAX_WAIT` lcycles (or whose magic patch
    /// has no neighbour that could ever be one). Their magic patch is measured in place (the
    /// in-place Y measurement) instead; see docs/t-gate-corrections.md.
    pub(crate) helpers_missing: usize,
    /// Magic patches measured in place, and how many of those needed the (slower) Y basis.
    pub(crate) inplace: usize,
    pub(crate) inplace_y: usize,
    /// Patch-lcycles spent holding magic and helper patches after their injection lcycle.
    pub(crate) hold_patch_lcycles: usize,
    /// Largest gap between an injection and its correction decision, in lcycles.
    pub(crate) max_decision_delay: usize,
    /// One entry per injection: (lcycle, T product, magic patch, decision lcycle). Helpers are
    /// logged separately as (lcycle, magic patch, helper), since a helper can come later.
    pub(crate) injection_log: Vec<(usize, i32, u16, usize)>,
    pub(crate) helper_log: Vec<(usize, u16, u16)>,
}

/// Bookkeeping for qubits hosted on injection patches.
#[derive(Default)]
pub(crate) struct HostedAncillas {
    /// Products of the input circuit not yet completed, per hosted qubit.
    remaining: Vec<usize>,
    /// Products on each hosted qubit, in program order (a chain: each depends on the
    /// previous one on the same qubit), and how many of them have completed.
    chain: Vec<Vec<i32>>,
    done: Vec<usize>,
    pub(crate) live: Vec<bool>,
    /// Lcycles each hosted qubit spent live, and how often it went live.
    pub(crate) live_lcycles: Vec<usize>,
    pub(crate) activations: Vec<usize>,
    /// Products completed in the current lcycle (filled by `unlock_children`).
    completed_buf: Vec<i32>,
    n_initial_products: usize,
}

impl Scheduler {
    pub(crate) fn new(
        circuit: Circuit, topo: TopoGraph, magic_state_lambda: f64, log_level: &str,
        plot_option: String, rseed: u32, no_t_failures: bool, record_cultivation_dist: bool,
    ) -> Self {
        let n_qubits = circuit.n_qubits;
        let n_initial_products = circuit.n_products() as i32;
        if log_level != "none" {
            // Installed regardless of build profile, but debug_sched!/info_sched! (see
            // utils.rs) are compiled out entirely in release, so a release build still
            // creates trace_fname, it just never gets anything written to it.
            let trace_fname = format!("{}.sched_trace", circuit_stem(&circuit.circuit_fname));
            let level_filter = match log_level.to_lowercase().as_str() {
                "debug" => log::LevelFilter::Debug,
                "info" => log::LevelFilter::Info,
                _ => log::LevelFilter::Off,
            };
            simple_logging::log_to_file(&trace_fname, level_filter)
                .expect("Failed to initialize logging");
        }
        let n_data_qubits = topo.n_data_qubits;
        let n_bus_qubits = topo.n_bus_qubits;
        let n_magic_qubits = topo.n_magic_qubits;
        let n_nodes = topo.n_nodes;
        let mut timers = AccumTimers::new();
        let loop_timer = timers.add_or_get("schedule loop");
        let other_timer = timers.add_or_get("other ");
        Scheduler {
            input: SchedulerInput { circuit, topo },
            rng_uniform: StdRng::seed_from_u64(rseed as u64),
            magic_state_lambda,
            plot_option,
            cultivation: CultivationManager::new(rseed),
            stats: ScheduleStats::new(n_data_qubits, n_bus_qubits, n_magic_qubits),
            lcycle_scheduled: Vec::new(),
            scheduled_products: IndexSet::new(),
            used: vec![false; n_nodes],
            clifford_paths: IndexMap::new(),
            t_gate_failures: 0,
            correction_gates_emitted: 0,
            stree_computation: SteinerTree::new(n_nodes),
            astar: AStar::new(n_nodes),
            no_t_failures,
            record_cultivation_dist,
            terminals_buf: Vec::new(),
            scheduled_ids_buf: Vec::new(),
            children_buf: Vec::new(),
            precomputed_clifford_trees: HashMap::new(),
            remaining_ids_buf: Vec::new(),
            precomputed_terminals: Vec::new(),
            precomputed_root_info: Vec::new(),
            timers: timers,
            loop_timer: loop_timer,
            other_timer: other_timer,
            pps_pending: Vec::new(),
            remaining_parents: Vec::new(),
            pp_paths: Vec::new(),
            current_lcycle: 0,
            correction_power: vec![0u8; n_qubits],
            correction_basis: vec!['X'; n_qubits],
            next_correction_id: n_initial_products,
            t_conjugated_bases: HashMap::new(),
            conjugated_t_products: HashSet::new(),
            free_merged_products: HashSet::new(),
            reaction_latency: 0,
            delayed_unlocks: Vec::new(),
            cond_outcomes: CondOutcomes::Random,
            cond_p1: 0.5,
            rng_outcomes: StdRng::seed_from_u64(rseed as u64 ^ 0x6f75_7463_6f6d_6573),
            outcomes: IndexMap::new(),
            skipped: Vec::new(),
            skipped_set: HashSet::new(),
            n_skipped_t: 0,
            hosted: HostedAncillas::default(),
            t_correction: TCorrection::LitinskiReady,
            held: Vec::new(),
            helper_wanted: Vec::new(),
            inplace_pending: HashMap::new(),
            failed_waiting: HashSet::new(),
            y_meas_lcycles: 1,
            holding_now: false,
            t_decisions: Vec::new(),
            t_stats: TCorrectionStats::default(),
            frame: None,
            frame_undecided: Vec::new(),
            frame_exec: HashMap::new(),
            frame_applied: HashSet::new(),
            frame_stats: FrameStats::default(),
        }
    }

    /// Sets how failed T injections are corrected (default: Litinski auto-correction).
    pub(crate) fn set_t_correction(&mut self, mode: TCorrection) {
        self.t_correction = mode;
    }

    /// True if a failure of T product `pp_id` is folded into the frame (frame: every T; hybrid:
    /// one-qubit T gates). Otherwise it is auto-corrected on a helper.
    fn frame_owns(&self, pp_id: i32) -> bool {
        match self.t_correction {
            TCorrection::Frame => true,
            TCorrection::Hybrid => self.input.circuit.product(pp_id).operators.len() == 1,
            _ => false,
        }
    }

    /// Hybrid: makes CX `pp_id` safe to run without spreading the frame. Returns true if it can
    /// run now. Otherwise it waits for an undecoded outcome on its qubits, or a flush rotation
    /// was placed (or could not be) on a qubit whose frame does not commute with the CX.
    fn hybrid_cx_ready(&mut self, pp_id: i32, plotting: bool) -> bool {
        let pp = self.input.circuit.product(pp_id);
        let c = pp.operators.iter().find(|op| op.basis == 'Z').unwrap().qubit;
        let t = pp.operators.iter().find(|op| op.basis == 'X').unwrap().qubit;
        if self
            .frame_undecided
            .iter()
            .any(|(_, ax)| ax.letter(c as usize) != '_' || ax.letter(t as usize) != '_')
        {
            info_sched!("  CX {} waits for an undecoded T outcome on its qubits", pp_id);
            self.frame_stats.waits += 1;
            self.holding_now = true;
            return false;
        }
        // CX commutes with a Z-diagonal Clifford on the control and an X-diagonal one on the
        // target (up to Paulis); anything else would spread the frame.
        let mut ready = true;
        if self.frame.as_ref().unwrap().local_letters(c as usize).1 != 'Z' {
            ready = false;
            self.emit_frame_flush(c, plotting);
        }
        if self.frame.as_ref().unwrap().local_letters(t as usize).0 != 'X' {
            ready = false;
            self.emit_frame_flush(t, plotting);
        }
        ready
    }

    /// Hybrid: physically applies one pi/4 rotation on qubit `q` that moves its frame toward
    /// the identity, as a scheduler-added S-type product. Returns true if it was placed.
    fn emit_frame_flush(&mut self, q: u16, plotting: bool) -> bool {
        let Some(gate) = self.frame.as_ref().unwrap().flush_step(q as usize) else {
            return true;
        };
        let (basis, name) = match gate {
            FrameGate::S(_) => ('Z', "S"),
            FrameGate::SX(_) => ('X', "SX"),
            FrameGate::SY(_) => ('Y', "SY"),
            FrameGate::CX(..) => unreachable!(),
        };
        let s_id = self.next_correction_id;
        self.input.circuit.pps.push(PauliProduct {
            id: s_id,
            gate_type: GateType::S,
            operators: vec![Operator { qubit: q, basis }],
            parents: vec![],
            children: vec![],
            after: Vec::new(),
            cond: Vec::new(),
            max_qubit: q,
        });
        let ops = vec![Operator { qubit: q, basis }];
        let terminals = operators_to_node_ids(&self.input.topo, &ops);
        let root_info = Self::compute_root_info_for_terminals(&self.input.topo, &terminals);
        self.precomputed_terminals.push(terminals);
        self.precomputed_root_info.push(root_info);
        let placed = if self.terminal_nodes(s_id) {
            self.sched_s_sx(s_id, plotting)
        } else {
            PathResult::NoPath
        };
        match placed {
            PathResult::PathFound(opt_graph) => {
                info_sched!("  frame flush {} on q{} as product {}", name, q, s_id);
                self.next_correction_id += 1;
                self.correction_gates_emitted += 1;
                if let Some(ref g) = opt_graph {
                    let node_ids: Vec<u16> = g.iter_nodes().collect();
                    self.mark_nodes_used(&node_ids);
                }
                self.pp_paths.push((s_id, opt_graph.map(Rc::new)));
                self.frame.as_mut().unwrap().apply_physical(gate);
                self.frame_applied.insert(s_id);
                self.frame_stats.flushes += 1;
                true
            }
            PathResult::NoPath => {
                self.input.circuit.pps.pop();
                self.precomputed_terminals.pop();
                self.precomputed_root_info.pop();
                false
            }
        }
    }

    /// Frame mode, start of an lcycle: forget injections whose outcome is now decoded.
    fn frame_begin_lcycle(&mut self) {
        let now = self.current_lcycle;
        self.frame_undecided.retain(|&(last, _)| last >= now);
    }

    /// Frame mode: prepares T or measurement `pp_id` to run in the current frame. Returns false
    /// if it must wait (its rewritten Pauli depends on an undecoded outcome). Otherwise its
    /// terminals and root candidates are those of W.Q.W^dag.
    fn frame_prepare(&mut self, pp_id: i32) -> bool {
        let Some(frame) = self.frame.as_ref() else {
            return true;
        };
        let ideal = SPauli::from_ops(frame.words, &self.input.circuit.product(pp_id).operators);
        if self.frame_undecided.iter().any(|(_, axis)| axis.anticommutes(&ideal)) {
            info_sched!("  {} waits for an undecoded T outcome", pp_id);
            self.frame_stats.waits += 1;
            self.holding_now = true; // not a dead end: the outcome arrives
            return false;
        }
        let exec = frame.image(&ideal);
        if exec == ideal {
            if self.frame_exec.remove(&pp_id).is_some() {
                // Back to the Pauli as written (the frame changed since the last attempt).
                let ops = self.input.circuit.product(pp_id).operators.clone();
                self.set_terminals(pp_id, &ops);
                self.conjugated_t_products.remove(&pp_id);
            }
            return true;
        }
        let ops = exec.to_ops();
        if self.frame_exec.get(&pp_id) != Some(&ops) {
            self.set_terminals(pp_id, &ops);
            self.frame_exec.insert(pp_id, ops);
            self.conjugated_t_products.insert(pp_id);
        }
        true
    }

    /// Routes product `pp_id` on the data patches of `ops` instead of its own operators.
    fn set_terminals(&mut self, pp_id: i32, ops: &[Operator]) {
        let terminals = operators_to_node_ids(&self.input.topo, ops);
        let root_info = Self::compute_root_info_for_terminals(&self.input.topo, &terminals);
        self.precomputed_terminals[pp_id as usize] = terminals;
        self.precomputed_root_info[pp_id as usize] = root_info;
    }

    /// Frame mode: the operators product `pp_id` runs on (its own unless rewritten).
    fn exec_ops(&self, pp_id: i32) -> &[Operator] {
        match self.frame_exec.get(&pp_id) {
            Some(ops) => ops,
            None => &self.input.circuit.product(pp_id).operators,
        }
    }

    /// Frame mode: bookkeeping for T or measurement `pp_id`, scheduled this lcycle.
    fn frame_executed(&mut self, pp_id: i32) {
        if self.frame.is_none() {
            return;
        }
        let ideal_w = self.input.circuit.product(pp_id).operators.len();
        let ops = self.exec_ops(pp_id).to_vec();
        let st = &mut self.frame_stats;
        st.products += 1;
        st.ideal_weight += ideal_w;
        st.exec_weight += ops.len();
        st.max_exec_weight = st.max_exec_weight.max(ops.len());
        if self.frame_exec.contains_key(&pp_id) {
            st.rewritten += 1;
        }
        #[cfg(debug_assertions)]
        {
            let frame = self.frame.as_ref().unwrap();
            let label = SPauli::from_ops(frame.words, &ops).to_label(frame.n);
            info_sched!("  product {} executed as {}", pp_id, label);
        }
        if cfg!(any(test, debug_assertions)) {
            self.frame_stats.log.push((self.current_lcycle, pp_id, ops));
        }
        if self.input.circuit.product(pp_id).gate_type.is_t() && self.frame_owns(pp_id) {
            // Until decoded, its outcome decides the Pauli of anything that anticommutes with it.
            let frame = self.frame.as_ref().unwrap();
            let axis = SPauli::from_ops(frame.words, &self.input.circuit.product(pp_id).operators);
            self.frame_undecided.push((self.current_lcycle + self.reaction_latency, axis));
        }
    }

    /// Frame mode, end of an lcycle: fold failed T gates and newly started physical Cliffords
    /// into the frame.
    fn frame_end_lcycle(&mut self, failed: &[i32]) {
        let Some(frame) = self.frame.as_mut() else {
            return;
        };
        for &pp_id in failed {
            let axis = SPauli::from_ops(frame.words, &self.input.circuit.product(pp_id).operators);
            frame.absorb_failed_t(&axis);
            self.frame_stats.failures += 1;
        }
        for i in 0..self.pp_paths.len() {
            let pp_id = self.pp_paths[i].0;
            if pp_id as usize >= self.input.circuit.n_products()
                || self.frame_applied.contains(&pp_id)
            {
                continue;
            }
            let pp = self.input.circuit.product(pp_id);
            let gate = match pp.gate_type {
                GateType::S => FrameGate::S(pp.operators[0].qubit as usize),
                GateType::SX => FrameGate::SX(pp.operators[0].qubit as usize),
                GateType::CX => {
                    let c = pp.operators.iter().find(|op| op.basis == 'Z').unwrap().qubit;
                    let t = pp.operators.iter().find(|op| op.basis == 'X').unwrap().qubit;
                    FrameGate::CX(c as usize, t as usize)
                }
                _ => continue,
            };
            frame.absorb_physical_clifford(gate);
            info_sched!("  physical Clifford {} folded into the frame", pp_id);
            for (_, axis) in &mut self.frame_undecided {
                axis.conjugate_by(gate);
            }
            self.frame_applied.insert(pp_id);
            self.frame_stats.cliffords += 1;
        }
    }

    /// Sets how many extra lcycles an in-place Y measurement of a magic patch takes (default 1:
    /// floor(d/2)+2 rounds, rounded up to whole lcycles).
    pub(crate) fn set_y_meas_lcycles(&mut self, lcycles: usize) {
        self.y_meas_lcycles = lcycles;
    }

    /// True if the Paulis of products `a` and `b` anticommute (they differ on an odd number of
    /// shared qubits).
    fn anticommute(&self, a: i32, b: i32) -> bool {
        let pa = &self.input.circuit.product(a).operators;
        let pb = &self.input.circuit.product(b).operators;
        let mut odd = false;
        for oa in pa {
            if let Some(ob) = pb.iter().find(|ob| ob.qubit == oa.qubit) {
                odd ^= oa.basis != ob.basis;
            }
        }
        odd
    }

    /// The lcycle in which the correction of T product `pp_id`, injected this lcycle, is decided.
    ///
    /// Its own injection outcome is decoded after the reaction latency L. Whether a correction is
    /// owed also depends on the Pauli-frame bits of earlier T gates whose Paulis anticommute with
    /// this one (their byproduct flips this rotation's sign), and those bits are known only L
    /// lcycles after their own decision. So decisions chain: with L = 0 the decision is immediate.
    fn decision_lcycle(&self, pp_id: i32) -> usize {
        let now = self.current_lcycle;
        let l = self.reaction_latency;
        let mut decision = now + l;
        if l == 0 {
            return decision;
        }
        for op in &self.input.circuit.product(pp_id).operators {
            for &(j, d) in self.t_decisions.get(op.qubit as usize).map_or(&[][..], |v| v) {
                // An earlier decision binds until its frame bits are known (d + L).
                if d + l > now && d + l > decision && self.anticommute(j, pp_id) {
                    decision = d + l;
                }
            }
        }
        decision
    }

    /// Records T product `pp_id`'s decision lcycle on its qubits for later T gates.
    fn record_decision(&mut self, pp_id: i32, decision: usize) {
        let (now, l) = (self.current_lcycle, self.reaction_latency);
        if l == 0 {
            return;
        }
        for i in 0..self.input.circuit.product(pp_id).operators.len() {
            let q = self.input.circuit.product(pp_id).operators[i].qubit as usize;
            if self.t_decisions.len() <= q {
                self.t_decisions.resize(q + 1, Vec::new());
            }
            // Entries whose frame bits are already known can no longer bind.
            self.t_decisions[q].retain(|&(_, d)| d + l > now);
            self.t_decisions[q].push((pp_id, decision));
        }
    }

    /// Rank of node `nb` as a |0> helper for a T injection, lower is better, or None if it can
    /// never be one (a data patch, a hosted patch, the route-only half of one). A free routing
    /// (bus) patch comes first, then the cultivating magic patch with the least progress (its
    /// cultivation restarts), and last, with `litinski-ready` only, a ready magic state, which
    /// is wasted. With magic routing (-u) there are no bus patches, so the helper is always a
    /// magic patch. Without the last resort (`litinski-y`, and the hybrid's helpers), the free
    /// neighbours of a magic patch are often all ready states, and injections wait for a helper
    /// or fall back to measuring the magic patch in place.
    fn helper_rank(&self, nb: u16) -> Option<(u8, i32)> {
        let topo = &self.input.topo;
        if topo.is_hosted(nb) {
            return None;
        }
        match topo.node(nb).node_type {
            NodeType::Bus => Some((0, 0)),
            NodeType::Magic if topo.cultivation_times[nb as usize] > 0 => {
                Some((1, topo.busy_counts[nb as usize]))
            }
            NodeType::Magic if self.t_correction == TCorrection::LitinskiReady => Some((2, 0)),
            _ => None,
        }
    }

    /// Occupies helper `h` (a neighbour of `magic`) until `decision` and logs it.
    fn take_helper(&mut self, magic: u16, h: u16, decision: usize) {
        info_sched!("    helper |0> patch {}", self.input.topo.label(h));
        if self.helper_rank(h).map_or(false, |r| r.0 == 2) {
            self.t_stats.helpers_ready += 1;
        }
        self.t_stats.helper_log.push((self.current_lcycle, magic, h));
        self.t_stats.helpers += 1;
        self.mark_nodes_used(&[h]);
        self.hold(h, decision);
    }

    /// A free neighbour of `magic` to borrow as a |0> helper (best `helper_rank`).
    fn pick_helper(&self, magic: u16) -> Option<u16> {
        let mut best: Option<((u8, i32), u16)> = None;
        for &nb in self.input.topo.node(magic).nbs_slice() {
            if self.used[nb as usize] {
                continue;
            }
            if let Some(rank) = self.helper_rank(nb) {
                if best.map_or(true, |(r, _)| rank < r) {
                    best = Some((rank, nb));
                }
            }
        }
        best.map(|(_, nb)| nb)
    }

    /// True if some neighbour of `magic` could ever serve as its helper.
    fn helper_possible(&self, magic: u16) -> bool {
        self.input.topo.node(magic).nbs_slice().iter().any(|&nb| self.helper_rank(nb).is_some())
    }

    /// Keeps patch `node` occupied through lcycle `last` (nothing to do if `last` is now).
    fn hold(&mut self, node: u16, last: usize) {
        let now = self.current_lcycle;
        if last > now {
            self.held.push((last, node));
            self.t_stats.hold_patch_lcycles += last - now;
        }
    }

    /// Pushes T product `pp_id`'s recorded decision back by `by` lcycles: its frame bit comes
    /// from a measurement that ends that much later.
    fn delay_decision(&mut self, pp_id: i32, by: usize) {
        for i in 0..self.input.circuit.product(pp_id).operators.len() {
            let q = self.input.circuit.product(pp_id).operators[i].qubit as usize;
            if let Some(v) = self.t_decisions.get_mut(q) {
                for e in v.iter_mut().filter(|e| e.0 == pp_id) {
                    e.1 += by;
                }
            }
        }
    }

    /// The magic patch of T product `pp_id` is measured in place once the correction is decided
    /// (lcycle `decision`): in X, or in Y if the T failed. `failed` is None while the injection
    /// outcome has not been drawn yet (the injection lcycle); `process_t_gate_outcomes` then
    /// adds the Y time if needed.
    fn measure_in_place(&mut self, pp_id: i32, magic: u16, decision: usize, failed: Option<bool>) {
        self.t_stats.inplace += 1;
        let y = self.y_meas_lcycles;
        match failed {
            Some(true) => {
                self.t_stats.inplace_y += 1;
                self.delay_decision(pp_id, y);
                self.hold(magic, decision + y);
            }
            Some(false) => self.hold(magic, decision),
            None => {
                self.hold(magic, decision);
                self.inplace_pending.insert(pp_id, (magic, decision));
            }
        }
    }

    /// Bookkeeping for a T product injected this lcycle from magic patch `magic`.
    ///
    /// Litinski: P (x) Z_m and Z_m (x) Y_h are measured together, then the magic patch in X, so
    /// the magic patch is free again after this lcycle. Only the helper h waits for the decision
    /// (it is measured in X to apply the correction, in Z not to).
    fn inject_t(&mut self, pp_id: i32, magic: u16) {
        let now = self.current_lcycle;
        let decision = self.decision_lcycle(pp_id);
        self.record_decision(pp_id, decision);
        self.t_stats.injections += 1;
        self.t_stats.injection_log.push((now, pp_id, magic, decision));
        self.t_stats.max_decision_delay = self.t_stats.max_decision_delay.max(decision - now);
        info_sched!(
            "  T {} injected from magic {}; correction decided in lcycle {}",
            pp_id,
            self.input.topo.label(magic),
            decision
        );
        if self.t_correction == TCorrection::InplaceY {
            self.measure_in_place(pp_id, magic, decision, None);
            return;
        }
        match self.pick_helper(magic) {
            Some(h) => self.take_helper(magic, h, decision),
            None if self.helper_possible(magic) => {
                // The magic patch keeps its state until a helper is free (hold_t_patches).
                info_sched!("    no free neighbour for the helper yet");
                self.t_stats.helpers_deferred += 1;
                self.helper_wanted.push((pp_id, magic, decision, now + HELPER_MAX_WAIT));
            }
            None => {
                info_sched!("    no neighbour can be a helper; magic patch measured in place");
                self.t_stats.helpers_missing += 1;
                self.measure_in_place(pp_id, magic, decision, None);
            }
        }
    }

    /// Start of an lcycle: release expired holds, mark the rest used, and give injections that
    /// are still waiting for a |0> helper the first pick of free neighbours.
    fn hold_t_patches(&mut self) {
        let now = self.current_lcycle;
        self.held.retain(|&(last, _)| last >= now);
        let nodes: Vec<u16> = self.held.iter().map(|&(_, n)| n).collect();
        self.holding_now = !nodes.is_empty() || !self.helper_wanted.is_empty();
        if self.holding_now {
            info_sched!(
                "  held patches {:?}; waiting for a helper: {:?}",
                nodes.iter().map(|&n| self.input.topo.label(n)).collect::<Vec<_>>(),
                self.helper_wanted
                    .iter()
                    .map(|&(_, m, _, _)| self.input.topo.label(m))
                    .collect::<Vec<_>>()
            );
        }
        self.mark_nodes_used(&nodes);
        let wanted = std::mem::take(&mut self.helper_wanted);
        // Every waiting magic patch still holds its state: occupy them all before any helper is
        // picked, so no waiting patch is borrowed as another one's helper.
        for &(_, magic, _, _) in &wanted {
            self.mark_nodes_used(&[magic]);
            self.t_stats.hold_patch_lcycles += 1;
        }
        for (pp_id, magic, decision, give_up) in wanted {
            match self.pick_helper(magic) {
                Some(h) => {
                    // Z_m (x) Y_h and X_m now; the magic patch is free from the next lcycle.
                    self.take_helper(magic, h, decision);
                    self.failed_waiting.remove(&pp_id);
                }
                None if now < give_up => self.helper_wanted.push((pp_id, magic, decision, give_up)),
                None => {
                    // Waiting longer could deadlock (two magic patches each waiting on the
                    // other); measure the magic patch in place instead.
                    info_sched!(
                        "  no helper for {}; measured in place",
                        self.input.topo.label(magic)
                    );
                    self.t_stats.helpers_missing += 1;
                    let failed = self.failed_waiting.remove(&pp_id);
                    self.measure_in_place(pp_id, magic, decision, Some(failed));
                }
            }
        }
    }

    /// Sets how conditional products are decided (default: `Random` with `p1` = 0.5).
    pub(crate) fn set_cond_outcomes(&mut self, mode: CondOutcomes, p1: f64) {
        assert!((0.0..=1.0).contains(&p1), "outcome probability {} is not in [0, 1]", p1);
        self.cond_outcomes = mode;
        self.cond_p1 = p1;
    }

    /// Number of products with a condition.
    pub(crate) fn n_conditional(&self) -> usize {
        self.input.circuit.pps.iter().filter(|pp| !pp.cond.is_empty()).count()
    }

    /// The true outcome of measurement product `m`, sampled the first time it is asked for.
    fn outcome(&mut self, m: i32) -> bool {
        if let Some(&o) = self.outcomes.get(&m) {
            return o;
        }
        let o = self.rng_outcomes.gen_bool(self.cond_p1);
        self.outcomes.insert(m, o);
        o
    }

    /// True if product `id` runs: it has no condition, conditions are ignored (`Worst`), or
    /// every term holds. Only called once the product is ready, so every outcome it names has
    /// been measured (a condition's products are always in its `@` list).
    fn cond_holds(&mut self, id: i32) -> bool {
        if self.cond_outcomes == CondOutcomes::Worst {
            return true;
        }
        let cond = self.input.circuit.product(id).cond.clone();
        // Sample every term (no short-circuit), so the outcomes drawn do not depend on order.
        let values: Vec<bool> = cond.iter().map(|&(m, _)| self.outcome(m)).collect();
        cond.iter().zip(values).all(|(&(_, want), got)| got == want)
    }

    /// Filters products that just became ready. A conditional product whose condition fails
    /// is skipped: it completes at once without being scheduled (no lcycle, no magic state, no
    /// T-failure draw, no S correction), and its children are unlocked and admitted in turn.
    /// Returns, in order, the products that should run.
    fn admit(&mut self, ready: Vec<i32>) -> Vec<i32> {
        let mut run = Vec::with_capacity(ready.len());
        let mut work: std::collections::VecDeque<i32> = ready.into();
        while let Some(id) = work.pop_front() {
            if self.input.circuit.product(id).cond.is_empty() || self.cond_holds(id) {
                run.push(id);
                continue;
            }
            info_sched!("  conditional product {} skipped: condition false", id);
            self.skipped.push(id);
            self.skipped_set.insert(id);
            if self.input.circuit.product(id).gate_type.is_t() {
                self.n_skipped_t += 1;
                self.cultivation.t_products_remaining =
                    self.cultivation.t_products_remaining.saturating_sub(1);
            }
            if !self.hosted.live.is_empty() {
                self.hosted.completed_buf.push(id);
            }
            // A skipped product has no outcome, so no child waits on it through the reaction
            // latency (only measurements are named by `@`, and a measurement is never
            // conditional); every edge from it releases at once.
            let children = self.input.circuit.product(id).children.clone();
            for child in children {
                self.remaining_parents[child as usize] -= 1;
                if self.remaining_parents[child as usize] == 0 {
                    work.push_back(child);
                }
            }
        }
        run
    }

    /// Sets the classical reaction latency (lcycles) applied to `@`-dependency edges.
    pub(crate) fn set_reaction_latency(&mut self, lcycles: usize) {
        self.reaction_latency = lcycles;
    }

    /// Releases every delayed classical unlock whose latency has elapsed, making children whose
    /// last outstanding parent it was available to the scheduler this lcycle.
    fn release_delayed_unlocks(&mut self) {
        if self.delayed_unlocks.is_empty() {
            return;
        }
        let now = self.current_lcycle;
        let mut still_waiting = Vec::with_capacity(self.delayed_unlocks.len());
        let mut ready = Vec::new();
        for (due, child_id) in std::mem::take(&mut self.delayed_unlocks) {
            if due >= now {
                still_waiting.push((due, child_id));
                continue;
            }
            self.remaining_parents[child_id as usize] -= 1;
            if self.remaining_parents[child_id as usize] == 0 {
                ready.push(child_id);
            }
        }
        self.delayed_unlocks = still_waiting;
        for child_id in self.admit(ready) {
            let child = self.input.circuit.product(child_id).clone();
            self.pps_pending.push(child);
            if self.hosting() {
                self.activate_hosted_for(&[child_id]);
            }
        }
    }

    pub(crate) fn hosted_live_lcycles(&self, q: u16) -> usize {
        self.hosted.live_lcycles.get(q as usize).copied().unwrap_or(0)
    }

    pub(crate) fn hosted_activations(&self, q: u16) -> usize {
        self.hosted.activations.get(q as usize).copied().unwrap_or(0)
    }

    /// Debug check of one lcycle: a product on a hosted qubit runs only while it is live,
    /// and a live hosted patch is used only by products on that qubit (scheduled, carried
    /// forward, or blocked and holding its terminals).
    #[cfg(debug_assertions)]
    fn check_hosted_lcycle(&self) -> io::Result<()> {
        let err = |m: String| Err(io::Error::new(io::ErrorKind::Other, m));
        let touches = |pp: &PauliProduct, q: u16| pp.operators.iter().any(|op| op.qubit == q);
        for &q in &self.input.topo.hosted_qubits {
            let live = self.hosted.live[q as usize];
            let scheduled_on_q =
                self.pp_paths.iter().any(|(id, _)| touches(self.input.circuit.product(*id), q));
            if scheduled_on_q && !live {
                return err(format!(
                    "lcycle {}: product on hosted q{} scheduled while dormant",
                    self.current_lcycle, q
                ));
            }
            let [x_id, z_id] = self.input.topo.data_ids[q as usize];
            if live && (self.used[x_id as usize] || self.used[z_id as usize]) {
                let pending_on_q = self.pps_pending.iter().any(|pp| touches(pp, q));
                if !scheduled_on_q && !pending_on_q {
                    return err(format!(
                        "lcycle {}: live hosted q{}'s patch used by another product",
                        self.current_lcycle, q
                    ));
                }
            }
            if live
                && self.cultivation.ready_magic_positions.contains(&self.input.topo.node(x_id).pos)
            {
                return err(format!(
                    "lcycle {}: live hosted q{} counted as a ready magic state",
                    self.current_lcycle, q
                ));
            }
        }
        Ok(())
    }

    fn hosting(&self) -> bool {
        !self.input.topo.hosted_qubits.is_empty()
    }

    /// Counts each hosted qubit's products and makes every hosted qubit dormant. Runs after
    /// the terminal/root/Clifford-tree precomputation, which therefore saw hosted nodes as
    /// Data: no precomputed tree or root ever relies on a hosted node.
    fn init_hosted(&mut self) {
        let n_products = self.input.circuit.n_products();
        let hosted_qubits = self.input.topo.hosted_qubits.clone();
        // A hosted qubit no product touches can lie past the circuit's highest used qubit.
        let n_qubits = hosted_qubits
            .iter()
            .map(|&q| q as usize + 1)
            .fold(self.input.circuit.n_qubits, usize::max);
        let h = &mut self.hosted;
        h.n_initial_products = n_products;
        h.remaining = vec![0; n_qubits];
        h.chain = vec![Vec::new(); n_qubits];
        h.done = vec![0; n_qubits];
        h.live = vec![false; n_qubits];
        h.live_lcycles = vec![0; n_qubits];
        h.activations = vec![0; n_qubits];
        for id in 0..n_products {
            for op in &self.input.circuit.product(id as i32).operators {
                if hosted_qubits.contains(&op.qubit) {
                    h.remaining[op.qubit as usize] += 1;
                    h.chain[op.qubit as usize].push(id as i32);
                }
            }
        }
        for &q in &hosted_qubits {
            self.input.topo.set_hosted_live(q, false);
        }
    }

    /// Route-only halves never cultivate. Called after `init_magic_nodes`.
    fn mark_route_only_nodes(&mut self) {
        for &q in &self.input.topo.hosted_qubits.clone() {
            let z_id = self.input.topo.data_ids[q as usize][1];
            self.input.topo.cultivation_times[z_id as usize] = NOT_CULTIVATING;
            self.input.topo.busy_counts[z_id as usize] = 0;
        }
    }

    /// Makes dormant hosted qubits touched by products `ids` live: their nodes become
    /// Data and any cultivation in progress on them is discarded. Runs between lcycles,
    /// as soon as such a product is ready, so a hosted node is Data whenever a product
    /// could use it as a terminal.
    fn activate_hosted_for(&mut self, ids: &[i32]) {
        for &id in ids {
            let qubits: Vec<u16> =
                self.input.circuit.product(id).operators.iter().map(|op| op.qubit).collect();
            for q in qubits {
                if !self.input.topo.is_hosted(self.input.topo.data_ids[q as usize][0])
                    || self.hosted.live[q as usize]
                {
                    continue;
                }
                self.hosted.live[q as usize] = true;
                self.hosted.activations[q as usize] += 1;
                self.input.topo.set_hosted_live(q, true);
                for node_id in self.input.topo.data_ids[q as usize] {
                    self.input.topo.cultivation_times[node_id as usize] = NOT_CULTIVATING;
                    self.input.topo.busy_counts[node_id as usize] = 0;
                }
                info_sched!("  hosted ancilla q{} live", q);
            }
        }
    }

    /// True if product `pp`, once complete, leaves patch `q` in a known state that a fresh
    /// patch can reproduce: a single-patch X or Z measurement (re-prepare the measured
    /// eigenstate; the outcome goes to the Pauli frame) or a re-initialization `<R>`.
    /// A Y measurement does not count: preparing a Y eigenstate needs an S gate.
    fn releases_patch(pp: &PauliProduct, q: u16) -> bool {
        match pp.operators.as_slice() {
            [op] if op.qubit == q => {
                pp.gate_type.is_r() || (pp.gate_type.is_m() && matches!(op.basis, 'X' | 'Z'))
            }
            _ => false,
        }
    }

    /// Makes hosted qubits dormant after products completed this lcycle: after the qubit's
    /// last product, or after a product that releases the patch (`releases_patch`) when the
    /// next product on the qubit is not ready yet. The nodes become Magic, the X node starts
    /// a fresh cultivation, and any pending S correction is dropped (the state it belonged
    /// to is gone; the patch is re-prepared when the qubit goes live again).
    fn retire_hosted(&mut self) {
        let completed = std::mem::take(&mut self.hosted.completed_buf);
        for &id in &completed {
            if id as usize >= self.hosted.n_initial_products {
                continue; // a scheduler-added S correction, not part of any chain
            }
            let qubits: Vec<u16> =
                self.input.circuit.product(id).operators.iter().map(|op| op.qubit).collect();
            for q in qubits {
                let qi = q as usize;
                if !self.hosted.live.get(qi).copied().unwrap_or(false) {
                    // A skipped product on a dormant qubit: it is done, but there was no work,
                    // so the patch stays dormant.
                    if self.skipped_set.contains(&id) && qi < self.hosted.done.len() {
                        debug_assert_eq!(
                            self.hosted.chain[qi].get(self.hosted.done[qi]).copied(),
                            Some(id),
                            "hosted q{}: skipped product out of program order",
                            q
                        );
                        self.hosted.done[qi] += 1;
                        self.hosted.remaining[qi] -= 1;
                    }
                    continue;
                }
                debug_assert_eq!(
                    self.hosted.chain[qi].get(self.hosted.done[qi]).copied(),
                    Some(id),
                    "hosted q{}: products completed out of program order",
                    q
                );
                self.hosted.done[qi] += 1;
                self.hosted.remaining[qi] -= 1;
                let dormant = match self.hosted.chain[qi].get(self.hosted.done[qi]) {
                    None => true,
                    Some(&next) => {
                        self.releases_patch_now(id, q)
                            && !self.pps_pending.iter().any(|pp| pp.id == next)
                    }
                };
                if !dormant {
                    continue;
                }
                let [x_id, z_id] = self.input.topo.data_ids[qi];
                debug_assert!(
                    self.clifford_paths.values().all(|(_, _, nodes, _)| {
                        !nodes.contains(&x_id) && !nodes.contains(&z_id)
                    }),
                    "hosted q{} retired while a Clifford still holds its patch",
                    q
                );
                self.hosted.live[qi] = false;
                self.input.topo.set_hosted_live(q, false);
                self.cultivation.restart(&mut self.input.topo, x_id);
                self.input.topo.cultivation_times[z_id as usize] = NOT_CULTIVATING;
                self.correction_power[qi] = 0;
                info_sched!("  hosted ancilla q{} dormant", q);
            }
        }
        self.hosted.completed_buf = completed;
        self.hosted.completed_buf.clear();
    }

    pub(crate) fn count_t_products(&self) -> usize {
        (0..self.input.circuit.n_products())
            .filter(|&id| self.input.circuit.product(id as i32).gate_type.is_t())
            .count()
    }

    /// Greedily assigns products to lcycles. Returns (total lcycles, total scheduled products).
    pub(crate) fn sched_circuit(&mut self) -> io::Result<(usize, usize)> {
        let _timer = fn_timer!();
        self.cultivation
            .set_lambda(self.magic_state_lambda)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        let hosting = self.hosting();
        if matches!(self.t_correction, TCorrection::Frame | TCorrection::Hybrid) {
            if hosting {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "--t-correction frame/hybrid does not support --ancilla-qubits: a hosted \
                     patch cannot be released while the frame still acts on it",
                ));
            }
            if !self.no_t_failures {
                self.frame = Some(CliffordFrame::new(self.input.circuit.n_qubits));
            }
        }
        if hosting {
            // Precompute while hosted nodes are still Data, then make them dormant (Magic)
            // so `init_magic_nodes` picks up their cultivators.
            self.precompute_terminals_and_roots();
            self.precompute_multi_term_clifford_trees();
            self.init_hosted();
        }
        self.init_magic_nodes();
        if hosting {
            self.mark_route_only_nodes();
        }
        let n_t_products = self.count_t_products();
        self.cultivation.t_products_remaining = n_t_products;
        // Pre-generate a large pool: ~120 samples per T gate plus one per topology node.
        self.cultivation.fill_pool(120 * n_t_products.max(1) + self.input.topo.n_nodes);
        if !hosting {
            self.precompute_terminals_and_roots();
            self.precompute_multi_term_clifford_trees();
        }
        self.pps_pending = self.input.circuit.initial_products().cloned().collect();
        if hosting {
            let initial: Vec<i32> = self.pps_pending.iter().map(|pp| pp.id).collect();
            self.activate_hosted_for(&initial);
        }
        self.remaining_parents = (0..self.input.circuit.n_products())
            .map(|id| self.input.circuit.product(id as i32).parents.len())
            .collect();
        debug_sched!("Initial pps_to_sched len {}", self.pps_pending.len());
        let mut plot_lcycles = 0usize;
        let mut path_dir: Option<String> = None;
        if self.plot_option.contains("paths") {
            let dir_name = format!("{}.paths", circuit_stem(&self.input.circuit.circuit_fname));
            std::fs::create_dir_all(&dir_name)?;
            path_dir = Some(dir_name);
            plot_lcycles = 30;
        }
        let plotting = path_dir.is_some();
        let tot_pps_to_sched = self.input.circuit.n_products();
        let mut prev_pct_complete = 0usize;
        self.current_lcycle = 0;
        self.pp_paths = Vec::new();
        while !self.pps_pending.is_empty()
            || !self.clifford_paths.is_empty()
            || !self.delayed_unlocks.is_empty()
        {
            self.timers.start(self.loop_timer);
            self.current_lcycle += 1;
            self.release_delayed_unlocks();
            for (q, &live) in self.hosted.live.iter().enumerate() {
                if live {
                    self.hosted.live_lcycles[q] += 1;
                }
            }
            info_sched!(
                "{}",
                format!(
                    "lcycle {}: {:?}",
                    self.current_lcycle,
                    self.pps_pending
                        .iter()
                        .map(|pp| format!("{}:{}", pp.id, pp.to_operator_str()))
                        .collect::<Vec<_>>(),
                )
                .cyan()
            );
            if self.schedule_lcycle(plotting) {
                self.complete_lcycle()?;
                let n_scheduled = self.scheduled_products.len();
                // Show a progress bar during the non-plotting middle section of the run.
                // The first `plot_lcycles` and last `plot_lcycles` are plotted instead.
                if self.current_lcycle >= plot_lcycles
                    && n_scheduled + plot_lcycles <= tot_pps_to_sched
                {
                    if self.current_lcycle == plot_lcycles {
                        print!("Scheduling {} products:    ", tot_pps_to_sched);
                    }
                    let pct_complete = ((n_scheduled * 100) / tot_pps_to_sched).min(100);
                    if pct_complete > prev_pct_complete {
                        // \x08 is backspace; overwrite the previous "XX%" in-place.
                        print!("\x08\x08\x08{:02}%", pct_complete);
                        std::io::stdout().flush()?;
                        prev_pct_complete = pct_complete;
                    }
                    if n_scheduled + plot_lcycles == tot_pps_to_sched {
                        print!("\n");
                    }
                } else if plotting {
                    let plot_info_str = self.stats.plot_info_str();
                    assert!(!plot_info_str.is_empty());
                    let fname_added = format!(".{}", self.current_lcycle);
                    let curr_dir = std::env::current_dir()?;
                    std::env::set_current_dir(path_dir.as_ref().unwrap())?;
                    let pp_paths_plot: Vec<(PauliProduct, Rc<TreeGraph>, u32)> = self
                        .pp_paths
                        .iter()
                        .filter_map(|(pp_id, opt_tree)| {
                            opt_tree.as_ref().map(|t| {
                                let pp = self.input.circuit.product(*pp_id);
                                // Determine which cycle of a multi-lcycle product this is;
                                // clifford_paths already reflects post-advance_clifford_state.
                                let cycle: u32 = if pp.gate_type.is_clifford() {
                                    match self.clifford_paths.get(pp_id) {
                                        Some((c, _, _, _)) => {
                                            let initial =
                                                if pp.gate_type.is_cx() { 1u32 } else { 2u32 };
                                            initial - *c as u32 + 1
                                        }
                                        None => {
                                            if pp.gate_type.is_cx() {
                                                2
                                            } else {
                                                3
                                            }
                                        }
                                    }
                                } else if pp.gate_type.is_t() {
                                    1
                                } else {
                                    1
                                };
                                (pp.clone(), Rc::clone(t), cycle)
                            })
                        })
                        .collect();
                    self.input
                        .topo
                        .plot(&fname_added, &pp_paths_plot, &plot_info_str)
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    std::env::set_current_dir(curr_dir)?;
                }
            } else {
                debug_sched!("Could not schedule anything on lcycle {}", self.current_lcycle);
                // Only error if T gates are pending: pending Cliffords may just be waiting on a
                // temporarily-blocked correction S gate and don't need magic to make progress.
                let has_pending_t = self.pps_pending.iter().any(|pp| pp.gate_type.is_t());
                let holds_outstanding =
                    self.holding_now || !self.held.is_empty() || !self.helper_wanted.is_empty();
                if has_pending_t
                    && !holds_outstanding
                    && !(0..self.input.topo.n_nodes)
                        .any(|node_i| self.input.topo.is_cultivating(node_i as u16))
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Cannot schedule on current layout".red().to_string(),
                    ));
                }
            }
            self.timers.stop(self.loop_timer);
        }
        self.print_sched_stats(self.current_lcycle);
        #[cfg(debug_assertions)]
        self.check_clifford_repetitions()?;
        #[cfg(debug_assertions)]
        self.check_schedule()?;
        Ok((self.current_lcycle, self.scheduled_products.len()))
    }

    fn init_magic_nodes(&mut self) {
        self.cultivation.init_magic_nodes(&mut self.input.topo);
    }

    /// Multi-terminal non-T products have topology-independent routing trees that
    /// can be computed once on an empty grid and reused every lcycle.
    fn should_precompute(pp: &PauliProduct) -> bool {
        !pp.gate_type.is_t() && pp.operators.len() > 1
    }

    fn precompute_steiner_tree(&mut self, pp: &PauliProduct) -> Option<TreeGraph> {
        if !self.terminal_nodes(pp.id) {
            return None;
        }
        let root_ids = self.root_nodes(pp.id as usize, &self.terminals_buf[..]);
        if root_ids.is_empty() {
            return None;
        }
        self.stree_computation.compute(
            &self.input.topo,
            &self.used,
            &root_ids,
            &self.terminals_buf,
            pp.gate_type,
        )
    }

    /// Fills `terminals_buf` from precomputed IDs. Returns false early if any terminal
    /// is occupied or all its root candidates are occupied (cheap pre-check before `root_nodes`).
    #[inline]
    fn terminal_nodes(&mut self, pp_id: i32) -> bool {
        let pp_id = pp_id as usize;
        self.terminals_buf.clear();
        let terminals = &self.precomputed_terminals[pp_id];
        let root_info = &self.precomputed_root_info[pp_id];
        for (i, &node_id) in terminals.iter().enumerate() {
            if self.used[node_id as usize] {
                info_sched!("  Node {} is already used", node_id);
                return false;
            }
            let (_, preferred, side) = &root_info[i];
            // Only reject when candidates exist and all are occupied; empty preferred+side
            // means the gate uses sched_s_sx (no root needed), so don't reject vacuously.
            if (!preferred.is_empty() || !side.is_empty())
                && preferred.iter().all(|&rid| self.used[rid as usize])
                && side.iter().all(|&rid| self.used[rid as usize])
            {
                info_sched!("  No unused root candidates for node {}", node_id);
                return false;
            }
            self.terminals_buf.push(node_id);
        }
        true
    }

    /// Returns routing nodes adjacent to each terminal.
    /// For Y-basis operators the paired X/Z data nodes can share a single routing node
    /// (the "preferred" root), decrementing `unmatched_count` by 2; if none is free,
    /// each terminal falls back to a side nb.
    fn root_nodes(&self, pp_id: usize, terminals: &[u16]) -> Vec<u16> {
        let root_info = &self.precomputed_root_info[pp_id];
        let mut root_ids: Vec<u16> = Vec::new();
        let mut unmatched_count: usize = terminals.len();
        for (i, _node_id) in terminals.iter().enumerate() {
            let (is_paired, preferred, side) = &root_info[i];
            let mut pair_found = false;
            if *is_paired {
                for &nb_id in preferred {
                    if self.used[nb_id as usize] {
                        continue;
                    }
                    if !root_ids.contains(&nb_id) {
                        root_ids.push(nb_id);
                    }
                    // One root serves both X and Z terminals of the Y pair.
                    unmatched_count = unmatched_count.saturating_sub(2);
                    pair_found = true;
                    break;
                }
            }
            if !pair_found {
                let fallback = if *is_paired { side.as_slice() } else { preferred.as_slice() };
                for &nb_id in fallback {
                    if self.used[nb_id as usize] {
                        continue;
                    }
                    if !root_ids.contains(&nb_id) {
                        root_ids.push(nb_id);
                    }
                    unmatched_count = unmatched_count.saturating_sub(1);
                    break;
                }
            }
        }
        if unmatched_count > 0 {
            debug_sched!(
                "    could not find root nodes for {} unmatched terminals",
                unmatched_count
            );
            return Vec::new();
        }
        root_ids
    }

    fn precompute_multi_term_clifford_trees(&mut self) {
        let _timer = fn_timer!("precompute_clifford_trees");
        self.used.fill(false);
        let n_products = self.input.circuit.n_products();
        let mut n_precomputed = 0;
        for pp_id in 0..n_products {
            let pp = self.input.circuit.product(pp_id as i32).clone();
            if Self::should_precompute(&pp) {
                if let Some(tree) = self.precompute_steiner_tree(&pp) {
                    self.precomputed_clifford_trees.insert(pp.id, Rc::new(tree));
                    n_precomputed += 1;
                } else {
                    eprintln!(
                        "{}",
                        format!("Warning: failed to precompute tree for {}", pp).yellow()
                    );
                }
            }
        }
        println!("Precomputed {} multi-term Clifford trees", n_precomputed);
    }

    /// Computes root candidate info for a set of already-computed terminal node IDs.
    /// Extracted from `precompute_terminals_and_roots` so it can be called for dynamically
    /// constructed operator lists (e.g. after S^k conjugation of T gate operators).
    fn compute_root_info_for_terminals(
        topo: &TopoGraph, terminals: &[u16],
    ) -> Vec<(bool, Vec<u16>, Vec<u16>)> {
        let mut root_info = Vec::with_capacity(terminals.len());
        for &term_id in terminals {
            let node = topo.node(term_id);
            let is_paired =
                node.paired_data_id.map(|pid| terminals.contains(&pid)).unwrap_or(false);
            let mut preferred = Vec::new();
            let mut side = Vec::new();
            if is_paired {
                let is_x = topo.label(term_id).contains('X');
                for &nb_id in node.nbs_slice() {
                    let nb = topo.node(nb_id);
                    // A dormant hosted node routes, but can go live at any time: never a root.
                    if !nb.is_routing() || topo.is_hosted(nb_id) {
                        continue;
                    }
                    if (is_x && nb.pos.1 < node.pos.1) || (!is_x && nb.pos.1 > node.pos.1) {
                        preferred.push(nb_id);
                    } else if nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                        side.push(nb_id);
                    }
                }
            } else {
                for &nb_id in node.nbs_slice() {
                    let nb = topo.node(nb_id);
                    if nb.is_routing()
                        && !topo.is_hosted(nb_id)
                        && nb.pos.0 != node.pos.0
                        && nb.pos.1 == node.pos.1
                    {
                        preferred.push(nb_id);
                    }
                }
            }
            root_info.push((is_paired, preferred, side));
        }
        root_info
    }

    /// Conjugates a Pauli label by S_axis (sign dropped, matching this codebase's existing
    /// no-phase-tracking convention: `PauliProduct`/`Operator` carry no sign field). A label
    /// equal to `axis` is fixed; any other label is swapped with the remaining third label,
    /// e.g. S_Z fixes Z and swaps X<->Y, S_X fixes X and swaps Y<->Z, S_Y fixes Y and swaps X<->Z.
    fn conjugate_pauli_label(basis: char, axis: char) -> char {
        match (basis, axis) {
            (b, a) if b == a => b,
            ('X', 'Y') | ('Y', 'X') => 'Z',
            ('X', 'Z') | ('Z', 'X') => 'Y',
            ('Y', 'Z') | ('Z', 'Y') => 'X',
            _ => basis,
        }
    }

    /// Applies the current per-qubit S^k conjugation to a T gate's operators, updating
    /// `precomputed_terminals`/`precomputed_root_info` and returning any (qubit, new_basis) swaps.
    /// Always recomputes from the circuit's original operators so changing correction_power
    /// between lcycles is reflected correctly.
    /// True if product `pp_id`'s basis must be (re)computed against the pending S corrections:
    /// one of its qubits has an odd correction power, or it was conjugated before and the
    /// correction has since been resolved.
    fn conjugation_due(&self, pp_id: i32) -> bool {
        self.input
            .circuit
            .product(pp_id)
            .operators
            .iter()
            .any(|op| self.correction_power[op.qubit as usize] % 2 == 1)
            || self.conjugated_t_products.contains(&pp_id)
    }

    /// Re-derives product `pp_id`'s terminals from the pending corrections (`apply_t_conjugation`)
    /// and records whether it is now conjugated. Returns the qubits whose basis changed.
    fn refresh_conjugation(&mut self, pp_id: i32) -> Vec<(u16, char)> {
        let swapped = self.apply_t_conjugation(pp_id);
        if swapped.is_empty() {
            self.conjugated_t_products.remove(&pp_id);
        } else {
            self.conjugated_t_products.insert(pp_id);
        }
        swapped
    }

    /// True if product `id`, once complete, leaves patch `q` in a state a fresh patch can
    /// reproduce (`releases_patch`), judged in the basis it was actually measured in: a pending
    /// odd S correction on `q` turns an X measurement into Y, which does not release the patch.
    fn releases_patch_now(&self, id: i32, q: u16) -> bool {
        let pp = self.input.circuit.product(id);
        if !Self::releases_patch(pp, q) {
            return false;
        }
        if pp.gate_type.is_r() || self.correction_power[q as usize] % 2 == 0 {
            return true;
        }
        let basis =
            Self::conjugate_pauli_label(pp.operators[0].basis, self.correction_basis[q as usize]);
        matches!(basis, 'X' | 'Z')
    }

    /// A re-initialization `<R>` replaces the patch's state, so any S correction still owed to
    /// the old state is void.
    fn clear_corrections_on_reinit(&mut self) {
        for i in 0..self.pp_paths.len() {
            let pp = self.input.circuit.product(self.pp_paths[i].0);
            if pp.gate_type.is_r() {
                let q = pp.operators[0].qubit as usize;
                self.correction_power[q] = 0;
            }
        }
    }

    fn apply_t_conjugation(&mut self, pp_id: i32) -> Vec<(u16, char)> {
        // Copy original operator data to release the borrow on self.input.circuit.
        let orig_ops: Vec<(u16, char)> = self
            .input
            .circuit
            .product(pp_id)
            .operators
            .iter()
            .map(|op| (op.qubit, op.basis))
            .collect();
        // Apply S^k conjugation in a single pass: odd power conjugates by the pending
        // correction's own axis (correction_basis), not always Z - a failed T gate's
        // correction is S about whichever axis that T gate itself targeted.
        let mut new_ops = Vec::with_capacity(orig_ops.len());
        let mut swapped = Vec::new();
        for &(qubit, basis) in &orig_ops {
            let new_basis = if self.correction_power[qubit as usize] % 2 == 1 {
                Self::conjugate_pauli_label(basis, self.correction_basis[qubit as usize])
            } else {
                basis
            };
            if new_basis != basis {
                swapped.push((qubit, new_basis));
            }
            new_ops.push(Operator { qubit, basis: new_basis });
        }
        #[cfg(debug_assertions)]
        for &(q, new_basis) in &swapped {
            info_sched!(
                "  T gate {}: qubit {} conjugated to {} (S^{} correction active)",
                pp_id,
                q,
                new_basis,
                self.correction_power[q as usize]
            );
        }
        let terminals = operators_to_node_ids(&self.input.topo, &new_ops);
        let root_info = Self::compute_root_info_for_terminals(&self.input.topo, &terminals);
        self.precomputed_terminals[pp_id as usize] = terminals;
        self.precomputed_root_info[pp_id as usize] = root_info;
        swapped
    }

    /// Precomputes terminal node IDs and root candidates for every product.
    ///
    /// For each terminal data node the root candidates are split into:
    /// - `preferred`: the routing node between the X and Z patches of a Y-pair
    ///   (X looks downward toward Z; Z looks upward toward X), or the same-row
    ///   side nb for unpaired terminals.
    /// - `side`: same-row side nbs used as fallback for Y-paired terminals.
    fn precompute_terminals_and_roots(&mut self) {
        let _timer = fn_timer!("precompute_terminals_and_roots");
        let n_products = self.input.circuit.n_products();
        self.precomputed_terminals = vec![Vec::new(); n_products];
        self.precomputed_root_info = vec![Vec::new(); n_products];
        for pp_id in 0..n_products {
            let pp = self.input.circuit.product(pp_id as i32).clone();
            let terminals = operators_to_node_ids(&self.input.topo, &pp.operators);
            let mut root_info: Vec<(bool, Vec<u16>, Vec<u16>)> =
                Vec::with_capacity(terminals.len());
            for &term_id in &terminals {
                let node = self.input.topo.node(term_id);
                // is_paired: the operator is Y-basis, so both its X and Z data nodes are terminals.
                let is_paired =
                    node.paired_data_id.map(|pid| terminals.contains(&pid)).unwrap_or(false);
                let mut preferred: Vec<u16> = Vec::new();
                let mut side: Vec<u16> = Vec::new();
                if is_paired {
                    // X nodes look downward (toward paired Z), Z nodes look upward.
                    let is_x = self.input.topo.label(term_id).contains('X');
                    for &nb_id in node.nbs_slice() {
                        let nb = self.input.topo.node(nb_id);
                        if !nb.is_routing() || self.input.topo.is_hosted(nb_id) {
                            continue;
                        }
                        if (is_x && nb.pos.1 < node.pos.1) || (!is_x && nb.pos.1 > node.pos.1) {
                            preferred.push(nb_id);
                        } else if nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                            side.push(nb_id);
                        }
                    }
                } else {
                    // Unpaired terminal: only same-row side nbs are valid roots.
                    for &nb_id in node.nbs_slice() {
                        let nb = self.input.topo.node(nb_id);
                        if nb.is_routing()
                            && !self.input.topo.is_hosted(nb_id)
                            && nb.pos.0 != node.pos.0
                            && nb.pos.1 == node.pos.1
                        {
                            preferred.push(nb_id);
                        }
                    }
                }
                root_info.push((is_paired, preferred, side));
            }
            self.precomputed_terminals[pp_id] = terminals;
            self.precomputed_root_info[pp_id] = root_info;
        }
        println!("Precomputed terminals and root candidates for {} products", n_products);
    }

    fn mark_nodes_used(&mut self, node_ids: &[u16]) {
        for &node_id in node_ids {
            self.used[node_id as usize] = true;
            let node = self.input.topo.node(node_id);
            self.stats.inc_with_cultivation(
                node.node_type,
                self.input.topo.cultivation_times[node_id as usize],
            );
        }
    }

    /// Reserves the nodes of an in-progress multi-lcycle product for this lcycle.
    fn carry_forward_path(
        &mut self, pp_id: i32, node_ids: &[u16], opt_tree: Option<Rc<TreeGraph>>,
    ) {
        self.mark_nodes_used(node_ids);
        self.pp_paths.push((pp_id, opt_tree));
    }

    /// Schedules as many products as possible in one lcycle; returns false if nothing scheduled.
    fn schedule_lcycle(&mut self, plotting: bool) -> bool {
        let _timer = accum_start!(self.timers);
        self.timers.start(self.other_timer);
        let mut n_avail_magic = self.update_cultivators();
        let initial_magic = n_avail_magic;
        self.pp_paths.clear();
        self.used.fill(false);
        // Collect carry-forward data before the loop to release borrows on the map.
        let clifford_carry: Vec<(i32, Vec<u16>, Option<Rc<TreeGraph>>)> = self
            .clifford_paths
            .values()
            .map(|(_, pp, node_ids, opt_tree)| {
                (pp.id, node_ids.clone(), opt_tree.as_ref().map(Rc::clone))
            })
            .collect();
        for (pp_id, node_ids, opt_tree) in clifford_carry {
            self.carry_forward_path(pp_id, &node_ids, opt_tree);
        }
        if matches!(
            self.t_correction,
            TCorrection::LitinskiReady
                | TCorrection::LitinskiY
                | TCorrection::InplaceY
                | TCorrection::Hybrid
        ) {
            self.hold_t_patches();
        }
        if self.frame.is_some() {
            if self.t_correction == TCorrection::Frame {
                self.holding_now = false; // hold_t_patches resets it in the other modes
            }
            self.frame_begin_lcycle();
        }
        let carry_forward_count = self.clifford_paths.len();
        let tot_available = carry_forward_count + self.pps_pending.len();
        info_sched!("  Remaining to schedule: {}", self.pps_pending.len());
        self.sched_precomputed(plotting);
        self.timers.stop(self.other_timer);
        self.sched_remaining(&mut n_avail_magic, plotting);
        self.stats.update(
            self.current_lcycle,
            self.pp_paths.len(),
            tot_available,
            initial_magic,
            n_avail_magic,
            plotting,
        );
        if self.pp_paths.is_empty() {
            if n_avail_magic > 0 {
                // Only panic when T gates couldn't consume available magic; pending Cliffords
                // whose correction S gate emission failed will simply retry next lcycle.
                let has_pending_t = self.pps_pending.iter().any(|pp| pp.gate_type.is_t());
                // Patches held after earlier injections can block every route for a few
                // lcycles; that resolves itself when the holds expire.
                let holds_outstanding =
                    self.holding_now || !self.held.is_empty() || !self.helper_wanted.is_empty();
                if has_pending_t && !holds_outstanding {
                    panic!(
                        "{}",
                        format!(
                            "lcycle {}: Cannot schedule products [{}] on current layout ({} magic)",
                            self.current_lcycle,
                            self.pps_pending
                                .iter()
                                .map(|pp| pp.to_operator_str())
                                .collect::<Vec<_>>()
                                .join(", "),
                            n_avail_magic,
                        )
                        .red()
                    );
                }
            }
            false
        } else {
            true
        }
    }

    pub(crate) fn update_cultivators(&mut self) -> usize {
        let _timer = accum_start!(self.timers);
        let n_avail_magic = self.cultivation.update_cultivators(&mut self.input.topo, &self.used);
        info_sched!("  Available magic {}", n_avail_magic);
        n_avail_magic
    }

    /// First pass: schedule products with precomputed Clifford trees.
    /// Products whose tree nodes are all free are scheduled immediately.
    /// Products whose data nodes are blocked mark those nodes as used to prevent
    /// other products from routing through them this lcycle.
    fn sched_precomputed(&mut self, plotting: bool) {
        let _timer = accum_start!(self.timers);
        self.remaining_ids_buf.clear();
        self.remaining_ids_buf.extend(self.pps_pending.iter().map(|pp| pp.id));
        let mut to_remove: Vec<i32> = Vec::new();
        // Clone into a local Vec to release the borrow on `remaining_ids_buf`
        // before the loop body calls `&mut self` methods.
        let ids_to_process: Vec<i32> = self.remaining_ids_buf.clone();
        for &pp_id in &ids_to_process {
            // A multi-qubit measurement whose basis a pending S correction changes cannot use
            // its tree (precomputed for the original basis): route it now, on its conjugated
            // terminals.
            let is_tree_m = self.input.circuit.product(pp_id).gate_type.is_m()
                && self.precomputed_clifford_trees.contains_key(&pp_id);
            if is_tree_m && self.frame.is_some() && !self.frame_prepare(pp_id) {
                continue;
            }
            if is_tree_m
                && (self.frame.is_some()
                    || (self.t_correction == TCorrection::Legacy && self.conjugation_due(pp_id)))
            {
                if self.frame.is_none() {
                    self.refresh_conjugation(pp_id);
                }
                if self.conjugated_t_products.contains(&pp_id) {
                    let result = if !self.terminal_nodes(pp_id) {
                        PathResult::NoPath
                    } else if self.terminals_buf.len() == 1 {
                        self.sched_measurement(pp_id, plotting)
                    } else {
                        self.sched_t_or_multi(pp_id, plotting)
                    };
                    match result {
                        PathResult::PathFound(opt_graph) => {
                            to_remove.push(pp_id);
                            if let Some(ref g) = opt_graph {
                                let node_ids: Vec<u16> = g.iter_nodes().collect();
                                self.mark_nodes_used(&node_ids);
                            }
                            let keep = if plotting { opt_graph.map(Rc::new) } else { None };
                            self.pp_paths.push((pp_id, keep));
                            self.frame_executed(pp_id);
                        }
                        PathResult::NoPath => {
                            let pp = self.input.circuit.product(pp_id);
                            Self::mark_blocked_product_as_used(
                                &mut self.used,
                                &self.input.topo,
                                pp,
                            );
                        }
                    }
                    continue;
                }
                // The correction no longer changes its basis: the precomputed tree applies.
            }
            if self.t_correction == TCorrection::Hybrid
                && self.frame.is_some()
                && self.input.circuit.product(pp_id).gate_type.is_cx()
                && !self.hybrid_cx_ready(pp_id, plotting)
            {
                let pp = self.input.circuit.product(pp_id);
                Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
                continue;
            }
            // Clone the Rc to end the immutable borrow on `precomputed_clifford_trees`.
            let Some(tree) = self.precomputed_clifford_trees.get(&pp_id).map(Rc::clone) else {
                continue;
            };
            let all_free = tree.iter_nodes().all(|nid| !self.used[nid as usize]);
            if all_free {
                to_remove.push(pp_id);
                let node_ids: Vec<u16> = tree.iter_nodes().collect();
                let (_tree_n_nodes, _tree_n_edges) = (tree.n_nodes, tree.n_edges);
                let opt_tree: Option<Rc<TreeGraph>> =
                    if plotting { Some(Rc::clone(&tree)) } else { None };
                // Drop `tree` before the `&mut self` call to `mark_nodes_used`.
                drop(tree);
                self.mark_nodes_used(&node_ids);
                info_sched!(
                    "  Scheduled product {} (precomputed) with {} nodes and {} edges",
                    self.input.circuit.product(pp_id),
                    _tree_n_nodes,
                    _tree_n_edges
                );
                self.pp_paths.push((pp_id, opt_tree));
                if self.input.circuit.product(pp_id).gate_type.is_m() {
                    self.frame_executed(pp_id);
                }
            } else {
                let pp = self.input.circuit.product(pp_id);
                Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
            }
        }
        self.pps_pending.retain(|pp| !to_remove.contains(&pp.id));
    }

    /// Marks the data nodes of a blocked product as used so that other products
    /// cannot route through them this lcycle (takes separate params to avoid
    /// borrow conflicts in caller loops).
    fn mark_blocked_product_as_used(used: &mut Vec<bool>, topo: &TopoGraph, pp: &PauliProduct) {
        for node_id in operators_to_node_ids(topo, &pp.operators) {
            used[node_id as usize] = true;
        }
    }

    /// Attempts to emit a physical S correction gate for `qubit` when `correction_power[qubit]`
    /// is non-zero. Returns `true` if the qubit's correction was handled (gate emitted or power=2
    /// classical-only), `false` if scheduling the physical S gate failed this lcycle.
    ///
    /// On success the correction power is cleared to 0. On failure (NoPath), the circuit push is
    /// rolled back and the correction power is left unchanged so the caller can retry next lcycle.
    ///
    /// Only reached when the pending correction's axis doesn't match the blocked S/SX
    /// product's own fixed axis (S is Z, SX is X) - a same-axis correction shares the
    /// blocked product's generator, so `try_merge_clifford_correction` folds the two
    /// together directly instead of materializing a standalone gate first.
    fn try_emit_s_correction(&mut self, qubit: u16, plotting: bool) -> bool {
        let q = qubit as usize;
        let power = self.correction_power[q];
        info_sched!(
            "  Correction emit for qubit {} (power={}, basis={})",
            qubit,
            power,
            self.correction_basis[q]
        );
        if power == 0 {
            return true;
        }
        if power == 2 {
            // Z correction: tracked classically — no physical gate needed.
            info_sched!("  Qubit {}: Z correction (classical), clearing power", qubit);
            self.correction_power[q] = 0;
            return true;
        }
        // power 1 (S) or 3 (S†): emit a physical S gate on the qubit.
        let s_id = self.next_correction_id;
        let basis = self.correction_basis[q];
        let s_pp = PauliProduct {
            id: s_id,
            gate_type: GateType::S,
            operators: vec![Operator { qubit, basis }],
            parents: vec![],
            children: vec![],
            after: Vec::new(),
            cond: Vec::new(),
            max_qubit: qubit,
        };
        self.input.circuit.pps.push(s_pp);
        let new_ops = vec![Operator { qubit, basis }];
        let terminals = operators_to_node_ids(&self.input.topo, &new_ops);
        let root_info = Self::compute_root_info_for_terminals(&self.input.topo, &terminals);
        self.precomputed_terminals.push(terminals);
        self.precomputed_root_info.push(root_info);

        if !self.terminal_nodes(s_id) {
            self.input.circuit.pps.pop();
            self.precomputed_terminals.pop();
            self.precomputed_root_info.pop();
            return false;
        }
        let result = self.sched_s_sx(s_id, plotting);
        match result {
            PathResult::PathFound(opt_graph) => {
                info_sched!(
                    "  Emitted correction S gate id={} for qubit {} (power={}), clearing",
                    s_id,
                    qubit,
                    power
                );
                self.next_correction_id += 1;
                self.correction_gates_emitted += 1;
                if let Some(ref g) = opt_graph {
                    let node_ids: Vec<u16> = g.iter_nodes().collect();
                    self.mark_nodes_used(&node_ids);
                }
                self.pp_paths.push((s_id, opt_graph.map(Rc::new)));
                self.correction_power[q] = 0;
                true
            }
            PathResult::NoPath => {
                info_sched!(
                    "  Correction S gate emit for qubit {} (power={}) failed (NoPath), will retry next lcycle",
                    qubit,
                    power
                );
                self.input.circuit.pps.pop();
                self.precomputed_terminals.pop();
                self.precomputed_root_info.pop();
                false
            }
        }
    }

    /// Merges a pending correction directly into a blocked `S` or `SX` product on the same
    /// qubit, rather than materializing it as a separate gate first. The caller must have
    /// already verified the correction's axis matches this product's own fixed axis (Z for
    /// S, X for SX) - only then do the pending correction and the blocked product share a
    /// generator, making composition pure exponent addition mod 4: at most one physical gate
    /// is ever needed for the combined effect, and it is `pp_id` itself - no new product is
    /// injected. Returns `true` if `pp_id` was fully handled this lcycle (pushed onto
    /// `pp_paths`, whether or not a physical gate was scheduled), `false` if the physical
    /// gate could not be routed this lcycle (`correction_power` is left unchanged so the
    /// caller retries next lcycle).
    fn try_merge_clifford_correction(&mut self, pp_id: i32, qubit: u16, plotting: bool) -> bool {
        let q = qubit as usize;
        let gate_type = self.input.circuit.product(pp_id).gate_type;
        debug_assert_eq!(self.correction_basis[q], if gate_type.is_s() { 'Z' } else { 'X' });
        let merged_power = (self.correction_power[q] + 1) % 4;
        info_sched!(
            "  Merging {:?} product {} (qubit {}): pending power={} + 1 -> merged={}",
            gate_type,
            pp_id,
            qubit,
            self.correction_power[q],
            merged_power
        );
        if merged_power == 0 || merged_power == 2 {
            // Combined effect is classical only (identity or Z/X) - no physical gate needed.
            info_sched!(
                "  Product {} fully absorbed classically (merged power={}), clearing correction",
                pp_id,
                merged_power
            );
            self.correction_power[q] = 0;
            self.free_merged_products.insert(pp_id);
            self.pp_paths.push((pp_id, None));
            return true;
        }
        // merged_power is 1 or 3: exactly one physical gate realizes the combined effect,
        // and sched_s_sx's footprint doesn't depend on which net phase it represents, so
        // scheduling the pending product itself (unmodified) suffices.
        if !self.terminal_nodes(pp_id) {
            info_sched!("  Cannot merge-schedule {}: no data nodes found in working graph", pp_id);
            return false;
        }
        match self.sched_s_sx(pp_id, plotting) {
            PathResult::PathFound(opt_graph) => {
                info_sched!(
                    "  Merge-scheduled {:?} product {} (qubit {}, merged power={}), clearing correction",
                    gate_type,
                    pp_id,
                    qubit,
                    merged_power
                );
                if let Some(ref g) = opt_graph {
                    let node_ids: Vec<u16> = g.iter_nodes().collect();
                    self.mark_nodes_used(&node_ids);
                }
                self.pp_paths.push((pp_id, opt_graph.map(Rc::new)));
                self.correction_power[q] = 0;
                true
            }
            PathResult::NoPath => {
                info_sched!(
                    "  Merge-schedule for {:?} product {} (qubit {}) failed (NoPath), retry next lcycle",
                    gate_type,
                    pp_id,
                    qubit
                );
                false
            }
        }
    }

    /// Second pass: greedily schedule T gates, measurements, and S/SX gates.
    /// T gates are skipped when no magic state is available.
    fn sched_remaining(&mut self, n_avail_magic: &mut usize, plotting: bool) {
        let _timer = accum_start!(self.timers);
        self.t_conjugated_bases.clear();
        for i in 0..self.pps_pending.len() {
            let pp_id = self.pps_pending[i].id;
            let pp = self.input.circuit.product(pp_id);
            if Self::should_precompute(pp) {
                continue;
            }
            let (pp_id, gate_type) = (pp.id, pp.gate_type);
            // For a blocked S/SX, fold in a pending correction that shares its fixed axis
            // (S is always Z-basis, SX is always X-basis, per real .trans circuits) rather
            // than materializing a standalone gate first. A correction about a different
            // axis (common, since failed T gates can target any axis) is a different
            // Clifford generator and can't be folded in - it must still be materialized
            // first. CX is multi-qubit and handled by sched_precomputed, so it never
            // appears here.
            if gate_type.is_s() || gate_type.is_sx() {
                let op_qubit = self.input.circuit.product(pp_id).operators[0].qubit;
                let power = self.correction_power[op_qubit as usize];
                let native_axis = if gate_type.is_s() { 'Z' } else { 'X' };
                if power != 0 && self.correction_basis[op_qubit as usize] == native_axis {
                    if !self.try_merge_clifford_correction(pp_id, op_qubit, plotting) {
                        let pp = self.input.circuit.product(pp_id);
                        Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
                    }
                    continue;
                } else if power != 0 {
                    info_sched!(
                        "  Clifford {:?} {} (qubit {}) deferred: pending correction power={}",
                        gate_type,
                        pp_id,
                        op_qubit,
                        power
                    );
                    let emitted = self.try_emit_s_correction(op_qubit, plotting);
                    // Block the qubit node only if the correction S gate was placed; otherwise
                    // leave it free so T gates can still route through it (avoids deadlock).
                    if emitted {
                        let pp = self.input.circuit.product(pp_id);
                        Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
                    }
                    continue;
                }
            }
            // Apply just-in-time S^k conjugation (X<->Y swap for odd power) only when needed;
            // skipping the common case avoids Vec allocations and topology lookups every lcycle.
            // A measurement is conjugated like a T: measuring P on the state before a pending
            // correction S is measuring S^dag P S, and the correction stays pending afterwards.
            if (gate_type.is_t() || gate_type.is_m()) && self.frame.is_some() {
                if !self.frame_prepare(pp_id) {
                    continue;
                }
            } else if (gate_type.is_t() || gate_type.is_m())
                && self.t_correction == TCorrection::Legacy
                && self.conjugation_due(pp_id)
            {
                let swapped = self.refresh_conjugation(pp_id);
                if gate_type.is_t() && !swapped.is_empty() {
                    self.t_conjugated_bases.insert(pp_id, swapped);
                }
            }
            if *n_avail_magic > 0 || !gate_type.is_t() {
                info_sched!("  Trying to schedule product {}", self.input.circuit.product(pp_id));
                let result = if !self.terminal_nodes(pp_id) {
                    info_sched!(
                        "    Cannot schedule {}: no data nodes found in working graph",
                        pp_id
                    );
                    PathResult::NoPath
                } else if self.terminals_buf.len() == 1 && (gate_type.is_m() || gate_type.is_r()) {
                    // A single-patch measurement and a patch re-initialization (`R`) both
                    // occupy just the data patch's node for one lcycle.
                    self.sched_measurement(pp_id, plotting)
                } else if gate_type.is_s() || gate_type.is_sx() {
                    self.sched_s_sx(pp_id, plotting)
                } else {
                    self.sched_t_or_multi(pp_id, plotting)
                };
                if let PathResult::PathFound(opt_graph) = result {
                    info_sched!("  Scheduled product {}", self.input.circuit.product(pp_id));
                    if let Some(ref pp_graph) = opt_graph {
                        let node_ids: Vec<u16> = pp_graph.iter_nodes().collect();
                        self.mark_nodes_used(&node_ids);
                    }
                    let magic = opt_graph
                        .as_ref()
                        .and_then(|g| g.root_node_id)
                        .unwrap_or(self.astar.last_goal);
                    self.pp_paths.push((pp_id, opt_graph.map(Rc::new)));
                    if gate_type.is_t() || gate_type.is_m() {
                        self.frame_executed(pp_id);
                    }
                    if gate_type.is_t() {
                        *n_avail_magic -= 1;
                        self.stats.inc_t();
                        // With T failures switched off (-F) no correction is ever owed, so the
                        // auto-correction needs no helper and no hold.
                        let helper_corrected = match self.t_correction {
                            TCorrection::LitinskiReady
                            | TCorrection::LitinskiY
                            | TCorrection::InplaceY => true,
                            TCorrection::Hybrid => !self.frame_owns(pp_id),
                            _ => false,
                        };
                        if helper_corrected && !self.no_t_failures {
                            self.inject_t(pp_id, magic);
                        }
                    }
                    continue;
                }
            }
            info_sched!("  Could not schedule {} on graph", pp_id);
            if self.frame_exec.contains_key(&pp_id) {
                for i in 0..self.precomputed_terminals[pp_id as usize].len() {
                    let nid = self.precomputed_terminals[pp_id as usize][i];
                    self.used[nid as usize] = true;
                }
            } else {
                let pp = self.input.circuit.product(pp_id);
                Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
            }
        }
    }

    fn sched_measurement(&mut self, _pp_id: i32, plotting: bool) -> PathResult {
        let node_id = self.terminals_buf[0];
        let node = self.input.topo.node(node_id);
        if self.used[node.id as usize] {
            info_sched!(
                "    Cannot schedule {}: node for M {} is used",
                _pp_id,
                self.input.topo.label(node_id)
            );
            return PathResult::NoPath;
        }
        if !plotting {
            self.used[node_id as usize] = true;
            self.stats.inc_with_cultivation(
                node.node_type,
                self.input.topo.cultivation_times[node_id as usize],
            );
            return PathResult::PathFound(None);
        }
        let mut g = TreeGraph::new(self.input.topo.n_nodes);
        g.add_node(node, self.input.topo.label(node_id));
        PathResult::PathFound(Some(g))
    }

    /// Schedules an S or SX gate: data node plus one same-row ancilla nb.
    /// S/SX require two adjacent nodes in the same row (the data qubit and its
    /// horizontal routing nb).
    fn sched_s_sx(&mut self, pp_id: i32, plotting: bool) -> PathResult {
        if self.terminals_buf.len() == 2 {
            return self.sched_s_y(pp_id, plotting);
        }
        let node_id = self.terminals_buf[0];
        let node = self.input.topo.node(node_id);
        if self.used[node.id as usize] {
            let _gate_type = self.input.circuit.product(pp_id).gate_type;
            info_sched!(
                "    Cannot schedule {}: node for {:?} {} is used",
                pp_id,
                _gate_type,
                self.input.topo.label(node_id)
            );
            return PathResult::NoPath;
        }
        for &nb_id in node.nbs_slice() {
            let nb = self.input.topo.node(nb_id);
            if nb.pos.1 == node.pos.1 {
                info_sched!(
                    "    product {} on node {} has available ancilla {}",
                    self.input.circuit.product(pp_id),
                    self.input.topo.label(node_id),
                    self.input.topo.label(nb_id)
                );
                if !self.used[nb_id as usize] {
                    // Always return the two-node tree, plotting or not: an S/SX holds its
                    // data node and this ancilla for all 3 of its lcycles, and
                    // advance_clifford_state takes the nodes to hold from this tree. (A
                    // single-qubit S/SX has no precomputed tree, so returning None here left
                    // its nodes free in lcycles 2-3 whenever paths were not being plotted.)
                    let _ = plotting;
                    let mut g = TreeGraph::new(self.input.topo.n_nodes);
                    g.add_node(node, self.input.topo.label(node_id));
                    g.add_node(nb, self.input.topo.label(nb_id));
                    g.add_edge(node_id, nb_id);
                    return PathResult::PathFound(Some(g));
                }
            }
        }
        info_sched!("    Cannot schedule S/SX {}: no available ancilla", pp_id);
        PathResult::NoPath
    }

    /// Schedules a pi/4 rotation about Y on one qubit (a scheduler-added correction or flush):
    /// like a Y measurement it needs both of the qubit's data nodes, joined by a routing tree
    /// that serves as its ancilla, held for the same lcycles as an S.
    fn sched_s_y(&mut self, pp_id: i32, plotting: bool) -> PathResult {
        if self.terminals_buf.iter().any(|&t| self.used[t as usize]) {
            info_sched!("    Cannot schedule Y rotation {}: a data node is used", pp_id);
            return PathResult::NoPath;
        }
        self.sched_t_or_multi(pp_id, plotting)
    }

    fn sched_t_or_multi(&mut self, pp_id: i32, plotting: bool) -> PathResult {
        debug_assert!(!self.terminals_buf.iter().any(|node_id| self.used[*node_id as usize]));
        let root_ids = self.root_nodes(pp_id as usize, &self.terminals_buf[..]);
        if root_ids.is_empty() {
            info_sched!("    Cannot schedule {}: no roots available", pp_id);
            return PathResult::NoPath;
        }
        let pp = self.input.circuit.product(pp_id);
        let gate_type = pp.gate_type;
        let n_operators = self.exec_ops(pp_id).len();
        let result = if gate_type.is_t() && n_operators == 1 {
            self.astar.compute(
                &self.terminals_buf[..],
                &root_ids[..],
                &self.input.topo,
                &mut self.used,
                &self.cultivation.ready_magic_positions,
                plotting,
            )
        } else {
            debug_assert!(
                !Self::should_precompute(self.input.circuit.product(pp_id))
                    || self.conjugated_t_products.contains(&pp_id),
                "should_precompute product {:?} reached Steiner path",
                pp_id
            );
            match self.stree_computation.compute(
                &self.input.topo,
                &self.used,
                &root_ids,
                &self.terminals_buf,
                gate_type,
            ) {
                Some(tree) => PathResult::PathFound(Some(tree)),
                None => PathResult::NoPath,
            }
        };
        if let PathResult::PathFound(opt_g) = result {
            return PathResult::PathFound(opt_g);
        }
        info_sched!("    Cannot schedule {}: no path found", pp_id);
        PathResult::NoPath
    }

    /// Post-lcycle bookkeeping: remove scheduled products, unlock children,
    /// advance Clifford state, and process T-gate coin-flip outcomes.
    fn complete_lcycle(&mut self) -> io::Result<()> {
        let _timer = accum_start!(self.timers);
        #[cfg(debug_assertions)]
        if self.hosting() {
            self.check_hosted_lcycle()?;
        }
        self.scheduled_ids_buf.clear();
        self.scheduled_ids_buf.extend(self.pp_paths.iter().map(|(id, _)| *id));
        self.pps_pending.retain(|pp| !self.scheduled_ids_buf.contains(&pp.id));
        debug_sched!("After purge, pps_to_sched len {}", self.pps_pending.len());
        let t_newly_scheduled = self
            .pp_paths
            .iter()
            .filter(|(id, _)| self.input.circuit.product(*id).gate_type.is_t())
            .count();
        self.cultivation.t_products_remaining =
            self.cultivation.t_products_remaining.saturating_sub(t_newly_scheduled);
        self.process_t_gate_outcomes();
        self.clear_corrections_on_reinit();
        self.unlock_children();
        let ready = std::mem::take(&mut self.children_buf);
        self.children_buf = self.admit(ready);
        self.advance_clifford_state();
        debug_sched!(
            "After inserting previous lcycle cliffords, pps_to_sched len {}",
            self.pps_pending.len()
        );
        self.pps_pending
            .extend(self.children_buf.iter().map(|&id| self.input.circuit.product(id).clone()));
        if self.hosting() {
            // advance_clifford_state has released finished Cliffords' paths, so a hosted
            // patch whose last product just completed is free to go dormant.
            self.retire_hosted();
            let children = self.children_buf.clone();
            self.activate_hosted_for(&children);
        }
        debug_sched!(
            "After adding {} children, pps_to_sched len {}",
            self.children_buf.len(),
            self.pps_pending.len()
        );
        let lcycle_ids: Vec<i32> = self.pp_paths.iter().map(|(id, _)| *id).collect();
        self.lcycle_scheduled.push((self.current_lcycle, lcycle_ids));
        #[cfg(debug_assertions)]
        self.check_lcycle()?;
        self.scheduled_products.extend(self.pp_paths.iter().map(|(id, _)| *id));
        Ok(())
    }

    /// Coin-flip T gate outcomes and update per-qubit correction tracking.
    ///
    /// T gates complete in one lcycle regardless of outcome. A failed T gate
    /// increments `correction_power` for each of its qubits (tracked classically);
    /// no recovery lcycle is needed.
    /// A T corrected on an ancilla failed: nothing is owed on the data. A magic patch measured in
    /// place needs the slower Y basis; one still waiting for a helper remembers the failure.
    fn auto_corrected_failure(&mut self, pp_id: i32) {
        if let Some((magic, decision)) = self.inplace_pending.remove(&pp_id) {
            let y = self.y_meas_lcycles;
            self.t_stats.inplace_y += 1;
            self.delay_decision(pp_id, y);
            // Already held through `decision`; extend by the Y measurement.
            let now = self.current_lcycle;
            let start = decision.max(now);
            self.held.push((decision + y, magic));
            self.t_stats.hold_patch_lcycles += decision + y - start;
        } else if self.helper_wanted.iter().any(|w| w.0 == pp_id) {
            self.failed_waiting.insert(pp_id);
        }
    }

    fn process_t_gate_outcomes(&mut self) {
        let pp_ids: Vec<i32> = self.pp_paths.iter().map(|(id, _)| *id).collect();
        if self.frame.is_some() {
            let mut failed = Vec::new();
            for &pp_id in &pp_ids {
                if !self.input.circuit.product(pp_id).gate_type.is_t() {
                    continue;
                }
                if self.rng_uniform.gen_bool(0.5) {
                    info_sched!("  T gate {} succeeded", pp_id);
                } else if self.frame_owns(pp_id) {
                    self.t_gate_failures += 1;
                    info_sched!(
                        "  T gate {} failed (50% probability), folded into the frame",
                        pp_id
                    );
                    failed.push(pp_id);
                } else {
                    self.t_gate_failures += 1;
                    info_sched!("  T gate {} failed (50% probability), auto-corrected", pp_id);
                    self.auto_corrected_failure(pp_id);
                }
            }
            self.frame_end_lcycle(&failed);
            #[cfg(debug_assertions)]
            if self.t_correction == TCorrection::Hybrid {
                assert!(self.frame.as_ref().unwrap().is_local(), "hybrid frame became entangling");
            }
            return;
        }
        for pp_id in pp_ids {
            let pp = self.input.circuit.product(pp_id);
            if !pp.gate_type.is_t() {
                continue;
            }
            if self.no_t_failures || self.rng_uniform.gen_bool(0.5) {
                info_sched!("  T gate {} succeeded", pp_id);
                self.inplace_pending.remove(&pp_id);
            } else {
                self.t_gate_failures += 1;
                if self.t_correction != TCorrection::Legacy {
                    // Auto-corrected: the correction is a choice of ancilla measurement basis, so
                    // nothing is owed on the data. A magic patch measured in place needs the
                    // slower Y basis; one still waiting for a helper remembers the failure.
                    info_sched!("  T gate {} failed (50% probability), auto-corrected", pp_id);
                    self.auto_corrected_failure(pp_id);
                    continue;
                }
                info_sched!("  T gate {} failed (50% probability), S correction tracked", pp_id);
                // Each failed qubit's S^k power increments, using the conjugated basis
                // recorded in sched_remaining.
                let pp = self.input.circuit.product(pp_id);
                let conjugated_map: HashMap<u16, char> = self
                    .t_conjugated_bases
                    .get(&pp_id)
                    .map(|v| v.iter().copied().collect())
                    .unwrap_or_default();
                let orig_ops: Vec<(u16, char)> =
                    pp.operators.iter().map(|op| (op.qubit, op.basis)).collect();
                for (qubit, orig_basis) in orig_ops {
                    let q = qubit as usize;
                    self.correction_power[q] = (self.correction_power[q] + 1) % 4;
                    self.correction_basis[q] =
                        conjugated_map.get(&qubit).copied().unwrap_or(orig_basis);
                    info_sched!(
                        "    Qubit {} correction: power now {} (basis={})",
                        qubit,
                        self.correction_power[q],
                        self.correction_basis[q]
                    );
                }
            }
        }
    }

    /// Decrements `remaining_parents` for each completed product and collects
    /// newly-ready children into `children_buf`.
    fn unlock_children(&mut self) {
        self.children_buf.clear();
        for i in 0..self.pp_paths.len() {
            let pp_id = self.pp_paths[i].0;
            let pp = self.input.circuit.product(pp_id);
            let gate_type = pp.gate_type;
            if gate_type.is_clifford() && !self.free_merged_products.contains(&pp_id) {
                match self.clifford_paths.get(&pp_id) {
                    // S/SX: count==2 means this is the second of three lcycles;
                    // children are not unlocked until the third (final) lcycle.
                    Some((count, _, _, _)) if *count == 2 => {
                        debug_assert!(gate_type.is_s() || gate_type.is_sx());
                        continue;
                    }
                    // count==1 (CX) or count==1 (S/SX final): fall through to unlock.
                    // None means this is the first lcycle: children not yet unlocked.
                    None => continue,
                    _ => {}
                }
            }
            // Free-merged S products (see try_merge_s_correction) are single-lcycle,
            // zero-node events - fall straight through to unlocking their children.
            if !self.hosted.live.is_empty() {
                self.hosted.completed_buf.push(pp_id); // pp_id completes this lcycle
            }
            let children: Vec<i32> = self.input.circuit.product(pp_id).children.clone();
            for child_id in children {
                // A classical edge (the child's `@` list names this product) is held back by the
                // reaction latency; ordinary qubit-overlap edges are not.
                if self.reaction_latency > 0
                    && self.input.circuit.product(child_id).after.contains(&pp_id)
                {
                    self.delayed_unlocks
                        .push((self.current_lcycle + self.reaction_latency, child_id));
                    continue;
                }
                self.remaining_parents[child_id as usize] -= 1;
                if self.remaining_parents[child_id as usize] == 0
                    && !self.children_buf.contains(&child_id)
                {
                    self.children_buf.push(child_id);
                }
            }
        }
    }

    /// Advances multi-lcycle Clifford state: decrements remaining-lcycle counters
    /// and inserts new entries for first-seen Clifford products.
    ///
    /// CX requires 2 consecutive lcycles (initial count = 1, removed when it hits 0).
    /// S/SX require 3 consecutive lcycles (initial count = 2).
    fn advance_clifford_state(&mut self) {
        let pp_paths_snapshot: Vec<(i32, Option<Rc<TreeGraph>>)> =
            self.pp_paths.iter().map(|(id, opt)| (*id, opt.as_ref().map(Rc::clone))).collect();
        for (pp_id, opt_pp_path) in &pp_paths_snapshot {
            let pp_id = *pp_id;
            let pp = self.input.circuit.product(pp_id);
            if !pp.gate_type.is_clifford() || self.free_merged_products.contains(&pp_id) {
                // Free-merged S products (see try_merge_s_correction) are already complete
                // after their single lcycle - never enter the multi-lcycle carry-forward state.
                continue;
            }
            let gate_type = pp.gate_type;
            if let Some(clifford_path) = self.clifford_paths.get_mut(&pp_id) {
                clifford_path.0 -= 1;
                if clifford_path.0 == 0 {
                    self.clifford_paths.swap_remove(&pp_id);
                }
            } else {
                // First lcycle for this Clifford: insert with remaining count.
                let count = if gate_type.is_cx() { 1 } else { 2 };
                let node_ids: Vec<u16> = if let Some(tree) = opt_pp_path {
                    tree.iter_nodes().collect()
                } else {
                    self.precomputed_clifford_trees
                        .get(&pp_id)
                        .map(|t| t.iter_nodes().collect())
                        .unwrap_or_default()
                };
                let pp_owned = self.input.circuit.product(pp_id).clone();
                self.clifford_paths.insert(
                    pp_id,
                    (count, pp_owned, node_ids, opt_pp_path.as_ref().map(Rc::clone)),
                );
            }
        }
    }

    fn print_sched_stats(&mut self, n_lcycles: usize) {
        self.stats.summarize(n_lcycles);
        let tot_t = self.count_t_products() - self.n_skipped_t;
        let fail_pct =
            if tot_t > 0 { 100.0 * self.t_gate_failures as f64 / tot_t as f64 } else { 0.0 };
        println!("Magic state cultivation time:");
        let mean = self.cultivation.cultivation_times_log.iter().sum::<i32>() as f64
            / self.cultivation.cultivation_times_log.len() as f64;
        let min = self.cultivation.cultivation_times_log.iter().min().copied().unwrap_or(0);
        let max = self.cultivation.cultivation_times_log.iter().max().copied().unwrap_or(0);
        println!("  number:  {}", self.cultivation.cultivation_times_log.len());
        println!("  average: {:.2}", mean);
        println!("  min:     {}", min);
        println!("  max:     {}", max);
        println!("T gate failures: {}/{} ({:.1}%)", self.t_gate_failures, tot_t, fail_pct);
        match self.t_correction {
            TCorrection::Legacy => println!("T correction: legacy (per-qubit S record)"),
            TCorrection::Hybrid => {
                let f = &self.frame_stats;
                println!(
                    "T correction: Hybrid; {} one-qubit failures and {} physical Cliffords folded \
                     into the frame, {} flush rotations before CX, {} T/M rewritten of {}, {} \
                     product-lcycles waiting for decoding; multi-qubit T: {} injections, {} helper \
                     patches ({} found none), {} patch-lcycles held",
                    f.failures,
                    f.cliffords,
                    f.flushes,
                    f.rewritten,
                    f.products,
                    f.waits,
                    self.t_stats.injections,
                    self.t_stats.helpers,
                    self.t_stats.helpers_missing,
                    self.t_stats.hold_patch_lcycles
                );
            }
            TCorrection::Frame => {
                let f = &self.frame_stats;
                let avg =
                    |w: usize| if f.products > 0 { w as f64 / f.products as f64 } else { 0.0 };
                println!(
                    "T correction: Frame; {} failures and {} physical Cliffords folded into the \
                     frame, {} T/M rewritten of {}, mean weight {:.2} as written -> {:.2} executed, \
                     max {}, {} product-lcycles waiting for decoding",
                    f.failures,
                    f.cliffords,
                    f.rewritten,
                    f.products,
                    avg(f.ideal_weight),
                    avg(f.exec_weight),
                    f.max_exec_weight,
                    f.waits
                );
            }
            mode => println!(
                "T correction: {:?}; {} injections, {} helper patches ({} ready states wasted, {} \
                 waited for one, {} found none), {} patch-lcycles held, max decision delay {} \
                 lcycles, {} measured in place ({} in Y)",
                mode,
                self.t_stats.injections,
                self.t_stats.helpers,
                self.t_stats.helpers_ready,
                self.t_stats.helpers_deferred,
                self.t_stats.helpers_missing,
                self.t_stats.hold_patch_lcycles,
                self.t_stats.max_decision_delay,
                self.t_stats.inplace,
                self.t_stats.inplace_y
            ),
        }
        let n_cond = self.n_conditional();
        if n_cond > 0 {
            match self.cond_outcomes {
                CondOutcomes::Worst => {
                    println!("Conditional products: {} (all run: --cond-outcomes worst)", n_cond)
                }
                CondOutcomes::Random => println!(
                    "Conditional products: {}, skipped {} ({} T) on {} sampled outcome(s), p1={}",
                    n_cond,
                    self.skipped.len(),
                    self.n_skipped_t,
                    self.outcomes.len(),
                    self.cond_p1
                ),
            }
        }
        println!("Steiner tree computation called {} times", self.stree_computation.n_calls);
        println!("A* computation called {} times", self.astar.n_calls);

        if self.record_cultivation_dist {
            let dist_fname = format!("{}.cultivation_dist", self.input.circuit_stem());
            match self.write_cultivation_dist(&dist_fname, min, max) {
                Ok(()) => println!("Cultivation time distribution written to {}", dist_fname),
                Err(e) => eprintln!("Warning: could not write cultivation dist: {}", e),
            }
        }
    }

    /// Writes the normalized distribution of cultivation times to `fname`.
    /// Each line contains: `<cultivation_time> <normalized_count>` where
    /// `normalized_count = count(t) / total_events` for t in [min, max].
    fn write_cultivation_dist(&self, fname: &str, min: i32, max: i32) -> io::Result<()> {
        let times = &self.cultivation.cultivation_times_log;
        let total = times.len();
        if total == 0 || min > max {
            return Ok(());
        }
        let f = File::create(fname)?;
        let mut buf_f = BufWriter::new(f);
        for t in min..=max {
            let count = times.iter().filter(|&&v| v == t).count();
            let normalized = count as f64 / total as f64;
            writeln!(buf_f, "{} {:.6}", t, normalized)?;
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_lcycle(&self) -> io::Result<()> {
        let mut lcycle_used = vec![false; self.input.topo.n_nodes];
        for &(pp_id, ref opt_tree) in &self.pp_paths {
            let Some(tree) = opt_tree else { continue };
            let tree = tree.as_ref();
            let pp = self.input.circuit.product(pp_id);
            if self.scheduled_products.contains(&pp_id) && !pp.gate_type.is_clifford() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("product {} scheduled twice", pp_id),
                ));
            }
            for &parent_id in &pp.parents {
                // A skipped conditional parent completed without being scheduled.
                if !self.scheduled_products.contains(&parent_id)
                    && !self.skipped_set.contains(&parent_id)
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("product {} scheduled before parent {}", pp_id, parent_id),
                    ));
                }
            }
            for &nid in &self.precomputed_terminals[pp_id as usize] {
                if !tree.contains_node(nid) {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "product {} terminal node {} missing from tree",
                            pp_id,
                            self.input.topo.label(nid)
                        ),
                    ));
                }
            }
            if pp.gate_type.is_t() {
                match tree.root_node_id {
                    None => {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            format!("product {}: T gate has no magic root node", pp_id),
                        ));
                    }
                    Some(magic_id) => {
                        if self.input.topo.node(magic_id).node_type != NodeType::Magic {
                            return Err(io::Error::new(
                                io::ErrorKind::Other,
                                format!(
                                    "product {}: root node {} is not a Magic node",
                                    pp_id, magic_id
                                ),
                            ));
                        }
                    }
                }
            }
            for node_id in tree.iter_nodes() {
                if lcycle_used[node_id as usize] {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "product {} shares node '{}' with another \
                                                       product in the same lcycle",
                            pp_id,
                            self.input.topo.label(node_id)
                        ),
                    ));
                }
                lcycle_used[node_id as usize] = true;
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_clifford_repetitions(&self) -> io::Result<()> {
        let mut cx_counts: IndexMap<i32, Vec<usize>> = IndexMap::new();
        let mut s_counts: IndexMap<i32, Vec<usize>> = IndexMap::new();
        for (lcycle_i, lcycle_ids) in &self.lcycle_scheduled {
            for &pp_id in lcycle_ids {
                let pp = self.input.circuit.product(pp_id);
                if pp.gate_type.is_cx() {
                    let lcycles = cx_counts.entry(pp_id).or_insert(Vec::new());
                    lcycles.push(*lcycle_i);
                } else if pp.gate_type.is_s() || pp.gate_type.is_sx() {
                    let lcycles = s_counts.entry(pp_id).or_insert(Vec::new());
                    lcycles.push(*lcycle_i);
                }
            }
        }
        let mut errors = Vec::new();
        for (pp_id, lcycles) in &cx_counts {
            let pp = self.input.circuit.product(*pp_id);
            if pp.gate_type.is_cx() {
                if lcycles.len() != 2 || lcycles[0] != lcycles[1] - 1 {
                    errors.push(format!("  product {} not scheduled 2x {:?}", pp, lcycles));
                }
            }
        }
        for (pp_id, lcycles) in &s_counts {
            let pp = self.input.circuit.product(*pp_id);
            if self.free_merged_products.contains(pp_id) {
                if lcycles.len() != 1 {
                    errors.push(format!(
                        "  merge-completed product {} expected exactly 1 lcycle, got {:?}",
                        pp, lcycles
                    ));
                }
            } else if pp.gate_type.is_s() || pp.gate_type.is_sx() {
                if lcycles.len() != 3
                    || lcycles[0] != lcycles[1] - 1
                    || lcycles[1] != lcycles[2] - 1
                {
                    errors.push(format!("  product {} not scheduled 3x {:?}", pp, lcycles));
                }
            }
        }
        if !errors.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Clifford repetition errors:\n{}", errors.join("\n")),
            ));
        }
        println!(
            "Clifford repetition check passed ({} CX, {} S/SX products)",
            cx_counts.len(),
            s_counts.len()
        );
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_schedule(&self) -> io::Result<()> {
        let n_products = self.input.circuit.n_products();
        let mut errors: Vec<String> = Vec::new();
        for pp_id in 0..n_products as i32 {
            let scheduled = self.scheduled_products.contains(&pp_id);
            let skipped = self.skipped_set.contains(&pp_id);
            if !scheduled && !skipped {
                errors.push(format!("  product {} was never scheduled", pp_id));
            } else if scheduled && skipped {
                errors.push(format!("  product {} was both scheduled and skipped", pp_id));
            } else if skipped && self.input.circuit.product(pp_id).cond.is_empty() {
                errors.push(format!("  unconditional product {} was skipped", pp_id));
            }
        }
        if !errors.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Completeness errors:\n{}", errors.join("\n")),
            ));
        }
        println!(
            "Schedule check passed: all {} products scheduled or skipped ({} skipped)",
            n_products,
            self.skipped.len()
        );
        Ok(())
    }

    pub(crate) fn print_schedule(&self, hdr: &str) -> io::Result<()> {
        let _timer = fn_timer!();
        debug_sched!("Printing schedule");
        let output_fname = format!("{}.schedule", self.input.circuit_stem());

        let f = File::create(&output_fname)?;
        let mut buf_f = BufWriter::new(f);

        let max_lcycle: usize =
            self.lcycle_scheduled.last().map(|(lcycle_i, _)| *lcycle_i).unwrap_or(0);
        let max_width = max_lcycle.to_string().len();
        let tot_products = self.lcycle_scheduled.iter().map(|(_, v)| v.len()).sum::<usize>();
        writeln!(buf_f, "{}", hdr)?;
        writeln!(buf_f, "# Total active logical cycles: {}", self.lcycle_scheduled.len())?;
        writeln!(buf_f, "# Total logical cycles: {}", max_lcycle)?;
        writeln!(buf_f, "# Total products: {}", tot_products)?;
        writeln!(buf_f, "# Parallelism: {:.2}", tot_products as f64 / max_lcycle as f64)?;
        if self.n_conditional() > 0 {
            // Read by plot_dynamic.py to check that exactly the failing conditions were skipped.
            let mode = match self.cond_outcomes {
                CondOutcomes::Random => "random",
                CondOutcomes::Worst => "worst",
            };
            writeln!(buf_f, "# cond_outcomes: {} p1={}", mode, self.cond_p1)?;
            let mut outs: Vec<(i32, bool)> = self.outcomes.iter().map(|(&m, &o)| (m, o)).collect();
            outs.sort_unstable();
            let outs: Vec<String> =
                outs.iter().map(|(m, o)| format!("{}={}", m, *o as u8)).collect();
            writeln!(buf_f, "# outcomes: {}", outs.join(" "))?;
            let mut skipped = self.skipped.clone();
            skipped.sort_unstable();
            let skipped: Vec<String> = skipped.iter().map(|id| id.to_string()).collect();
            writeln!(buf_f, "# skipped: {}", skipped.join(" "))?;
        }

        let colors = [
            Color::Green,
            Color::Red,
            Color::Yellow,
            Color::Blue,
            Color::Magenta,
            Color::Cyan,
            Color::White,
            Color::BrightGreen,
            Color::BrightRed,
            Color::BrightYellow,
            Color::BrightBlue,
            Color::BrightMagenta,
            Color::BrightCyan,
            Color::BrightWhite,
        ];

        // prev_cx tracks CX products seen in the previous lcycle so we can
        // suppress the second operator on the correct lcycle in the output.
        let mut prev_cx: IndexSet<i32> = IndexSet::new();
        for (lcycle_i, lcycle_ids) in &self.lcycle_scheduled {
            let mut sorted_ids = lcycle_ids.clone();
            sorted_ids.sort_by_key(|&id| {
                self.input
                    .circuit
                    .product(id)
                    .operators
                    .iter()
                    .map(|op| op.qubit)
                    .min()
                    .unwrap_or(u16::MAX)
            });
            let mut combined_chars = vec!['_'; self.input.circuit.n_qubits];
            let mut combined_colors: Vec<Option<Color>> = vec![None; self.input.circuit.n_qubits];
            for (idx, &pp_id) in sorted_ids.iter().enumerate() {
                let pp = self.input.circuit.product(pp_id);
                let color = colors[idx % colors.len()];
                for op in &pp.operators {
                    if op.qubit < self.input.circuit.n_qubits as u16 {
                        combined_chars[op.qubit as usize] = op.basis;
                        combined_colors[op.qubit as usize] = Some(color);
                    }
                }
                if pp.gate_type.is_cx() {
                    // CX spans 2 lcycles. On the first lcycle show only operator[0]
                    // (the control); on the second show only operator[1] (the target).
                    if !prev_cx.swap_remove(&pp_id) {
                        debug_sched!("  first lcycle of CX {} {}", pp_id, pp);
                        prev_cx.insert(pp_id);
                        let qubit = pp.operators[1].qubit;
                        combined_colors[qubit as usize] = None;
                        combined_chars[qubit as usize] = '_';
                    } else {
                        debug_sched!("  second lcycle of CX {} {}", pp_id, pp);
                        let qubit = pp.operators[0].qubit;
                        combined_colors[qubit as usize] = None;
                        combined_chars[qubit as usize] = '_';
                    }
                }
            }
            write!(buf_f, "{:width$}: ", lcycle_i, width = max_width)?;
            for i in 0..self.input.circuit.n_qubits {
                let ch = combined_chars[i].to_string();
                let colored_ch = match combined_colors[i] {
                    Some(c) => ch.color(c).to_string(),
                    None => ch,
                };
                write!(buf_f, "{}", colored_ch)?;
            }
            let mut id_string = String::new();
            for (idx, &pp_id) in sorted_ids.iter().enumerate() {
                let pp = self.input.circuit.product(pp_id);
                let color = colors[idx % colors.len()];
                id_string.push_str(&format!(
                    " {}",
                    format!("{}<{:?}>", pp_id, pp.gate_type).color(color)
                ));
            }
            writeln!(buf_f, "{}", id_string)?;
        }
        println!("Scheduled products written to {}", output_fname);
        Ok(())
    }
}

fn circuit_stem(fname: &str) -> &str {
    Path::new(fname).file_stem().and_then(|s| s.to_str()).unwrap_or("circuit")
}

/// Expands operators into data node IDs; Y-basis operators expand to two nodes
/// (X patch + Z patch) since a Y measurement needs both stabiliser data qubits.
fn operators_to_node_ids(topo: &TopoGraph, operators: &[Operator]) -> Vec<u16> {
    let mut node_ids = Vec::with_capacity(operators.len());
    for op in operators {
        if op.basis == 'Y' {
            node_ids.push(topo.get_data_node_id(op.qubit, 'X'));
            node_ids.push(topo.get_data_node_id(op.qubit, 'Z'));
        } else {
            node_ids.push(topo.get_data_node_id(op.qubit, op.basis.to_ascii_uppercase()));
        }
    }
    node_ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::node::Node;
    use crate::topograph::TopoGraph;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn run_scheduler(lines: &[&str], rseed: u32) -> Scheduler {
        Node::set_magic_routing(true);
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        let fname = f.path().to_string_lossy().to_string();
        let mut circuit = Circuit::new(&fname);
        circuit.load_circuit().expect("circuit load failed");
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        let mut sched =
            Scheduler::new(circuit, topo, 0.0387396, "none", String::new(), rseed, false, false);
        sched.sched_circuit().expect("sched_circuit failed");
        sched
    }

    #[test]
    fn t_gate_failures_bounded_by_total_t_gates() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 0);
        let tot_t = 4usize;
        assert!(
            sched.t_gate_failures <= tot_t,
            "t_gate_failures {} exceeds total T gates {}",
            sched.t_gate_failures,
            tot_t
        );
    }

    #[test]
    fn t_gate_failures_deterministic_with_fixed_seed() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched1 = run_scheduler(lines, 42);
        let sched2 = run_scheduler(lines, 42);
        assert_eq!(
            sched1.t_gate_failures, sched2.t_gate_failures,
            "t_gate_failures differs between runs with the same seed"
        );
    }

    #[test]
    fn t_gate_failures_varies_with_seed() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let counts: Vec<usize> =
            (0u32..20).map(|s| run_scheduler(lines, s).t_gate_failures).collect();
        let distinct = counts.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 1, "t_gate_failures never varied across 20 seeds: {:?}", counts);
    }

    #[test]
    fn all_products_appear_exactly_once_in_lcycle_scheduled() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let mut id_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for (_, ids) in &sched.lcycle_scheduled {
            for &id in ids {
                *id_counts.entry(id).or_insert(0) += 1;
            }
        }
        let n_products = 4;
        for pp_id in 0..n_products as i32 {
            let count = id_counts.get(&pp_id).copied().unwrap_or(0);
            assert_eq!(
                count, 1,
                "product {} appears {} times in lcycle_scheduled (expected 1)",
                pp_id, count
            );
        }
    }

    #[test]
    fn lcycle_scheduled_tot_entries_equals_n_products() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        let n_products = 4usize;
        assert_eq!(
            tot_entries, n_products,
            "total lcycle_scheduled entries {} != n_products {}",
            tot_entries, n_products
        );
    }

    #[test]
    fn lcycle_count_not_inflated_by_t_gate_failures() {
        // T gate failures are tracked classically (correction_power) and do not add
        // extra scheduling lcycles.
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let n_t = 4usize;
        let active_lcycles = sched.lcycle_scheduled.len();
        assert!(
            active_lcycles <= n_t,
            "active lcycles {} > n_t {} (failures should not add extra lcycles)",
            active_lcycles,
            n_t,
        );
    }

    #[test]
    fn conjugate_pauli_label_fixes_axis_and_swaps_the_other_two() {
        for axis in ['X', 'Y', 'Z'] {
            assert_eq!(Scheduler::conjugate_pauli_label(axis, axis), axis);
        }
        // S_Z fixes Z, swaps X<->Y.
        assert_eq!(Scheduler::conjugate_pauli_label('X', 'Z'), 'Y');
        assert_eq!(Scheduler::conjugate_pauli_label('Y', 'Z'), 'X');
        // S_X fixes X, swaps Y<->Z.
        assert_eq!(Scheduler::conjugate_pauli_label('Y', 'X'), 'Z');
        assert_eq!(Scheduler::conjugate_pauli_label('Z', 'X'), 'Y');
        // S_Y fixes Y, swaps X<->Z.
        assert_eq!(Scheduler::conjugate_pauli_label('X', 'Y'), 'Z');
        assert_eq!(Scheduler::conjugate_pauli_label('Z', 'Y'), 'X');
    }

    fn build_scheduler(lines: &[&str], rseed: u32) -> Scheduler {
        Node::set_magic_routing(true);
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        let fname = f.path().to_string_lossy().to_string();
        let mut circuit = Circuit::new(&fname);
        circuit.load_circuit().expect("circuit load failed");
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        Scheduler::new(circuit, topo, 0.0387396, "none", String::new(), rseed, false, false)
    }

    #[test]
    fn t_conjugation_uses_the_pending_correction_own_axis_not_always_z() {
        // A single T gate on qubit 0 measuring Z. A pending correction with axis X (as if
        // an earlier failed T gate on this qubit had targeted X, not Z) is active.
        let lines = &["+Z___<T>"];
        let mut sched = build_scheduler(lines, 0);
        sched.precompute_terminals_and_roots();
        sched.correction_power[0] = 1;
        sched.correction_basis[0] = 'X';
        let swapped = sched.apply_t_conjugation(0);
        // Conjugating a Z operand by S_X (fixes X, swaps Y<->Z) must give Y, not the
        // unconjugated Z that the old hardcoded-S_Z-only rule would have produced.
        assert_eq!(swapped, vec![(0u16, 'Y')]);
    }

    #[test]
    fn s_correction_merges_free_when_axis_matches_and_net_power_is_even() {
        // Native S products are always Z-basis, so a pending Z-axis correction shares S's
        // own generator. Power 1 (owes one S) plus the blocked S's own power gives net 2
        // (Z, classical) - fully absorbed with zero physical gates.
        let lines = &["+Z___<S>"];
        let mut sched = build_scheduler(lines, 0);
        sched.correction_power[0] = 1;
        sched.correction_basis[0] = 'Z';
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0);
        assert_eq!(sched.correction_power[0], 0);
        assert!(sched.free_merged_products.contains(&0));
        assert!(sched.scheduled_products.contains(&0));
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 1, "fully-absorbed correction costs zero extra lcycles");
    }

    #[test]
    fn s_correction_merges_to_one_physical_gate_when_axis_matches_and_net_power_is_odd() {
        // Power 2 (owes S^2 = Z) plus the blocked S's own power gives net 3 (S dagger) -
        // exactly one physical S gate realizes it, reusing the original product id.
        let lines = &["+Z___<S>"];
        let mut sched = build_scheduler(lines, 0);
        sched.correction_power[0] = 2;
        sched.correction_basis[0] = 'Z';
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0, "no new product was injected");
        assert_eq!(sched.correction_power[0], 0);
        assert!(!sched.free_merged_products.contains(&0));
        assert!(sched.scheduled_products.contains(&0));
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 3, "one physical S gate occupies exactly 3 lcycles");
    }

    #[test]
    fn s_correction_falls_back_to_materializing_when_axis_does_not_match() {
        // A pending correction about X (e.g. left over from a failed T gate that targeted
        // X) is a different generator from S's fixed Z axis and can't be folded in -
        // it must still be materialized as its own standalone gate first.
        let lines = &["+Z___<S>"];
        let mut sched = build_scheduler(lines, 0);
        sched.correction_power[0] = 1;
        sched.correction_basis[0] = 'X';
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 1, "standalone correction gate injected");
        assert_eq!(sched.correction_power[0], 0);
        assert!(sched.free_merged_products.is_empty());
        assert!(sched.scheduled_products.contains(&0), "original S product scheduled");
        assert!(sched.scheduled_products.contains(&1), "injected correction (id=1) scheduled");
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 6, "3 lcycles for the injected correction + 3 for the S gate");
    }

    #[test]
    fn sx_correction_merges_free_when_axis_matches_and_net_power_is_even() {
        // Native SX products are always X-basis, so a pending X-axis correction shares
        // SX's own generator, exactly like S/Z. Power 1 + SX's own power gives net 2
        // (classical) - fully absorbed with zero physical gates.
        let lines = &["+X___<SX>"];
        let mut sched = build_scheduler(lines, 0);
        sched.correction_power[0] = 1;
        sched.correction_basis[0] = 'X';
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0);
        assert_eq!(sched.correction_power[0], 0);
        assert!(sched.free_merged_products.contains(&0));
        assert!(sched.scheduled_products.contains(&0));
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 1, "fully-absorbed correction costs zero extra lcycles");
    }

    #[test]
    fn sx_correction_merges_to_one_physical_gate_when_axis_matches_and_net_power_is_odd() {
        // Power 2 + SX's own power gives net 3 - exactly one physical SX gate realizes
        // it, reusing the original product id.
        let lines = &["+X___<SX>"];
        let mut sched = build_scheduler(lines, 0);
        sched.correction_power[0] = 2;
        sched.correction_basis[0] = 'X';
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0, "no new product was injected");
        assert_eq!(sched.correction_power[0], 0);
        assert!(!sched.free_merged_products.contains(&0));
        assert!(sched.scheduled_products.contains(&0));
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 3, "one physical SX gate occupies exactly 3 lcycles");
    }

    #[test]
    fn sx_correction_falls_back_to_materializing_when_axis_does_not_match() {
        // A pending correction about Z is a different generator from SX's fixed X axis
        // and can't be folded in - it must still be materialized as its own gate first.
        let lines = &["+X___<SX>"];
        let mut sched = build_scheduler(lines, 0);
        sched.correction_power[0] = 1;
        sched.correction_basis[0] = 'Z';
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 1, "standalone correction gate injected");
        assert_eq!(sched.correction_power[0], 0);
        assert!(sched.free_merged_products.is_empty());
        assert!(sched.scheduled_products.contains(&0), "original SX product scheduled");
        assert!(sched.scheduled_products.contains(&1), "injected correction (id=1) scheduled");
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 6, "3 lcycles for the injected correction + 3 for the SX gate");
    }

    /// The lcycle in which product `id` was scheduled.
    /// Every lcycle product `id` occupies (several for an S or CX).
    fn lcycles_of(sched: &Scheduler, id: i32) -> Vec<usize> {
        let v: Vec<usize> = sched
            .lcycle_scheduled
            .iter()
            .filter(|(_, ids)| ids.contains(&id))
            .map(|(l, _)| *l)
            .collect();
        assert!(!v.is_empty(), "product {} was never scheduled", id);
        v
    }

    fn lcycle_of(sched: &Scheduler, id: i32) -> usize {
        sched
            .lcycle_scheduled
            .iter()
            .find(|(_, ids)| ids.contains(&id))
            .map(|(l, _)| *l)
            .unwrap_or_else(|| panic!("product {} was never scheduled", id))
    }

    #[test]
    fn classical_dependency_without_latency_costs_one_lcycle() {
        // Product 1 shares no qubit with product 0 but needs its outcome: it runs the lcycle after.
        let mut sched = build_scheduler(&["+Z___<M>", "+_Z__<M>@0"], 0);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(lcycle_of(&sched, 0), 1);
        assert_eq!(lcycle_of(&sched, 1), 2);
    }

    #[test]
    fn reaction_latency_delays_classically_dependent_products_exactly() {
        for latency in [1usize, 3, 7] {
            let mut sched = build_scheduler(&["+Z___<M>", "+_Z__<M>@0"], 0);
            sched.set_reaction_latency(latency);
            sched.sched_circuit().expect("sched_circuit failed");
            assert_eq!(lcycle_of(&sched, 0), 1);
            assert_eq!(lcycle_of(&sched, 1), 2 + latency, "latency {}", latency);
        }
    }

    // ── hosted ancillas (--ancilla-qubits) ────────────────────────────────────────

    fn hosted_scheduler(lines: &[&str], hosted: &[u16], rseed: u32) -> Scheduler {
        Node::set_magic_routing(true);
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        let fname = f.path().to_string_lossy().to_string();
        let mut circuit = Circuit::new(&fname);
        circuit.load_circuit().expect("circuit load failed");
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        topo.host_qubits(hosted).expect("host_qubits failed");
        Scheduler::new(circuit, topo, 0.0387396, "none", String::new(), rseed, false, false)
    }

    /// The first steps of `sched_circuit` with hosting on, without the scheduling loop.
    fn init_hosting_only(sched: &mut Scheduler) {
        sched.cultivation.set_lambda(sched.magic_state_lambda).unwrap();
        sched.precompute_terminals_and_roots();
        sched.precompute_multi_term_clifford_trees();
        sched.init_hosted();
        sched.init_magic_nodes();
        sched.mark_route_only_nodes();
        sched.cultivation.t_products_remaining = sched.count_t_products();
        sched.cultivation.fill_pool(1000);
    }

    const HOSTED_LINES: &[&str] = &[
        "+X___<T>",
        "-___X<T>",
        "+_X__<T>",
        "+XX__<CX>",
        "-__X_<T>",
        "+___Z<T>",
        "+_X_X<T>",
        "+X___<T>",
        "-_Z__<S>",
        "+__XX<CX>",
        "+X___<T>",
        "-_X__<T>",
        "+__Y_<T>",
    ];

    #[test]
    fn hosting_nothing_matches_stock_schedule() {
        let lines = HOSTED_LINES;
        for rseed in [0u32, 7, 29] {
            let stock = run_scheduler(lines, rseed);
            let mut empty = hosted_scheduler(lines, &[], rseed);
            empty.sched_circuit().unwrap();
            assert_eq!(stock.lcycle_scheduled, empty.lcycle_scheduled, "rseed {}", rseed);
        }
    }

    #[test]
    fn reaction_latency_leaves_independent_products_alone() {
        // Product 1 has no `@`: the latency must not touch it, nor a plain qubit-overlap edge.
        let mut sched = build_scheduler(&["+Z___<M>", "+_Z__<M>", "+Z___<M>"], 0);
        sched.set_reaction_latency(5);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(lcycle_of(&sched, 0), 1);
        assert_eq!(lcycle_of(&sched, 1), 1, "independent product runs in the first lcycle");
        assert_eq!(lcycle_of(&sched, 2), 2, "qubit-overlap edge is not delayed");
    }

    #[test]
    fn reaction_latency_waits_for_the_slowest_of_several_classical_parents() {
        // Product 2 needs products 0 and 1; 1 is itself delayed behind 0.
        let mut sched = build_scheduler(&["+Z___<M>", "+_Z__<M>@0", "+__Z_<M>@0,1"], 0);
        sched.set_reaction_latency(2);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(lcycle_of(&sched, 0), 1);
        assert_eq!(lcycle_of(&sched, 1), 4);
        assert_eq!(lcycle_of(&sched, 2), 7);
    }

    #[test]
    fn reaction_latency_scheduler_still_terminates_when_only_delayed_work_remains() {
        // After product 0 nothing is ready until the latency elapses; the loop must keep ticking.
        let mut sched = build_scheduler(&["+Z___<M>", "+_Z__<R>@0"], 0);
        sched.set_reaction_latency(10);
        let (tot, n) = sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(n, 2);
        assert_eq!(tot, 12);
    }

    #[test]
    fn dormant_hosted_qubit_cultivates_with_one_node() {
        // q3 has no products: hosted, it is a cultivator for the whole run.
        let lines = &["+X___<T>", "-_X__<T>", "+XX__<CX>", "+__X_<T>", "-X___<T>"];
        let mut sched = hosted_scheduler(lines, &[3], 1);
        sched.sched_circuit().unwrap();
        let [x_id, z_id] = sched.input.topo.data_ids[3];
        assert!(sched.cultivation.magic_node_ids.contains(&x_id), "X half must cultivate");
        assert!(!sched.cultivation.magic_node_ids.contains(&z_id), "Z half only routes");
        assert_eq!(sched.input.topo.node(x_id).node_type, NodeType::Magic);
        assert_eq!(sched.input.topo.node(z_id).node_type, NodeType::Magic);
        assert_eq!(sched.input.topo.cultivation_times[z_id as usize], NOT_CULTIVATING);
        assert_eq!(sched.hosted_activations(3), 0);
        assert_eq!(sched.hosted_live_lcycles(3), 0);
    }

    #[test]
    fn live_hosted_qubit_is_never_a_ready_magic_state() {
        let mut sched = hosted_scheduler(HOSTED_LINES, &[3], 2);
        init_hosting_only(&mut sched);
        let [x_id, z_id] = sched.input.topo.data_ids[3];
        // Let q3's cultivator become ready while dormant, then make q3 live.
        sched.input.topo.cultivation_times[x_id as usize] = 0;
        let n_ready_dormant = sched.update_cultivators();
        let x_pos = sched.input.topo.node(x_id).pos;
        assert!(sched.cultivation.ready_magic_positions.contains(&x_pos));
        sched.activate_hosted_for(&[1]); // product 1 is on q3
        assert!(sched.hosted.live[3]);
        assert_eq!(sched.input.topo.node(x_id).node_type, NodeType::Data);
        assert_eq!(sched.input.topo.node(z_id).node_type, NodeType::Data);
        for _ in 0..50 {
            let n_ready = sched.update_cultivators();
            assert!(!sched.cultivation.ready_magic_positions.contains(&x_pos));
            assert!(n_ready < sched.cultivation.magic_node_ids.len());
        }
        assert!(n_ready_dormant >= 1);
        assert!(!sched.input.topo.is_cultivating(x_id) && !sched.input.topo.is_cultivating(z_id));
    }

    #[test]
    fn retiring_hosted_qubit_restarts_cultivation() {
        let mut sched = hosted_scheduler(HOSTED_LINES, &[3], 3);
        init_hosting_only(&mut sched);
        let [x_id, z_id] = sched.input.topo.data_ids[3];
        sched.activate_hosted_for(&[1]);
        // Complete every product on q3 except the last: still live.
        let on_q3: Vec<i32> = (0..HOSTED_LINES.len() as i32)
            .filter(|&id| sched.input.circuit.product(id).operators.iter().any(|op| op.qubit == 3))
            .collect();
        sched.hosted.completed_buf.extend_from_slice(&on_q3[..on_q3.len() - 1]);
        sched.retire_hosted();
        assert!(sched.hosted.live[3]);
        sched.hosted.completed_buf.push(*on_q3.last().unwrap());
        sched.retire_hosted();
        assert!(!sched.hosted.live[3]);
        assert_eq!(sched.input.topo.node(x_id).node_type, NodeType::Magic);
        assert_eq!(sched.input.topo.busy_counts[x_id as usize], 0);
        assert!(sched.input.topo.cultivation_times[x_id as usize] >= 0, "fresh draw on X");
        assert_eq!(sched.input.topo.cultivation_times[z_id as usize], NOT_CULTIVATING);
    }

    const T_HEAVY_LINES: &[&str] = &[
        "+ZZ__<T>", "+X___<T>", "+_XX_<T>", "-Z__Z<T>", "+_Y__<T>", "+XXXX<T>", "+Z___<T>",
        "+__ZZ<T>", "-_X__<T>", "+X__X<T>", "+ZZZZ<T>", "+_Z_X<T>", "+X___<M>", "+_Z__<M>",
        "+__X_<M>", "+___Z<M>",
    ];

    #[test]
    fn litinski_leaves_every_later_product_unchanged() {
        for rseed in 0u32..8 {
            let mut sched = build_scheduler(T_HEAVY_LINES, rseed);
            sched.set_reaction_latency(2);
            let n_products = sched.input.circuit.n_products();
            let (_, n) = sched.sched_circuit().expect("sched_circuit failed");
            assert!(sched.t_gate_failures > 0, "seed {}: want some failures", rseed);
            assert_eq!(n, n_products, "seed {}: no correction products may be added", rseed);
            assert!(
                sched.t_conjugated_bases.is_empty(),
                "seed {}: a product was relabelled",
                rseed
            );
            assert!(sched.correction_power.iter().all(|&p| p == 0));
            assert_eq!(sched.t_stats.injections, 12);
        }
    }

    #[test]
    fn litinski_helpers_are_free_neighbours_and_holds_are_respected() {
        for latency in [0usize, 1, 3] {
            for rseed in 0u32..8 {
                let mut sched = build_scheduler(T_HEAVY_LINES, rseed);
                sched.set_reaction_latency(latency);
                sched.sched_circuit().expect("sched_circuit failed");
                let st = &sched.t_stats;
                assert_eq!(st.helpers + st.helpers_missing, st.injections);
                let topo = &sched.input.topo;
                for &(t, magic, h) in &st.helper_log {
                    assert!(topo.node(magic).nbs_slice().contains(&h), "helper not adjacent");
                    assert_ne!(topo.node(h).node_type, NodeType::Data, "helper on a data patch");
                    // The magic patch is held until its helper arrives (lcycle t); the helper
                    // until the decision d. Neither may serve another T as its magic state then.
                    let (t0, _, _, d) =
                        *st.injection_log.iter().rev().find(|e| e.2 == magic && e.0 <= t).unwrap();
                    for &(t1, pp, m1, _) in &st.injection_log {
                        if t1 > t0 && t1 <= t {
                            assert_ne!(m1, magic, "magic patch reused while held (T {})", pp);
                        }
                        if t1 >= t && t1 <= d {
                            assert_ne!(m1, h, "helper used as a magic state while held");
                        }
                    }
                }
                // Holds: one helper per injection for its decision delay, plus the lcycles a
                // magic patch waited for a helper (or held in place when none was found).
                let min_hold: usize = st.injection_log.iter().map(|&(t, _, _, d)| d - t).sum();
                assert!(
                    st.hold_patch_lcycles >= min_hold,
                    "L={} seed={} held {} < {} (helpers {} waited {} missing {})",
                    latency,
                    rseed,
                    st.hold_patch_lcycles,
                    min_hold,
                    st.helpers,
                    st.helpers_deferred,
                    st.helpers_missing
                );
                for &(t, pp, magic, d) in &st.injection_log {
                    assert!(d >= t + latency, "T {} decided before its outcome is decoded", pp);
                    assert_eq!(topo.node(magic).node_type, NodeType::Magic);
                }
            }
        }
    }

    #[test]
    fn inplace_y_holds_the_magic_patch_and_adds_y_time_for_failures() {
        for (latency, y) in [(0usize, 1usize), (1, 1), (3, 2)] {
            for rseed in 0u32..8 {
                let mut sched = build_scheduler(T_HEAVY_LINES, rseed);
                sched.set_t_correction(TCorrection::InplaceY);
                sched.set_reaction_latency(latency);
                sched.set_y_meas_lcycles(y);
                let n_products = sched.input.circuit.n_products();
                let (_, n) = sched.sched_circuit().expect("sched_circuit failed");
                let st = &sched.t_stats;
                assert_eq!(n, n_products);
                assert!(sched.t_conjugated_bases.is_empty());
                assert_eq!(st.helpers, 0, "in-place Y never borrows a helper");
                assert_eq!(st.inplace, st.injections);
                assert_eq!(st.inplace_y, sched.t_gate_failures, "one Y measurement per failure");
                // Each magic patch is held from injection to decision, plus y if it failed.
                let base: usize = st.injection_log.iter().map(|&(t, _, _, d)| d - t).sum();
                assert_eq!(st.hold_patch_lcycles, base + y * st.inplace_y);
                // A held magic patch never serves another T before its decision.
                for &(t0, _, m0, d0) in &st.injection_log {
                    for &(t1, pp, m1, _) in &st.injection_log {
                        if t1 > t0 && t1 <= d0 {
                            assert_ne!(m1, m0, "magic patch reused while held (T {})", pp);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn inplace_y_failure_delays_the_next_anticommuting_decision() {
        // Z then X on qubit 0: X's decision waits for Z's frame bit, which a Y measurement
        // (a failed Z) delivers y lcycles later.
        for rseed in 0u32..16 {
            let mut sched = build_scheduler(&["+Z___<T>", "+X___<T>"], rseed);
            sched.set_t_correction(TCorrection::InplaceY);
            sched.set_reaction_latency(2);
            sched.set_y_meas_lcycles(3);
            sched.sched_circuit().expect("sched_circuit failed");
            let log = &sched.t_stats.injection_log;
            let (t0, _, _, d0) = log[0];
            let (t1, _, _, d1) = log[1];
            assert_eq!(d0, t0 + 2);
            assert!(d1 == (t1 + 2).max(d0 + 2) || d1 == (t1 + 2).max(d0 + 3 + 2));
        }
    }

    #[test]
    fn litinski_helper_rank_prefers_bus_then_cultivating_then_ready() {
        let mut sched = build_scheduler(T_HEAVY_LINES, 0);
        sched.sched_circuit().expect("sched_circuit failed");
        let topo = &mut sched.input.topo;
        let magic: Vec<u16> = (0..topo.n_nodes as u16)
            .filter(|&n| topo.node(n).node_type == NodeType::Magic)
            .collect();
        let data =
            (0..topo.n_nodes as u16).find(|&n| topo.node(n).node_type == NodeType::Data).unwrap();
        let (a, b) = (magic[0], magic[1]);
        topo.cultivation_times[a as usize] = 0; // ready
        topo.cultivation_times[b as usize] = 7; // cultivating
        topo.busy_counts[b as usize] = 3;
        assert_eq!(sched.helper_rank(data), None, "a data patch is never a helper");
        let (ra, rb) = (sched.helper_rank(a).unwrap(), sched.helper_rank(b).unwrap());
        assert!(rb < ra, "a cultivating patch is preferred to wasting a ready state");
        sched.set_t_correction(TCorrection::LitinskiY);
        assert_eq!(sched.helper_rank(a), None, "litinski-y never borrows a ready state");
        assert!(sched.helper_rank(b).is_some());
    }

    #[test]
    fn litinski_y_never_wastes_a_ready_state_and_falls_back_in_place() {
        let (mut wasted_ready, mut any_inplace_or_wait) = (0, false);
        for latency in [0usize, 2] {
            for rseed in 0u32..8 {
                let mut y = build_scheduler(T_HEAVY_LINES, rseed);
                y.set_t_correction(TCorrection::LitinskiY);
                y.set_reaction_latency(latency);
                y.sched_circuit().expect("sched_circuit failed");
                let st = &y.t_stats;
                assert_eq!(st.helpers_ready, 0);
                assert_eq!(st.helpers + st.helpers_missing, st.injections);
                assert_eq!(st.inplace, st.helpers_missing);
                any_inplace_or_wait |= st.helpers_missing + st.helpers_deferred > 0;
                let mut r = build_scheduler(T_HEAVY_LINES, rseed);
                r.set_reaction_latency(latency);
                r.sched_circuit().expect("sched_circuit failed");
                wasted_ready += r.t_stats.helpers_ready;
            }
        }
        assert!(any_inplace_or_wait, "litinski-y should have to wait or fall back somewhere");
        assert!(wasted_ready > 0, "litinski-ready should borrow a ready state somewhere");
    }

    #[test]
    fn litinski_ready_never_borrows_a_hosted_patch() {
        for rseed in 0u32..4 {
            let mut sched = hosted_scheduler(HOSTED_LINES, &[3], rseed);
            sched.set_reaction_latency(1);
            sched.sched_circuit().expect("sched_circuit failed");
            for &(_, _, h) in &sched.t_stats.helper_log {
                assert!(!sched.input.topo.is_hosted(h), "helper on a hosted patch");
            }
        }
    }

    #[test]
    fn litinski_decisions_chain_only_through_anticommuting_t_gates() {
        // Z then X on qubit 0 anticommute: X's correction depends on Z's frame bit.
        let mut sched = build_scheduler(&["+Z___<T>", "+X___<T>"], 3);
        sched.set_reaction_latency(2);
        sched.sched_circuit().expect("sched_circuit failed");
        let log = &sched.t_stats.injection_log;
        assert_eq!(log.len(), 2);
        let (t0, _, _, d0) = log[0];
        let (t1, _, _, d1) = log[1];
        assert_eq!(d0, t0 + 2);
        assert_eq!(d1, (t1 + 2).max(d0 + 2));
        // Z then Z commute: each decision is just its own decoding delay.
        let mut sched = build_scheduler(&["+Z___<T>", "+Z___<T>"], 3);
        sched.set_reaction_latency(2);
        sched.sched_circuit().expect("sched_circuit failed");
        for &(t, _, _, d) in &sched.t_stats.injection_log {
            assert_eq!(d, t + 2);
        }
    }

    #[test]
    fn frame_mode_runs_every_product_once_and_rewrites_after_failures() {
        let n_tm =
            T_HEAVY_LINES.iter().filter(|l| l.ends_with("<T>") || l.ends_with("<M>")).count();
        let mut rewrote = false;
        for latency in [0usize, 2] {
            for rseed in 0u32..8 {
                let mut sched = build_scheduler(T_HEAVY_LINES, rseed);
                sched.set_t_correction(TCorrection::Frame);
                sched.set_reaction_latency(latency);
                let n_products = sched.input.circuit.n_products();
                let (_, n) = sched.sched_circuit().expect("sched_circuit failed");
                let f = &sched.frame_stats;
                assert_eq!(n, n_products, "no correction products in frame mode");
                assert_eq!(f.products, n_tm, "every T and measurement logged once");
                assert_eq!(f.failures, sched.t_gate_failures);
                assert_eq!(sched.correction_power.iter().filter(|&&p| p != 0).count(), 0);
                assert_eq!(sched.t_stats.injections, 0, "no helper machinery");
                if latency == 0 {
                    assert_eq!(f.waits, 0, "nothing waits when outcomes are known at once");
                }
                // Weights are counted from the log.
                let w: usize = f.log.iter().map(|(_, _, ops)| ops.len()).sum();
                assert_eq!(w, f.exec_weight);
                rewrote |= f.rewritten > 0;
            }
        }
        assert!(rewrote, "some seed must rewrite a product after a failure");
    }

    #[test]
    fn frame_mode_waits_for_an_anticommuting_outcome() {
        // T about Z then a measurement of X on the same qubit: with latency 3 the measurement
        // starts at least 4 lcycles after the T, whatever the T's outcome.
        for rseed in 0u32..8 {
            let mut sched = build_scheduler(&["+Z___<T>", "+X___<M>"], rseed);
            sched.set_t_correction(TCorrection::Frame);
            sched.set_reaction_latency(3);
            sched.sched_circuit().expect("sched_circuit failed");
            let log = &sched.frame_stats.log;
            let t_at = log.iter().find(|e| e.1 == 0).unwrap().0;
            let m_at = log.iter().find(|e| e.1 == 1).unwrap().0;
            assert!(m_at >= t_at + 4, "seed {}: T at {}, M at {}", rseed, t_at, m_at);
            // A commuting measurement need not wait.
            let mut sched = build_scheduler(&["+Z___<T>", "+Z___<M>"], rseed);
            sched.set_t_correction(TCorrection::Frame);
            sched.set_reaction_latency(3);
            sched.sched_circuit().expect("sched_circuit failed");
            assert_eq!(sched.frame_stats.waits, 0);
        }
    }

    #[test]
    fn frame_mode_without_failures_matches_legacy() {
        // With -F nothing is owed, so every model gives the same schedule.
        for rseed in 0u32..4 {
            let lc = |mode: TCorrection| {
                Node::set_magic_routing(true);
                let mut f = NamedTempFile::new().unwrap();
                for line in T_HEAVY_LINES {
                    writeln!(f, "{}", line).unwrap();
                }
                let mut circuit = Circuit::new(&f.path().to_string_lossy().to_string());
                circuit.load_circuit().unwrap();
                let mut topo = TopoGraph::new();
                topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
                let mut s = Scheduler::new(
                    circuit,
                    topo,
                    0.0387396,
                    "none",
                    String::new(),
                    rseed,
                    true,
                    false,
                );
                s.set_t_correction(mode);
                s.set_reaction_latency(2);
                s.sched_circuit().unwrap().0
            };
            assert_eq!(lc(TCorrection::Frame), lc(TCorrection::Legacy));
        }
    }

    /// One-qubit T gates about every axis between CXs in both directions, a multi-qubit T and
    /// measurements: exercises every hybrid path.
    const HYBRID_LINES: &[&str] = &[
        "+X___<T>",
        "+_Y__<T>",
        "+ZX__<CX>",
        "+Y___<T>",
        "+_Z__<T>",
        "+XZ__<CX>",
        "+Z___<T>",
        "+__X_<T>",
        "+_ZX_<CX>",
        "+XXZ_<T>",
        "+___Y<T>",
        "+__XZ<CX>",
        "+Y___<T>",
        "+ZX__<CX>",
        "+_Z__<S>",
        "+__X_<SX>",
        "+X___<M>",
        "+_Y__<M>",
        "+__ZX<M>",
        "+___X<M>",
    ];

    #[test]
    fn hybrid_keeps_the_frame_local_and_supports_unchanged() {
        let mut flushed = false;
        for latency in [0usize, 2] {
            for rseed in 0u32..12 {
                let mut sched = build_scheduler(HYBRID_LINES, rseed);
                sched.set_t_correction(TCorrection::Hybrid);
                sched.set_reaction_latency(latency);
                sched.sched_circuit().expect("sched_circuit failed");
                assert!(sched.frame.as_ref().unwrap().is_local(), "seed {}", rseed);
                let f = &sched.frame_stats;
                // A local frame changes letters, never supports.
                for (_, pp_id, ops) in &f.log {
                    let ideal: Vec<u16> = sched
                        .input
                        .circuit
                        .product(*pp_id)
                        .operators
                        .iter()
                        .map(|o| o.qubit)
                        .collect();
                    let exec: Vec<u16> = ops.iter().map(|o| o.qubit).collect();
                    assert_eq!(ideal, exec, "seed {}: product {} changed support", rseed, pp_id);
                }
                // The multi-qubit T goes to the helper, the one-qubit ones to the frame.
                assert_eq!(sched.t_stats.injections, 1);
                assert_eq!(f.flushes, sched.correction_gates_emitted);
                flushed |= f.flushes > 0;
            }
        }
        assert!(flushed, "some seed must need a flush rotation before a CX");
    }

    #[test]
    fn hybrid_runs_a_cx_only_after_its_flush() {
        // T about X on the control fails -> W Z_0 W^dag = Y_0, which does not commute with the CX:
        // a rotation runs on q0 first, and the CX starts after it ends.
        let mut checked = 0;
        for rseed in 0u32..16 {
            let mut sched = build_scheduler(&["+X___<T>", "+ZX__<CX>"], rseed);
            sched.set_t_correction(TCorrection::Hybrid);
            sched.sched_circuit().expect("sched_circuit failed");
            if sched.t_gate_failures == 0 {
                assert_eq!(sched.frame_stats.flushes, 0);
                continue;
            }
            checked += 1;
            assert_eq!(sched.frame_stats.flushes, 1, "a transposition needs one rotation");
            let flush_end = lcycles_of(&sched, 2).into_iter().max().unwrap();
            let cx_start = lcycles_of(&sched, 1).into_iter().min().unwrap();
            assert!(
                cx_start > flush_end,
                "seed {}: CX at {}, flush until {}",
                rseed,
                cx_start,
                flush_end
            );
        }
        assert!(checked > 0);
    }

    #[test]
    fn hybrid_without_failures_matches_legacy() {
        for rseed in 0u32..4 {
            let lc = |mode: TCorrection| {
                Node::set_magic_routing(true);
                let mut f = NamedTempFile::new().unwrap();
                for line in HYBRID_LINES {
                    writeln!(f, "{}", line).unwrap();
                }
                let mut circuit = Circuit::new(&f.path().to_string_lossy().to_string());
                circuit.load_circuit().unwrap();
                let mut topo = TopoGraph::new();
                topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
                let mut s = Scheduler::new(
                    circuit,
                    topo,
                    0.0387396,
                    "none",
                    String::new(),
                    rseed,
                    true,
                    false,
                );
                s.set_t_correction(mode);
                s.sched_circuit().unwrap().0
            };
            assert_eq!(lc(TCorrection::Hybrid), lc(TCorrection::Legacy));
        }
    }

    #[test]
    fn frame_mode_rejects_hosted_ancillas() {
        let mut sched = hosted_scheduler(HOSTED_LINES, &[3], 0);
        sched.set_t_correction(TCorrection::Frame);
        assert!(sched.sched_circuit().is_err());
    }

    #[test]
    fn legacy_mode_still_tracks_per_qubit_corrections() {
        // The old model stays available for comparison: some seed must relabel a product.
        let relabelled = (0u32..16).any(|rseed| {
            let lines = &["+Z___<T>", "+X___<T>", "+Y___<T>", "+Z___<T>"];
            let mut sched = build_scheduler(lines, rseed);
            sched.set_t_correction(TCorrection::Legacy);
            sched.sched_circuit().expect("sched_circuit failed");
            assert_eq!(sched.t_stats.injections, 0);
            !sched.t_conjugated_bases.is_empty()
        });
        assert!(relabelled);
    }

    #[test]
    fn hosted_schedules_satisfy_invariants() {
        // check_lcycle, check_schedule and check_hosted_lcycle all run in debug builds.
        for hosted in [&[0u16][..], &[3], &[1, 3], &[0, 1, 2, 3]] {
            for rseed in 0u32..6 {
                let mut sched = hosted_scheduler(HOSTED_LINES, hosted, rseed);
                let (lcycles, n) = sched.sched_circuit().expect("hosted schedule failed");
                assert!(n >= HOSTED_LINES.len());
                for &q in hosted {
                    assert!(!sched.hosted.live[q as usize], "q{} still live at the end", q);
                    assert_eq!(sched.hosted.remaining[q as usize], 0);
                    assert_eq!(sched.hosted_activations(q), 1, "static circuit: one live window");
                    assert!(sched.hosted_live_lcycles(q) <= lcycles);
                }
                let mut seen = std::collections::HashSet::new();
                for (_, ids) in &sched.lcycle_scheduled {
                    for &id in ids {
                        if (id as usize) < HOSTED_LINES.len()
                            && !sched.input.circuit.product(id).gate_type.is_clifford()
                        {
                            assert!(seen.insert(id), "product {} scheduled twice", id);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn hosted_qubit_goes_dormant_after_its_last_product() {
        // q3 is used only at the start; the rest of the circuit runs long after.
        let mut lines = vec!["+___X<T>", "-___Z<T>"];
        lines.extend(std::iter::repeat("+X___<T>").take(30));
        let mut sched = hosted_scheduler(&lines, &[3], 4);
        let (lcycles, _) = sched.sched_circuit().unwrap();
        let live = sched.hosted_live_lcycles(3);
        assert!(live > 0 && live < lcycles / 2, "live {} of {} lcycles", live, lcycles);
    }

    #[test]
    fn hosted_patch_sleeps_between_a_measurement_and_its_next_use() {
        // q3: T, Z-measure (releases the patch), then reused only after q0's long chain.
        let mut lines = vec!["+___X<T>", "+___Z<M>"];
        lines.extend(std::iter::repeat("+X___<T>").take(30));
        lines.push("+X__X<T>");
        let mut sched = hosted_scheduler(&lines, &[3], 5);
        let (lcycles, _) = sched.sched_circuit().expect("hosted schedule failed");
        assert_eq!(sched.hosted_activations(3), 2, "one window per use");
        let live = sched.hosted_live_lcycles(3);
        assert!(live < lcycles / 2, "live {} of {} lcycles", live, lcycles);
        assert!(!sched.hosted.live[3]);
    }

    #[test]
    fn hosted_patch_stays_live_through_a_y_or_multi_patch_measurement() {
        for meas in ["+___Y<M>", "+__ZZ<M>"] {
            let mut lines = vec!["+___X<T>", meas];
            lines.extend(std::iter::repeat("+X___<T>").take(30));
            lines.push("+X__X<T>");
            let mut sched = hosted_scheduler(&lines, &[3], 5);
            sched.sched_circuit().expect("hosted schedule failed");
            assert_eq!(sched.hosted_activations(3), 1, "{} must not release q3", meas);
        }
    }

    #[test]
    fn hosted_patch_stays_live_when_the_next_product_is_already_ready() {
        // The T after the measurement is ready as soon as the measurement completes.
        let lines = &["+___Z<M>", "+___X<T>", "+X___<T>"];
        let mut sched = hosted_scheduler(lines, &[3], 6);
        sched.sched_circuit().expect("hosted schedule failed");
        assert_eq!(sched.hosted_activations(3), 1);
    }

    #[test]
    fn hosted_patch_sleeps_through_reaction_latency_and_wakes_on_release() {
        // q3's next product needs q0's outcome: it is held back by the latency.
        let lines = &["+___Z<M>", "+Z___<M>", "+___X<T>@1"];
        let mut sched = hosted_scheduler(lines, &[3], 7);
        sched.set_reaction_latency(6);
        sched.sched_circuit().expect("hosted schedule failed");
        assert_eq!(sched.hosted_activations(3), 2);
        assert!(!sched.hosted.live[3]);
    }

    #[test]
    fn releasing_a_hosted_patch_drops_its_pending_correction() {
        let lines = &["+___X<T>", "+___Z<M>", "+X___<T>", "+X__X<T>"];
        let mut sched = hosted_scheduler(lines, &[3], 8);
        init_hosting_only(&mut sched);
        sched.activate_hosted_for(&[0]);
        // About Z, so the Z measurement stays a Z measurement and still releases the patch
        // (an S_X correction would make it a Y measurement, which does not).
        sched.correction_power[3] = 1;
        sched.correction_basis[3] = 'Z';
        sched.hosted.completed_buf.extend_from_slice(&[0, 1]); // T, then the releasing M
        sched.retire_hosted();
        assert!(!sched.hosted.live[3]);
        assert_eq!(sched.correction_power[3], 0);
    }

    // ── conditional products (`?id=v` suffix, --cond-outcomes) ────────────────────

    /// q0 is measured; q1 then gets an S if the outcome is 1, or a T if it is 0; an
    /// unconditional T on q1 follows both.
    const COND_CIRCUIT: [&str; 5] =
        ["+X___<T>", "+Z___<M>", "+_Z__<S>@1?1=1", "+_Z__<T>@1?1=0", "+_X__<T>"];

    fn cond_run(
        lines: &[&str], mode: CondOutcomes, p1: f64, latency: usize, rseed: u32,
    ) -> Scheduler {
        let mut sched = build_scheduler(lines, rseed);
        sched.set_reaction_latency(latency);
        sched.set_cond_outcomes(mode, p1);
        sched.sched_circuit().expect("sched_circuit failed");
        sched
    }

    fn ran(sched: &Scheduler, id: i32) -> bool {
        sched.lcycle_scheduled.iter().any(|(_, ids)| ids.contains(&id))
    }

    #[test]
    fn worst_mode_runs_every_conditional_product() {
        let sched = cond_run(&COND_CIRCUIT, CondOutcomes::Worst, 0.5, 2, 3);
        assert!(sched.skipped.is_empty());
        assert!(sched.outcomes.is_empty(), "worst mode samples nothing");
        assert!((0..5).all(|id| ran(&sched, id)));
    }

    #[test]
    fn random_mode_skips_exactly_the_failing_branch() {
        // p1 = 1: the outcome is 1, so the S (=1) runs and the T (=0) is skipped; p1 = 0: reverse.
        for (p1, runs, skips) in [(1.0, 2, 3), (0.0, 3, 2)] {
            let sched = cond_run(&COND_CIRCUIT, CondOutcomes::Random, p1, 2, 3);
            assert_eq!(sched.outcomes.get(&1).copied(), Some(p1 == 1.0));
            assert_eq!(sched.skipped, vec![skips], "p1 {}", p1);
            assert!(ran(&sched, runs) && !ran(&sched, skips), "p1 {}", p1);
            assert!(ran(&sched, 4), "the unconditional successor still runs");
        }
    }

    #[test]
    fn skipped_product_releases_its_successor_at_once() {
        // The conditional S takes 3 lcycles when it runs; skipped, the measurement after it on
        // q1 starts the lcycle the S is released (no lcycle is spent on the S).
        let lines = ["+Z___<M>", "+_Z__<S>@0?0=1", "+_X__<M>"];
        for latency in [0usize, 4] {
            let skip = cond_run(&lines, CondOutcomes::Random, 0.0, latency, 1);
            let worst = cond_run(&lines, CondOutcomes::Worst, 0.5, latency, 1);
            assert_eq!(skip.skipped, vec![1]);
            assert_eq!(lcycle_of(&worst, 1), 2 + latency);
            assert_eq!(lcycle_of(&skip, 2), 2 + latency, "latency {}", latency);
            assert_eq!(lcycle_of(&worst, 2), 5 + latency, "latency {}", latency);
        }
    }

    #[test]
    fn outcome_sampling_leaves_an_unconditional_schedule_unchanged() {
        // Outcomes use their own random stream, so cultivation and T-failure draws (and hence
        // the schedule of a circuit without conditions) are the same in every mode.
        let lines = ["+X___<T>", "+Z___<M>", "+_Z__<T>@1", "+XZ__<T>", "+__X_<T>", "+_Z_Z<T>"];
        let a = cond_run(&lines, CondOutcomes::Worst, 0.5, 2, 9);
        let b = cond_run(&lines, CondOutcomes::Random, 0.3, 2, 9);
        assert_eq!(a.lcycle_scheduled, b.lcycle_scheduled);
        assert_eq!(a.t_gate_failures, b.t_gate_failures);
    }

    #[test]
    fn random_outcomes_are_deterministic_per_seed() {
        let lines = ["+Z___<M>", "+_Z__<M>", "+__Z_<S>@0?0=1", "+___Z<S>@1?1=0", "+__X_<T>"];
        let a = cond_run(&lines, CondOutcomes::Random, 0.5, 1, 11);
        let b = cond_run(&lines, CondOutcomes::Random, 0.5, 1, 11);
        assert_eq!(a.outcomes, b.outcomes);
        assert_eq!(a.skipped, b.skipped);
        assert_eq!(a.lcycle_scheduled, b.lcycle_scheduled);
    }

    #[test]
    fn multi_term_condition_needs_every_term_and_samples_them_all() {
        // Runs only if q0 gave 1 and q1 gave 0. With p1 = 1 or 0 one term always fails.
        let lines = ["+Z___<M>", "+_Z__<M>", "+__Z_<S>@0,1?0=1,1=0", "+__X_<M>"];
        for p1 in [0.0, 1.0] {
            let sched = cond_run(&lines, CondOutcomes::Random, p1, 1, 2);
            assert_eq!(sched.skipped, vec![2], "p1 {}", p1);
            assert_eq!(sched.outcomes.len(), 2, "both terms sampled");
            assert!(ran(&sched, 3));
        }
    }

    #[test]
    #[should_panic(expected = "not in its @ list")]
    fn condition_outside_the_at_list_is_rejected() {
        build_scheduler(&["+Z___<M>", "+_Z__<S>?0=1"], 0);
    }

    #[test]
    #[should_panic(expected = "not a measurement")]
    fn condition_on_a_non_measurement_is_rejected() {
        build_scheduler(&["+Z___<T>", "+_Z__<S>@0?0=1"], 0);
    }

    #[test]
    fn skipped_product_on_a_dormant_hosted_qubit_keeps_it_dormant() {
        // q3's only product is a conditional S that is skipped: q3 never goes live.
        let mut lines = vec!["+Z___<M>", "+___Z<S>@0?0=1"];
        lines.extend(std::iter::repeat("+X___<T>").take(10));
        let mut sched = hosted_scheduler(&lines, &[3], 6);
        sched.set_cond_outcomes(CondOutcomes::Random, 0.0);
        sched.sched_circuit().expect("hosted schedule failed");
        assert_eq!(sched.skipped, vec![1]);
        assert_eq!(sched.hosted_activations(3), 0);
        assert_eq!(sched.hosted.remaining[3], 0);
    }

    #[test]
    fn hosted_qubit_wakes_for_the_product_after_a_skipped_one() {
        // q3: a skipped conditional S, then a real T: one live window, for the T only.
        let mut lines = vec!["+Z___<M>", "+___Z<S>@0?0=1", "+___X<T>"];
        lines.extend(std::iter::repeat("+X___<T>").take(10));
        let mut sched = hosted_scheduler(&lines, &[3], 6);
        sched.set_cond_outcomes(CondOutcomes::Random, 0.0);
        sched.sched_circuit().expect("hosted schedule failed");
        assert_eq!(sched.skipped, vec![1]);
        assert!(ran(&sched, 2));
        assert_eq!(sched.hosted_activations(3), 1);
        assert_eq!(sched.hosted.remaining[3], 0);
        assert!(!sched.hosted.live[3]);
    }

    // ── measurements under a pending T-failure correction ───────────────────────────

    /// Runs `lines` with a pending S correction (power 1, about `axis`) on qubit `q` at the start,
    /// under the legacy model (the only one that keeps per-qubit corrections).
    fn run_with_pending(lines: &[&str], q: usize, axis: char) -> Scheduler {
        let mut sched = build_scheduler(lines, 0);
        sched.set_t_correction(TCorrection::Legacy);
        sched.correction_power[q] = 1;
        sched.correction_basis[q] = axis;
        sched.sched_circuit().expect("sched_circuit failed");
        sched
    }

    #[test]
    fn pending_correction_turns_an_x_measurement_into_y_and_stays_pending() {
        let sched = run_with_pending(&["+X___<M>"], 0, 'Z');
        assert!(sched.conjugated_t_products.contains(&0), "S_Z conjugates X to Y");
        assert_eq!(sched.precomputed_terminals[0].len(), 2, "a Y measurement uses both nodes");
        assert_eq!(lcycle_of(&sched, 0), 1);
        assert_eq!(sched.correction_power[0], 1, "measuring does not discharge the correction");
    }

    #[test]
    fn pending_correction_about_the_same_axis_leaves_a_measurement_alone() {
        let sched = run_with_pending(&["+Z___<M>"], 0, 'Z');
        assert!(!sched.conjugated_t_products.contains(&0));
        assert_eq!(sched.precomputed_terminals[0].len(), 1);
    }

    #[test]
    fn multi_qubit_measurement_is_routed_live_on_its_conjugated_basis() {
        // Its precomputed tree is for X⊗Z; with S_Z pending on q0 it is measured as Y⊗Z.
        let sched = run_with_pending(&["+XZ__<M>"], 0, 'Z');
        assert!(sched.conjugated_t_products.contains(&0));
        assert_eq!(sched.precomputed_terminals[0].len(), 3, "Y on q0 (2 nodes) + Z on q1");
        assert_eq!(lcycle_of(&sched, 0), 1);
    }

    #[test]
    fn reinitialization_voids_a_pending_correction() {
        let sched = run_with_pending(&["+Z___<M>", "+Z___<R>"], 0, 'X');
        assert_eq!(sched.correction_power[0], 0);
    }

    #[test]
    fn hosted_patch_is_not_released_by_an_effective_y_measurement() {
        // q3: X measurement, then a T after q0's long chain. Plain, the measurement releases the
        // patch (two live windows); with S_Z pending on q3 it is a Y measurement and does not.
        let mut lines = vec!["+___X<M>"];
        lines.extend(std::iter::repeat("+X___<T>").take(30));
        lines.push("+X__X<T>");
        let run = |pending: bool| {
            let mut sched = hosted_scheduler(&lines, &[3], 5);
            if pending {
                sched.correction_power[3] = 1;
                sched.correction_basis[3] = 'Z';
            }
            sched.sched_circuit().expect("hosted schedule failed");
            sched.hosted_activations(3)
        };
        assert_eq!(run(false), 2);
        assert_eq!(run(true), 1);
    }
}
