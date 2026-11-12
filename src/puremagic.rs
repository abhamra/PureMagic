use clap::Parser;
use rand::SeedableRng;
use rand::rngs::StdRng;

mod astar;
mod circuit;
mod cultivation;
mod frame;
mod node;
mod pauliproduct;
mod scheduler;
mod steinertree;
mod topograph;
mod topograph_plotter;
mod treegraph;
#[macro_use]
mod utils;

use circuit::Circuit;
use scheduler::Scheduler;
use topograph::TopoGraph;
use utils::{CommonArgs, Timer};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,
    /// Randomize data qubit numbering.
    #[arg(short = 'R', long)]
    randomize_data_qubits: bool,
    /// Name of file containing topology. If this is not set, it will be generated.
    #[arg(short, long = "topo", default_value = "")]
    topo_fname: String,
    /// Show product IDs instead of Pauli terms when plotting the circuit.
    #[arg(short = 'I', long)]
    show_product_ids: bool,
    /// Log scheduler actions to <CIRCUIT_FNAME>.sched_trace. Only populated in debug
    /// builds: the debug_sched!/info_sched! call sites are compiled out entirely in
    /// release builds, so a release build still creates the file but leaves it empty.
    #[arg(
        short = 'l',
        long = "log-scheduler",
        default_value = "none",
        value_parser = |s: &str| {
            match s.to_lowercase().as_str() {
                "none" | "info" | "debug" => Ok(s.to_string()),
                _ => Err(format!(
                    "invalid log level '{}'; must be one of: none, info, debug",
                    s
                ))
            }
        },
        help = "Log level for scheduler (none, info, or debug); debug builds only"
    )]
    log_scheduler: String,
    /// Use magic qubits for routing in addition to bus qubits
    #[arg(short = 'u', long)]
    use_magic_routing: bool,
    /// Use only the sides of data qubits for edges, not the top and bottom
    #[arg(short = 'S', long = "sides_only")]
    sides_only: bool,
    /// Logical qubits to host on injection patches, e.g. `3,4` (pure-magic mode, -u, only).
    /// A hosted qubit's patch cultivates magic states while the qubit is idle: it is live
    /// from when its first product is ready until its last one completes. A listed qubit
    /// that no product touches stays dormant for the whole run.
    #[arg(long = "ancilla-qubits", value_delimiter = ',', num_args = 1..)]
    ancilla_qubits: Vec<u16>,
    /// Record normalized cultivation-time distribution to <CIRCUIT_FNAME>.cultivation_dist
    #[arg(short = 'C', long)]
    record_cultivation_dist: bool,
    /// Classical reaction latency in lcycles, for dynamic circuits: a product whose `.trans` line
    /// has an `@id,..` suffix starts this many lcycles after the products it names complete,
    /// modelling the time to decode a logical measurement before acting on its outcome. 0 = ideal.
    #[arg(long = "reaction-latency", default_value_t = 0)]
    reaction_latency: usize,
    /// How a failed T injection is corrected: `litinski` (auto-corrected pi/8 rotation: a borrowed
    /// |0> helper next to the magic patch, held until the correction is decided; nothing owed on
    /// the data), `inplace-y` (no helper: the magic patch is held until the decision and then
    /// measured in X, or in Y for a failed T), `frame` (no correction: an exact Clifford frame
    /// rewrites every later T and measurement; they wait for undecoded outcomes they depend on),
    /// `hybrid` (frame for one-qubit T gates, kept local by rotations before CX; helper for
    /// multi-qubit T gates)
    /// or `legacy` (the old per-qubit S record, which is
    /// incorrect for multi-qubit T gates and for failures about different axes; kept to compare
    /// against).
    #[arg(long = "t-correction", value_enum, default_value_t = scheduler::TCorrection::Litinski)]
    t_correction: scheduler::TCorrection,
    /// Extra lcycles an in-place Y measurement of a magic patch takes (`inplace-y`, and
    /// `litinski` injections that found no helper). Gidney's in-place Y measurement takes
    /// floor(d/2)+2 rounds, about half an lcycle; the default rounds it up to 1.
    #[arg(long = "y-meas-lcycles", default_value_t = 1)]
    y_meas_lcycles: usize,
    /// Conditional products (a `?id=v,..` suffix in the `.trans` file): `random` samples each
    /// measurement's outcome and skips products whose condition fails, so the schedule is that
    /// of the circuit those outcomes call for; `worst` runs them all (an upper bound).
    #[arg(long = "cond-outcomes", value_enum, default_value_t = scheduler::CondOutcomes::Random)]
    cond_outcomes: scheduler::CondOutcomes,
    /// Probability that a measurement's true outcome is 1, for `--cond-outcomes random`.
    #[arg(long = "cond-p1", default_value_t = 0.5, value_parser = |s: &str| {
        let p: f64 = s.parse().map_err(|_| format!("'{}' is not a number", s))?;
        if (0.0..=1.0).contains(&p) { Ok(p) } else { Err(format!("{} is not in [0, 1]", p)) }
    })]
    cond_p1: f64,
    #[arg(
        short,
        long,
        value_delimiter = ',',
        value_parser = |s: &str| {
            match s.to_lowercase().as_str() {
                "topo" | "circuit" | "coupling" | "cstats" | "paths" | "" => Ok(s.to_string()),
                _ => Err(format!(
                    "invalid plot option '{}'; must be one of: topo, circuit, cstats, paths",
                    s
                ))
            }
        },
        default_value = "",
        help = format!("Plot options (one or more):\n{}{}{}{}",
        "  topo:     plot topology in <CIRCUIT_FNAME>.topo.png\n",
        "  circuit:  plot full circuit in files in subdirectory <CIRCUIT_FNAME>.circuit\n",
        "  cstats:   plot circuit statistics over time in <CIRCUIT_FNAME>.layer_stats.svg\n",
        "  paths:    plot paths for first 100 lcycles in subdirectory <CIRCUIT_FNAME>.paths")
    )]
    plot: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _timer = Timer::new("main");
    let args = Args::parse();
    let mut hdr = utils::print_banner("PureMagic");
    println!("{:#?}", args);
    hdr = format!("# {}\n# {:?}", &hdr, args);
    let circuit_fname = args.common.circuit_fname;
    let mut circuit = Circuit::new(&circuit_fname);
    circuit.load_circuit()?;
    let n_products = circuit.n_products();
    let _n_layers = circuit.print_statistics();
    #[cfg(debug_assertions)]
    circuit.print()?;
    if args.plot.contains(&"circuit".to_string()) {
        circuit.plot(args.show_product_ids)?;
    }
    if args.plot.contains(&"coupling".to_string()) {
        circuit.plot_qubit_coupling()?;
    }
    if args.plot.contains(&"cstats".to_string()) {
        circuit.plot_layer_stats()?;
    }
    if !args.ancilla_qubits.is_empty() && !args.use_magic_routing {
        return Err("--ancilla-qubits needs pure-magic mode (-u): a hosted patch is a \
                    magic-state cultivator while idle, and only -u routes through those"
            .into());
    }
    if let Some(&q) = args.ancilla_qubits.iter().find(|&&q| q as usize >= circuit.n_qubits) {
        return Err(format!(
            "--ancilla-qubits: qubit {} is outside the circuit ({} qubits)",
            q, circuit.n_qubits
        )
        .into());
    }
    let mut topo_graph = TopoGraph::new();
    let rseed: u32 = if args.randomize_data_qubits { args.common.rseed as u32 } else { 0 };
    let n_data_qubits = circuit.n_qubits;
    topo_graph.set_topo(
        n_data_qubits,
        &circuit_fname.to_string(),
        &args.topo_fname,
        &rseed,
        args.use_magic_routing,
        args.common.ancilla_rows,
        args.sides_only,
    );
    if !args.ancilla_qubits.is_empty() {
        topo_graph.host_qubits(&args.ancilla_qubits)?;
        println!("Hosted ancilla qubits (injection patches): {:?}", topo_graph.hosted_qubits);
    }
    if args.plot.contains(&"topo".to_string()) {
        topo_graph.plot(".topo", &[], "")?;
        topo_graph.print()?;
    }
    let n_qubits = topo_graph.n_qubits;
    let n_magic_qubits = topo_graph.n_magic_qubits;
    let mut sched = Scheduler::new(
        circuit,
        topo_graph,
        args.common.magic_state_lambda,
        &args.log_scheduler,
        args.plot.join(" "),
        args.common.rseed as u32,
        args.common.no_t_failures,
        args.record_cultivation_dist,
    );
    sched.set_reaction_latency(args.reaction_latency);
    sched.set_cond_outcomes(args.cond_outcomes, args.cond_p1);
    sched.set_t_correction(args.t_correction);
    sched.set_y_meas_lcycles(args.y_meas_lcycles);

    let (tot_lcycles, n_scheduled) = sched.sched_circuit()?;
    assert!(n_scheduled + sched.skipped.len() >= n_products);
    let volume = n_qubits * tot_lcycles;
    println!("Scheduled {} in {} logical cycles, volume {}", n_scheduled, tot_lcycles, volume);
    // Estimates and ratios below describe the circuit that ran: conditional products the
    // scheduler skipped are left out entirely.
    let skipped: std::collections::HashSet<i32> = sched.skipped.iter().copied().collect();
    let ran_circuit = (!skipped.is_empty()).then(|| sched.input.circuit.without(&skipped));
    let ran = ran_circuit.as_ref().unwrap_or(&sched.input.circuit);
    let n_products = ran.n_products();
    let n_sx_cliffords =
        ran.pps.iter().filter(|pp| pp.gate_type.is_s() || pp.gate_type.is_sx()).count();
    if !args.common.no_t_failures && n_sx_cliffords > 0 {
        let corrections = sched.correction_gates_emitted;
        println!(
            "S correction gates: {} / {} S/SX Cliffords ({:.1}%)",
            corrections,
            n_sx_cliffords,
            corrections as f64 * 100.0 / n_sx_cliffords as f64
        );
    }
    println!("Parallelism: {:.3}x", n_scheduled as f64 / tot_lcycles as f64);
    for &q in &sched.input.topo.hosted_qubits {
        let live = sched.hosted_live_lcycles(q);
        println!(
            "Hosted ancilla q{}: live {} / {} lcycles ({:.1}%), went live {} time(s)",
            q,
            live,
            tot_lcycles,
            100.0 * live as f64 / tot_lcycles.max(1) as f64,
            sched.hosted_activations(q)
        );
    }

    let mut rng = StdRng::seed_from_u64(args.common.rseed);
    let est = ran.estimate_layer_volume(
        n_magic_qubits,
        n_qubits,
        args.common.magic_state_lambda,
        args.common.no_t_failures,
        &mut rng,
    );
    let max_parallelism_estimate = (n_products + est.n_t_gates / 2) as f64 / est.lmin as f64;
    println!("Max parallelism estimate: {:.3}", max_parallelism_estimate);
    println!("Volume estimate: {}", est.vmin);
    println!("Normalized scheduling efficiency: {:.3}", (est.vmin as f64 / volume as f64).min(1.0));

    sched.print_schedule(&hdr)?;
    Ok(())
}
