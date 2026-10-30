#!/usr/bin/env -S cargo run --bin transpile --
//! Transpile a Clifford+T circuit (QASM) to Pauli basis measurements (.trans).
//!
//! Input must be a `.cliffordt.qasm` file produced by one of the Clifford+T
//! compilers in `data_processing/`.

use clap::Parser;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[allow(dead_code)]
mod pauliproduct;
mod tableau;
#[allow(dead_code)]
#[macro_use]
mod utils;

use tableau::{Gate1Q, Gate2Q, PauliString, Tableau};
use utils::Timer;

#[derive(Parser, Debug)]
#[command(author, version, about = "Transpile a Clifford+T circuit to Pauli basis measurements")]
struct Args {
    #[arg(short, long = "input_file")]
    input_file: String,

    /// Defaults to the input stem with a .trans<max_width> suffix.
    #[arg(short, long = "output_file", default_value = "")]
    output_file: String,

    /// Maximum Pauli product weight; 0 = no limit. Ignored when --auto is set.
    #[arg(short = 'm', long = "max_width", default_value = "0")]
    max_width: i32,

    /// Choose the max Pauli-product weight automatically, by transpiling at each candidate
    /// weight and keeping the one with the lowest predicted circuit depth. Ignores --max_width.
    #[arg(long = "auto", default_value_t = false)]
    auto: bool,

    /// Hold back each qubit's trailing single-qubit Clifford run at a flush. Orthogonal to the
    /// weight threshold; always on under --auto.
    #[arg(long = "defer_trailing", default_value_t = false)]
    defer_trailing: bool,

    /// How a mid-circuit `reset` is compiled (see docs/dynamic-circuits.md).
    ///   frame: measure, then fix the state with a classical Pauli-frame update. No extra gates.
    ///   flush: like `frame`, but first flush the pending Cliffords so the reset's measurement
    ///          is weight 1 and later products on that qubit start from a clean frame.
    ///   init:  like `flush`, plus an `<R>` product that models re-initializing the patch.
    #[arg(long = "reset_mode", value_enum, default_value_t = ResetMode::Frame)]
    reset_mode: ResetMode,

    /// Write the classical sidecar `<output>.dyn.json` even for a circuit with no reset or
    /// feed-forward (it is written automatically for dynamic circuits).
    #[arg(long = "sidecar", default_value_t = false)]
    sidecar: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum ResetMode {
    Frame,
    Flush,
    Init,
}

#[derive(Debug, Clone)]
enum QasmGate {
    Clifford1Q { gate: Gate1Q, qubit: usize },
    Clifford2Q { gate: Gate2Q, control: usize, target: usize },
    T { qubit: usize },
    Tdg { qubit: usize },
    /// Z-basis measurement. `clbit` is the global classical-bit index it writes, if any.
    Measure { qubit: usize, clbit: Option<usize> },
    /// `reset q[i];` -- Z measurement followed by a classical X correction (Pauli frame).
    Reset { qubit: usize },
    /// A Pauli gate applied only when classical bit `clbit` equals `value`, i.e. the OpenQASM 2
    /// statement `if(c==value) x q[i];` on a 1-bit register. It never touches the layout: it is a
    /// classical Pauli-frame update that depends on an earlier measurement outcome.
    CondPauli { gate: Gate1Q, qubit: usize, clbit: usize, value: bool },
    /// Any other single-qubit gate (Clifford or T) applied only when *every* `(clbit, value)` term
    /// holds, i.e. `if(c0==1 && c1==0) h q[i];` (the `&&` form is a PureMagic extension of
    /// OpenQASM 2; each register is 1 bit wide). A Pauli under a single term is a
    /// [`QasmGate::CondPauli`] instead. Unlike that, this gate is a *physical* operation whose
    /// execution depends on outcomes, so it cannot live in the Pauli frame.
    CondGate { inner: Box<QasmGate>, terms: Vec<(usize, bool)> },
    Barrier,
}

impl QasmGate {
    fn qubits(&self) -> Vec<usize> {
        match self {
            QasmGate::Clifford1Q { qubit, .. } => vec![*qubit],
            QasmGate::Clifford2Q { control, target, .. } => {
                let mut v = vec![*control, *target];
                v.sort_unstable();
                v
            }
            QasmGate::T { qubit } => vec![*qubit],
            QasmGate::Tdg { qubit } => vec![*qubit],
            QasmGate::Measure { qubit, .. } => vec![*qubit],
            QasmGate::Reset { qubit } => vec![*qubit],
            QasmGate::CondPauli { qubit, .. } => vec![*qubit],
            QasmGate::CondGate { inner, .. } => inner.qubits(),
            QasmGate::Barrier => vec![],
        }
    }

    /// The classical bits this gate writes (`Measure`) or reads (`CondPauli`, `CondGate`).
    fn clbits(&self) -> Vec<usize> {
        match self {
            QasmGate::Measure { clbit, .. } => clbit.iter().copied().collect(),
            QasmGate::CondPauli { clbit, .. } => vec![*clbit],
            QasmGate::CondGate { terms, .. } => terms.iter().map(|&(c, _)| c).collect(),
            _ => vec![],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Sign {
    Plus,
    Minus,
}

impl fmt::Display for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sign::Plus => write!(f, "+"),
            Sign::Minus => write!(f, "-"),
        }
    }
}

#[derive(Debug, Clone)]
struct TransPauli {
    sign: Sign,
    /// One char per qubit: 'I', 'X', 'Y', 'Z'.
    ops: Vec<char>,
    /// "T", "M" or "R".
    label: String,
    /// Dynamic-circuit bookkeeping. Not part of the `.trans` line; written to the sidecar.
    meta: PauliMeta,
}

/// Classical information attached to a product, for circuits with mid-circuit measurement.
///
/// The scheduler only reads the product's support and basis, never its sign, so none of this
/// changes the schedule. It lets a downstream executor (or `validate_dynamic.py`) recover the
/// true outcome of every measurement: with `p` the physical outcome of product `k`,
///
/// ```text
/// true_outcome(k) = p  XOR  sign(k)  XOR  (XOR of true_outcome(j) for j in frame_deps(k))
/// ```
#[derive(Debug, Clone, Default)]
struct PauliMeta {
    /// Index of the outcome this `M` product produces (counted over all `M` products).
    meas_id: Option<usize>,
    /// Circuit-level qubit that was measured (the product itself is in the physical frame).
    logical_qubit: Option<usize>,
    /// Global classical-bit index the outcome was stored in, if the circuit named one.
    clbit: Option<usize>,
    /// True when this `M` is the measurement half of a `reset`.
    is_reset: bool,
    /// Earlier outcomes whose XOR flips this product's sign.
    frame_deps: Vec<usize>,
    /// For a *conditional* product: it executes only if every `(outcome id, value)` holds, where
    /// outcome ids are true outcomes as defined above. Empty for an unconditional product.
    cond: Vec<(usize, bool)>,
    /// Outcomes this product needs before it can start (its conditions, plus any owed Pauli that
    /// was paid for physically just before it). Written to the `.trans` line as `@line,..` so the
    /// scheduler orders it after the measurements that decide it.
    needs: Vec<usize>,
}

impl TransPauli {
    /// Attach the frame's dependencies to this product: constants flip the printed sign, the
    /// outcome-dependent rest goes to `meta.frame_deps`.
    fn apply_frame(&mut self, mut deps: Vec<usize>) {
        if take_const(&mut deps) {
            self.sign = match self.sign {
                Sign::Plus => Sign::Minus,
                Sign::Minus => Sign::Plus,
            };
        }
        self.meta.frame_deps = deps;
    }

    fn from_pauli_string(ps: &PauliString, label: &str) -> Self {
        let sign = if ps.sign { Sign::Minus } else { Sign::Plus };
        let ops: Vec<char> = (0..ps.n).map(|q| ps.pauli_at(q)).collect();
        TransPauli { sign, ops, label: label.to_string(), meta: PauliMeta::default() }
    }
}

/// Pseudo outcome id for frame terms that are *always* owed, i.e. XOR-ed with the constant 1.
///
/// The emitted Cliffords equal the true ones only up to a Pauli (X/Y/Z are never emitted; see
/// [`dropped_pauli`]). That constant Pauli joins the frame under this id, and is folded into the
/// printed sign of every later product instead of appearing in `frame_deps`.
const CONST_OUTCOME: usize = usize::MAX;

/// Removes [`CONST_OUTCOME`] from `deps`, returning whether it was present (an odd number of
/// constant terms flips the sign).
fn take_const(deps: &mut Vec<usize>) -> bool {
    let before = deps.len();
    deps.retain(|&d| d != CONST_OUTCOME);
    deps.len() != before
}

/// A classical Pauli frame: the Pauli correction `prod_q X_q^{a_q} Z_q^{b_q}` that has been
/// owed to the physical state, where each exponent is the XOR of a set of measurement outcomes.
///
/// `dx[q]` is the set of outcomes in `a_q`, `dz[q]` the set in `b_q`. A later Pauli product `P`
/// measures `+-P` instead of `P` exactly when it anticommutes with the frame, so its sign
/// depends on the XOR of the outcomes returned by [`PauliFrame::deps`]. Cliffords the transpiler
/// emits physically are pushed through the frame with [`PauliFrame::conj_gate`]; Cliffords it
/// keeps pending need no update because the frame is stored in the same (physical) coordinates
/// as the products.
#[derive(Debug, Clone)]
struct PauliFrame {
    dx: Vec<BTreeSet<usize>>,
    dz: Vec<BTreeSet<usize>>,
}

fn toggle(set: &mut BTreeSet<usize>, v: usize) {
    if !set.remove(&v) {
        set.insert(v);
    }
}

fn xor_into(dst: &mut BTreeSet<usize>, src: &BTreeSet<usize>) {
    for &v in src {
        toggle(dst, v);
    }
}

impl PauliFrame {
    fn new(n: usize) -> Self {
        PauliFrame { dx: vec![BTreeSet::new(); n], dz: vec![BTreeSet::new(); n] }
    }

    /// Owe the physical state the Pauli `p` whenever `outcome` is 1. The sign of `p` is ignored.
    fn add(&mut self, p: &PauliString, outcome: usize) {
        for q in 0..p.n {
            if p.x_bits[q] {
                toggle(&mut self.dx[q], outcome);
            }
            if p.z_bits[q] {
                toggle(&mut self.dz[q], outcome);
            }
        }
    }

    /// The outcomes whose XOR decides whether `p` anticommutes with the frame (sorted).
    fn deps(&self, p: &PauliString) -> Vec<usize> {
        let mut acc = BTreeSet::new();
        for q in 0..p.n.min(self.dx.len()) {
            if p.x_bits[q] {
                xor_into(&mut acc, &self.dz[q]); // X anticommutes with the Z part of the frame
            }
            if p.z_bits[q] {
                xor_into(&mut acc, &self.dx[q]); // Z anticommutes with the X part
            }
        }
        acc.into_iter().collect()
    }

    /// Push the frame forward through a Clifford `g` that was applied to the physical state.
    /// Pauli signs are irrelevant here, so `S`/`Sdg` and `SX`/`SXdg` act identically.
    fn conj_gate(&mut self, g: &QasmGate) {
        match g {
            QasmGate::Clifford1Q { gate, qubit } => {
                let q = *qubit;
                match gate {
                    Gate1Q::H => std::mem::swap(&mut self.dx[q], &mut self.dz[q]),
                    Gate1Q::S | Gate1Q::Sdg => {
                        let dx = self.dx[q].clone();
                        xor_into(&mut self.dz[q], &dx);
                    }
                    Gate1Q::SX | Gate1Q::SXdg => {
                        let dz = self.dz[q].clone();
                        xor_into(&mut self.dx[q], &dz);
                    }
                    Gate1Q::X | Gate1Q::Y | Gate1Q::Z => {}
                }
            }
            QasmGate::Clifford2Q { gate, control, target } => {
                let (c, t) = (*control, *target);
                match gate {
                    Gate2Q::CX => {
                        let dxc = self.dx[c].clone();
                        xor_into(&mut self.dx[t], &dxc);
                        let dzt = self.dz[t].clone();
                        xor_into(&mut self.dz[c], &dzt);
                    }
                    Gate2Q::CZ => {
                        let (dxc, dxt) = (self.dx[c].clone(), self.dx[t].clone());
                        xor_into(&mut self.dz[t], &dxc);
                        xor_into(&mut self.dz[c], &dxt);
                    }
                    Gate2Q::Swap => {
                        self.dx.swap(c, t);
                        self.dz.swap(c, t);
                    }
                }
            }
            _ => {}
        }
    }
}

impl fmt::Display for TransPauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sign)?;
        for &c in &self.ops {
            if c == 'I' {
                write!(f, "_")?;
            } else {
                write!(f, "{}", c)?;
            }
        }
        write!(f, "<{}>", self.label)
    }
}

#[derive(Debug, Clone)]
struct TransClifford {
    /// One char per qubit: '_', 'X', or 'Z'.
    ops: Vec<char>,
    /// Gate name: "CX", "S", "Sdg", "SX", "SXdg".
    name: String,
    /// Only `cond` and `needs` are used: set for the physical Cliffords of a conditional gate.
    meta: PauliMeta,
}

impl TransClifford {
    /// Returns `None` for gates not emitted in the .trans format (X, Z, H, Y, CZ, SWAP).
    fn from_qasm_gate(gate: &QasmGate, n_qubits: usize) -> Option<Self> {
        let mut ops = vec!['_'; n_qubits];
        let name = match gate {
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control, target } => {
                ops[*control] = 'Z';
                ops[*target] = 'X';
                "CX".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::S, qubit } => {
                ops[*qubit] = 'Z';
                "S".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::Sdg, qubit } => {
                ops[*qubit] = 'Z';
                "Sdg".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::SX, qubit } => {
                ops[*qubit] = 'X';
                "SX".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::SXdg, qubit } => {
                ops[*qubit] = 'X';
                "SXdg".to_string()
            }
            _ => return None,
        };
        Some(TransClifford { ops, name, meta: PauliMeta::default() })
    }
}

impl fmt::Display for TransClifford {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+")?;
        for &c in &self.ops {
            write!(f, "{}", c)?;
        }
        write!(f, "<{}>", self.name)
    }
}

/// Classical bookkeeping that is not a scheduled operation: it has no `.trans` line and costs no
/// logical cycles. It goes to the sidecar, anchored to the number of `.trans` lines before it.
#[derive(Debug, Clone)]
enum Event {
    /// Physically apply `X^(xor of x_deps, plus 1 if x_const) Z^(...)` to `qubit`. This pays off
    /// the Pauli the frame owes that qubit, so that a following *conditional* physical gate sees
    /// a clean frame (a conditional Clifford would otherwise make the frame nonlinear in the
    /// outcomes). A Pauli on a surface-code patch is a transversal/software operation, so it is
    /// not scheduled.
    Fix { qubit: usize, x_deps: Vec<usize>, x_const: bool, z_deps: Vec<usize>, z_const: bool },
    /// Defines the pseudo-outcome `id` as the AND of its `(outcome, value)` terms, so that a
    /// Pauli conditioned on several bits is still a *linear* frame term.
    Derived { id: usize, terms: Vec<(usize, bool)> },
}

/// Ids of derived (AND) outcomes start here, far above any real measurement id.
const DERIVED_BASE: usize = 1 << 40;

#[derive(Debug, Clone)]
enum TransItem {
    Pauli(TransPauli),
    Clifford(TransClifford),
    Event(Event),
}

impl fmt::Display for TransItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransItem::Pauli(p) => write!(f, "{}", p),
            TransItem::Clifford(c) => write!(f, "{}", c),
            TransItem::Event(_) => Ok(()),
        }
    }
}

/// Classical registers declared so far: name -> (global index of bit 0, size).
type Cregs = HashMap<String, (usize, usize)>;

fn invalid(line_no: usize, msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("line {}: {}", line_no, msg))
}

/// `creg name[size];` -> declares `size` global classical bits.
fn parse_creg(line: &str, cregs: &mut Cregs, n_clbits: &mut usize) {
    let body = line["creg".len()..].trim().trim_end_matches(';');
    if let (Some(open), Some(close)) = (body.find('['), body.find(']')) {
        if let Ok(size) = body[open + 1..close].parse::<usize>() {
            cregs.insert(body[..open].trim().to_string(), (*n_clbits, size));
            *n_clbits += size;
        }
    }
}

/// Resolve `name[idx]` (or a bare `name` for a 1-bit register) to a global classical-bit index.
fn resolve_clbit(text: &str, cregs: &Cregs, line_no: usize) -> io::Result<usize> {
    let text = text.trim().trim_end_matches(';').trim();
    let (name, idx) = match (text.find('['), text.find(']')) {
        (Some(open), Some(close)) => {
            let idx = text[open + 1..close]
                .trim()
                .parse::<usize>()
                .map_err(|_| invalid(line_no, format!("bad classical bit index in `{}`", text)))?;
            (text[..open].trim(), Some(idx))
        }
        _ => (text, None),
    };
    let &(base, size) = cregs
        .get(name)
        .ok_or_else(|| invalid(line_no, format!("unknown classical register `{}`", name)))?;
    match idx {
        Some(i) if i < size => Ok(base + i),
        Some(i) => Err(invalid(line_no, format!("bit {} out of range for `{}[{}]`", i, name, size))),
        None if size == 1 => Ok(base),
        None => Err(invalid(
            line_no,
            format!("`{}` has {} bits; index it explicitly (e.g. {}[0])", name, size, name),
        )),
    }
}

/// `if(c==1) <gate> q[i];` -> a conditional gate.
///
/// The condition is one or more `reg==0|1` terms joined by `&&`, each on a 1-bit register (so a
/// block that measures several ancillas into one register each can be conditioned on a
/// combination). A Pauli under a single term is a classical frame update ([`QasmGate::CondPauli`]);
/// every other single-qubit gate becomes a [`QasmGate::CondGate`]. Anything else conditioned on a
/// classical bit is rejected rather than silently dropped, because dropping it would change the
/// circuit.
fn parse_conditional(line: &str, cregs: &Cregs, line_no: usize) -> io::Result<QasmGate> {
    let unsupported = |why: &str| {
        invalid(
            line_no,
            format!(
                "unsupported conditional `{}`: {}. Supported: `if(r==0|1 [&& r2==0|1 ...]) g q[i];` \
                 with 1-bit registers and a single-qubit gate g in h,s,sdg,sx,sxdg,x,y,z,t,tdg",
                line, why
            ),
        )
    };
    let open = line.find('(').ok_or_else(|| unsupported("missing `(`"))?;
    let close = line.find(')').ok_or_else(|| unsupported("missing `)`"))?;
    let mut terms: Vec<(usize, bool)> = Vec::new();
    for term in line[open + 1..close].split("&&") {
        let (reg, val) =
            term.split_once("==").ok_or_else(|| unsupported("expected `creg==value`"))?;
        let value = match val.trim() {
            "0" => false,
            "1" => true,
            other => return Err(unsupported(&format!("condition value `{}` is not 0 or 1", other))),
        };
        let reg_name = reg.trim();
        let &(_, size) = cregs
            .get(reg_name)
            .ok_or_else(|| invalid(line_no, format!("unknown classical register `{}`", reg_name)))?;
        if size != 1 {
            return Err(unsupported("the register has more than one bit"));
        }
        terms.push((resolve_clbit(reg_name, cregs, line_no)?, value));
    }
    let rest = line[close + 1..].trim();
    let inner = try_parse_1q(rest).ok_or_else(|| unsupported("the conditioned gate is not a supported single-qubit gate"))?;
    match (&inner, terms.as_slice()) {
        (QasmGate::Clifford1Q { gate, qubit }, &[(clbit, value)])
            if matches!(gate, Gate1Q::X | Gate1Q::Y | Gate1Q::Z) =>
        {
            Ok(QasmGate::CondPauli { gate: *gate, qubit: *qubit, clbit, value })
        }
        _ => Ok(QasmGate::CondGate { inner: Box::new(inner), terms }),
    }
}

/// Settles conditions that are known without running the circuit.
///
/// A classical bit no measurement has written yet reads as 0, so a term demanding 1 from it can
/// never hold (the gate is dropped) and a term demanding 0 always holds (the term is dropped); a
/// gate left with no terms is unconditional and becomes the plain gate. Two terms that demand
/// different values of one bit also drop the gate. After this, every remaining term refers to a
/// bit that an earlier measurement wrote, so the transpiler can name the outcome it reads.
fn resolve_static_conditions(gates: &[QasmGate]) -> Vec<QasmGate> {
    let mut written: HashSet<usize> = HashSet::new();
    let mut out = Vec::with_capacity(gates.len());
    for g in gates {
        match g {
            QasmGate::Measure { clbit: Some(c), .. } => {
                written.insert(*c);
                out.push(g.clone());
            }
            QasmGate::CondGate { inner, terms } => {
                let mut keep: Vec<(usize, bool)> = Vec::new();
                let mut never = false;
                for &(cb, v) in terms {
                    if !written.contains(&cb) {
                        never |= v;
                    } else if let Some(&(_, prev)) = keep.iter().find(|(k, _)| *k == cb) {
                        never |= prev != v;
                    } else {
                        keep.push((cb, v));
                    }
                }
                if never {
                    continue;
                }
                if keep.is_empty() {
                    out.push((**inner).clone());
                } else {
                    out.push(QasmGate::CondGate { inner: inner.clone(), terms: keep });
                }
            }
            _ => out.push(g.clone()),
        }
    }
    out
}

/// Parse a `.cliffordt.qasm` file into a gate list and qubit count.
///
/// Unrecognised lines (headers, custom gate defs, etc.) are silently skipped, with one exception:
/// a statement starting with `if` must be a supported conditional gate (see
/// [`parse_conditional`]), otherwise parsing fails.
fn parse_qasm(path: &str) -> io::Result<(usize, Vec<QasmGate>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut n_qubits = 0usize;
    let mut gates = Vec::new();
    let mut cregs = Cregs::new();
    let mut n_clbits = 0usize;

    for (line_idx, line) in reader.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = line?;
        let line = line.trim();

        if line.starts_with("creg") {
            parse_creg(line, &mut cregs, &mut n_clbits);
            continue;
        }

        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with("OPENQASM")
            || line.starts_with("include")
            || line.starts_with("gate ")
            || line.starts_with('{')
            || line.starts_with('}')
            || line.starts_with("U(")
        {
            continue;
        }

        if line.starts_with("barrier") {
            gates.push(QasmGate::Barrier);
            continue;
        }

        if line.starts_with("qreg") {
            if let Some(n) = parse_qreg(line) {
                n_qubits = n;
            }
            continue;
        }

        if line.starts_with("measure") {
            // `measure q[i];` or `measure q[i] -> c[j];`
            let (lhs, rhs) = match line.split_once("->") {
                Some((l, r)) => (l, Some(r)),
                None => (line, None),
            };
            if let Some(q) = parse_single_qubit_index(lhs) {
                let clbit = match rhs {
                    Some(r) => Some(resolve_clbit(r, &cregs, line_no)?),
                    None => None,
                };
                gates.push(QasmGate::Measure { qubit: q, clbit });
            }
            continue;
        }

        if line.starts_with("reset") {
            if let Some(q) = parse_single_qubit_index(line) {
                gates.push(QasmGate::Reset { qubit: q });
            }
            continue;
        }

        if line.starts_with("if") && line[2..].trim_start().starts_with('(') {
            gates.push(parse_conditional(line, &cregs, line_no)?);
            continue;
        }

        if let Some(g) = try_parse_2q(line) {
            gates.push(g);
            continue;
        }
        if let Some(g) = try_parse_1q(line) {
            gates.push(g);
            continue;
        }
    }

    let gates = reorder_by_cycles(gates, n_qubits, n_clbits);
    Ok((n_qubits, gates))
}

/// Reorder gates into cycle-based order: pack each gate into the earliest
/// cycle where all its qubits (and its classical bit, if any) are free, then sort within each
/// cycle by minimum qubit index.  This is a semantics-preserving reordering of independent
/// (commuting) gates. A classical bit is a resource like a qubit, so a conditional gate can
/// never move before the measurement that writes its bit, nor a re-measurement before a read.
fn reorder_by_cycles(gates: Vec<QasmGate>, n_qubits: usize, n_clbits: usize) -> Vec<QasmGate> {
    if n_qubits == 0 {
        return gates;
    }

    let mut next_cycle = vec![0usize; n_qubits];
    let mut next_cycle_c = vec![0usize; n_clbits];
    let mut cycles: Vec<Vec<(usize, QasmGate)>> = Vec::new();

    for gate in gates {
        if matches!(gate, QasmGate::Barrier) {
            // Advance all qubits to the same cycle, place barrier, then advance past it.
            let sync = next_cycle.iter().chain(next_cycle_c.iter()).copied().max().unwrap_or(0);
            if sync >= cycles.len() {
                cycles.resize_with(sync + 1, Vec::new);
            }
            cycles[sync].push((usize::MAX, gate));
            let after = sync + 1;
            for q in next_cycle.iter_mut().chain(next_cycle_c.iter_mut()) {
                *q = after;
            }
            continue;
        }

        let qubits = gate.qubits();
        let clbits = gate.clbits();
        let cycle = qubits
            .iter()
            .map(|&q| next_cycle[q])
            .chain(clbits.iter().map(|&c| next_cycle_c[c]))
            .max()
            .unwrap_or(0);
        if cycle >= cycles.len() {
            cycles.resize_with(cycle + 1, Vec::new);
        }
        let min_q = *qubits.iter().min().unwrap();
        cycles[cycle].push((min_q, gate));
        for &q in &qubits {
            next_cycle[q] = cycle + 1;
        }
        for &c in &clbits {
            next_cycle_c[c] = cycle + 1;
        }
    }

    let mut result = Vec::new();
    for cycle in cycles {
        let mut sorted = cycle;
        sorted.sort_by_key(|(min_q, _)| *min_q);
        for (_, gate) in sorted {
            if !matches!(gate, QasmGate::Barrier) {
                result.push(gate);
            }
        }
    }
    result
}

fn parse_qreg(line: &str) -> Option<usize> {
    let start = line.find('[')? + 1;
    let end = line.find(']')?;
    line[start..end].parse().ok()
}

fn parse_single_qubit_index(line: &str) -> Option<usize> {
    let start = line.find('[')? + 1;
    let end = line.find(']')?;
    line[start..end].parse().ok()
}

fn parse_two_qubit_indices(line: &str) -> Option<(usize, usize)> {
    let mut indices = line.split('[').skip(1);
    let a: usize = indices.next()?.split(']').next()?.parse().ok()?;
    let b: usize = indices.next()?.split(']').next()?.parse().ok()?;
    Some((a, b))
}

fn try_parse_2q(line: &str) -> Option<QasmGate> {
    let lower = line.to_lowercase();
    if lower.starts_with("cx ") || lower.starts_with("cx\t") {
        let (a, b) = parse_two_qubit_indices(line)?;
        return Some(QasmGate::Clifford2Q { gate: Gate2Q::CX, control: a, target: b });
    }
    if lower.starts_with("cz ") || lower.starts_with("cz\t") {
        let (a, b) = parse_two_qubit_indices(line)?;
        return Some(QasmGate::Clifford2Q { gate: Gate2Q::CZ, control: a, target: b });
    }
    if lower.starts_with("swap ") || lower.starts_with("swap\t") {
        let (a, b) = parse_two_qubit_indices(line)?;
        return Some(QasmGate::Clifford2Q { gate: Gate2Q::Swap, control: a, target: b });
    }
    None
}

fn try_parse_1q(line: &str) -> Option<QasmGate> {
    let mut parts = line.splitn(2, |c: char| c.is_whitespace());
    let name = parts.next()?.trim().to_lowercase();
    let name = name.trim_end_matches(';');
    let q = parse_single_qubit_index(line)?;
    let gate = match name {
        "h" => QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: q },
        "s" => QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: q },
        "sdg" => QasmGate::Clifford1Q { gate: Gate1Q::Sdg, qubit: q },
        "sx" => QasmGate::Clifford1Q { gate: Gate1Q::SX, qubit: q },
        "sxdg" => QasmGate::Clifford1Q { gate: Gate1Q::SXdg, qubit: q },
        "x" => QasmGate::Clifford1Q { gate: Gate1Q::X, qubit: q },
        "y" => QasmGate::Clifford1Q { gate: Gate1Q::Y, qubit: q },
        "z" => QasmGate::Clifford1Q { gate: Gate1Q::Z, qubit: q },
        "t" => QasmGate::T { qubit: q },
        "tdg" => QasmGate::Tdg { qubit: q },
        _ => return None,
    };
    Some(gate)
}

// Each of the 24 single-qubit Cliffords is uniquely identified by the pair
// (X-image, Z-image) of its conjugation action on the Pauli group.

/// Encode a signed single-qubit Pauli as a u8.
/// Pauli type: 0=X, 1=Y, 2=Z.  Sign bit 2: 0=+, 1=−.
#[inline]
fn pauli_code(pauli: u8, neg: bool) -> u8 {
    pauli | ((neg as u8) << 2)
}

/// Apply a single-qubit gate to a Clifford state `(x_img, z_img)` and return
/// the updated state.  Each image is a [`pauli_code`]-encoded signed Pauli
/// representing where the gate sends X (resp. Z) under conjugation.
fn apply_1q_to_state(gate: Gate1Q, x_img: u8, z_img: u8) -> (u8, u8) {
    // Conjugation rules (P → G·P·G†); composing states applies each gate to
    // x_img and z_img independently.
    fn apply_gate_to_pauli(gate: Gate1Q, p: u8) -> u8 {
        let pauli = p & 0x3;
        let neg = (p >> 2) & 1 != 0;
        let (new_pauli, extra_neg) = match gate {
            Gate1Q::H => match pauli {
                0 => (2, false), // X→Z
                1 => (1, true),  // Y→-Y
                2 => (0, false), // Z→X
                _ => unreachable!(),
            },
            Gate1Q::S => match pauli {
                0 => (1, false), // X→Y
                1 => (0, true),  // Y→-X
                2 => (2, false), // Z→Z
                _ => unreachable!(),
            },
            Gate1Q::Sdg => match pauli {
                0 => (1, true),  // X→-Y
                1 => (0, false), // Y→X
                2 => (2, false), // Z→Z
                _ => unreachable!(),
            },
            Gate1Q::SX => match pauli {
                0 => (0, false), // X→X
                1 => (2, false), // Y→Z   (Y = iXZ → i·X·(-Y) = Z)
                2 => (1, true),  // Z→-Y
                _ => unreachable!(),
            },
            Gate1Q::SXdg => match pauli {
                0 => (0, false), // X→X
                1 => (2, true),  // Y→-Z
                2 => (1, false), // Z→Y
                _ => unreachable!(),
            },
            Gate1Q::X => match pauli {
                0 => (0, false), // X→X
                1 => (1, true),  // Y→-Y
                2 => (2, true),  // Z→-Z
                _ => unreachable!(),
            },
            Gate1Q::Y => match pauli {
                0 => (0, true),  // X→-X
                1 => (1, false), // Y→Y
                2 => (2, true),  // Z→-Z
                _ => unreachable!(),
            },
            Gate1Q::Z => match pauli {
                0 => (0, true),  // X→-X
                1 => (1, true),  // Y→-Y
                2 => (2, false), // Z→Z
                _ => unreachable!(),
            },
        };
        pauli_code(new_pauli, neg ^ extra_neg)
    }
    (apply_gate_to_pauli(gate, x_img), apply_gate_to_pauli(gate, z_img))
}

/// Compose a sequence of single-qubit gates into a single Clifford state
/// `(x_img, z_img)` by applying each gate in order to the identity state
/// `(X+, Z+)`.
fn simulate_gate_sequence(gates: &[Gate1Q]) -> (u8, u8) {
    let mut x_img = pauli_code(0, false); // X+
    let mut z_img = pauli_code(2, false); // Z+
    for &g in gates {
        let (nx, nz) = apply_1q_to_state(g, x_img, z_img);
        x_img = nx;
        z_img = nz;
    }
    (x_img, z_img)
}

/// Build a lookup table mapping each of the 24 single-qubit Clifford states
/// `(x_img, z_img)` to a minimal gate sequence (at most 4 gates, from
/// {I, S, Sdg, SX, SXdg, X, Z}) that produces it; no two sequences collide.
fn build_clifford_table() -> std::collections::HashMap<(u8, u8), Vec<Gate1Q>> {
    use Gate1Q::*;
    let sequences: &[&[Gate1Q]] = &[
        &[],             // I
        &[X],            // X
        &[X, Z],         // X·Z
        &[Z],            // Z
        &[S, SX, S],     // S·SX·S  (= H up to global phase)
        &[S],            // S
        &[Sdg],          // Sdg
        &[S, X],         // S·X
        &[Sdg, X],       // Sdg·X
        &[Z, SX, S],     // Z·SX·S
        &[SX, S],        // SX·S
        &[S, SX, Z],     // S·SX·Z
        &[Z, SX, Z],     // Z·SX·Z
        &[Z, SX, Z, X],  // Z·SX·Z·X
        &[Z, SXdg],      // Z·SXdg
        &[Z, SX],        // Z·SX
        &[Sdg, SX, S],   // Sdg·SX·S
        &[Sdg, SX, Sdg], // Sdg·SX·Sdg
        &[S, SX, Sdg],   // S·SX·Sdg
        &[S, SXdg, Z],   // S·SXdg·Z
        &[S, SXdg],      // S·SXdg
        &[S, SX],        // S·SX
        &[SX, Sdg],      // SX·Sdg
        &[Z, SX, Sdg],   // Z·SX·Sdg
    ];
    let mut table = std::collections::HashMap::new();
    for seq in sequences {
        let state = simulate_gate_sequence(seq);
        table.insert(state, seq.to_vec());
    }
    table
}

/// Reduce a sequence of single-qubit Clifford gates on one qubit to the
/// minimal equivalent sequence.  Returns an empty vec for the identity.
///
/// The lookup table is built once per thread and cached for subsequent calls.
fn optimize_single_qubit_sequence(gates: &[Gate1Q]) -> Vec<Gate1Q> {
    if gates.is_empty() {
        return vec![];
    }
    let state = simulate_gate_sequence(gates);
    use std::cell::RefCell;
    thread_local! {
        static TABLE: RefCell<Option<std::collections::HashMap<(u8, u8), Vec<Gate1Q>>>> =
            RefCell::new(None);
    }
    TABLE.with(|t| {
        let mut borrow = t.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(build_clifford_table());
        }
        borrow.as_ref().unwrap().get(&state).cloned().unwrap_or_default()
    })
}

/// Reduce a sequence of Clifford gates to a minimal equivalent sequence and
/// return the result as [`TransClifford`] items ready for the .trans output.
///
/// Single-qubit gates per qubit are compressed via the 24-element lookup;
/// a 2-qubit gate flushes and optimizes the pending sequences on its two
/// qubits first. X and Z are Pauli corrections omitted from the output.
/// Splits an accumulated Clifford queue into gates to emit and a trailing single-qubit layer.
///
/// Per-qubit runs of 1Q gates are collapsed by `optimize_single_qubit_sequence` and flushed
/// whenever a 2Q gate touches that qubit. Whatever remains at the end is each qubit's run *after*
/// its last 2Q gate, so the residual is a tensor product of single-qubit Cliffords acting last on
/// their own wires — `U = R · E` with `R` local, needing no commutation or resynthesis.
///
/// Keeping `R` pending is sound for width: a local Clifford maps `Z_q` to a single-qubit Pauli on
/// `q`, so products stay weight 1, merely rotated. Only 2Q gates ever spread support.
fn split_trailing_1q(gates: &[QasmGate], n_qubits: usize) -> (Vec<QasmGate>, Vec<QasmGate>) {
    let mut per_qubit: Vec<Vec<Gate1Q>> = vec![Vec::new(); n_qubits];
    let mut emitted: Vec<QasmGate> = Vec::new();

    fn flush_qubit(qubit: usize, per_qubit: &mut [Vec<Gate1Q>], out: &mut Vec<QasmGate>) {
        let seq = std::mem::take(&mut per_qubit[qubit]);
        if seq.is_empty() {
            return;
        }
        for gate in optimize_single_qubit_sequence(&seq) {
            out.push(QasmGate::Clifford1Q { gate, qubit });
        }
    }

    for gate in gates {
        match gate {
            QasmGate::Clifford1Q { gate: g, qubit } => per_qubit[*qubit].push(*g),
            QasmGate::Clifford2Q { control, target, .. } => {
                flush_qubit(*control, &mut per_qubit, &mut emitted);
                flush_qubit(*target, &mut per_qubit, &mut emitted);
                emitted.push(gate.clone());
            }
            _ => {}
        }
    }

    let mut residual: Vec<QasmGate> = Vec::new();
    for qubit in 0..n_qubits {
        flush_qubit(qubit, &mut per_qubit, &mut residual);
    }
    (emitted, residual)
}

/// The inverse of a single-qubit Clifford (S and SX are the only non-Hermitian ones).
fn inverse_1q(g: Gate1Q) -> Gate1Q {
    match g {
        Gate1Q::S => Gate1Q::Sdg,
        Gate1Q::Sdg => Gate1Q::S,
        Gate1Q::SX => Gate1Q::SXdg,
        Gate1Q::SXdg => Gate1Q::SX,
        other => other, // H, X, Y, Z are their own inverses
    }
}

/// Adds the next gate of a pending Clifford sequence to its tableau.
///
/// A rotation or measurement placed after Cliffords `C` is written `C^dagger P C` (the
/// Heisenberg pull-back), so that it can run before `C`. `Tableau::conjugate` computes
/// `U P U^dagger`, and its `prepend` composes gates in reverse, so pulling back through `C` means
/// feeding it the *inverse* gates. For H, CX and the Paulis that is a no-op; for S and SX it
/// matters, and omitting it flips the sign of any product that passes through one (the unsigned
/// Pauli, which is all the scheduler reads, is unaffected).
fn push_clifford(tableau: &mut Tableau, gate: &QasmGate) {
    match gate {
        QasmGate::Clifford1Q { gate: g, qubit } => tableau.prepend_1q_correct(inverse_1q(*g), *qubit),
        QasmGate::Clifford2Q { gate: g, control, target } => {
            tableau.prepend_2q(*g, *control, *target)
        }
        _ => {}
    }
}

/// Rebuilds the tableau for a pending Clifford sequence.
fn tableau_of(gates: &[QasmGate], n_qubits: usize) -> Tableau {
    let mut tableau = Tableau::new(n_qubits);
    for gate in gates {
        push_clifford(&mut tableau, gate);
    }
    tableau
}

/// Inverse of [`TransClifford::from_qasm_gate`], for the gates the `.trans` format can express.
fn trans_to_qasm(c: &TransClifford) -> QasmGate {
    let at = |ch: char| c.ops.iter().position(|&o| o == ch);
    match c.name.as_str() {
        "CX" => QasmGate::Clifford2Q {
            gate: Gate2Q::CX,
            control: at('Z').expect("CX line has a Z"),
            target: at('X').expect("CX line has an X"),
        },
        name => {
            let gate = match name {
                "S" => Gate1Q::S,
                "Sdg" => Gate1Q::Sdg,
                "SX" => Gate1Q::SX,
                "SXdg" => Gate1Q::SXdg,
                other => unreachable!("unknown .trans Clifford {}", other),
            };
            let qubit = c.ops.iter().position(|&o| o != '_').expect("1Q line has an operator");
            QasmGate::Clifford1Q { gate, qubit }
        }
    }
}

/// The Pauli `G` by which the emitted `.trans` Cliffords differ from the true gates.
///
/// X, Y and Z are never emitted (they are Pauli-frame corrections), and the single-qubit
/// lookup realises some Cliffords only with such gates, so the physical unitary is
/// `U_emit = G^-1 U_true` for a Pauli `G`. Both tableaus agree on every unsigned image
/// (`same_up_to_sign`); they differ only in signs. With `conjugate(P) = U^dagger P U`, `G`
/// anticommutes with a basis Pauli exactly when the two signs differ, which fixes `G` up to
/// its own (irrelevant) sign. The physical state is then `G` times the ideal state: a constant
/// frame term, owed from this point on.
fn dropped_pauli(true_gates: &[QasmGate], n_qubits: usize) -> PauliString {
    let emitted: Vec<QasmGate> =
        to_trans_cliffords(true_gates, n_qubits).iter().map(trans_to_qasm).collect();
    let (t_true, t_emit) = (tableau_of(true_gates, n_qubits), tableau_of(&emitted, n_qubits));
    let mut g = PauliString::identity(n_qubits);
    for q in 0..n_qubits {
        let mut px = PauliString::identity(n_qubits);
        px.x_bits[q] = true;
        if t_true.conjugate(&px).sign != t_emit.conjugate(&px).sign {
            g.z_bits[q] = true; // anticommutes with X_q  <=>  has a Z component
        }
        let mut pz = PauliString::identity(n_qubits);
        pz.z_bits[q] = true;
        if t_true.conjugate(&pz).sign != t_emit.conjugate(&pz).sign {
            g.x_bits[q] = true; // anticommutes with Z_q  <=>  has an X component
        }
    }
    g
}

fn to_trans_cliffords(gates: &[QasmGate], n_qubits: usize) -> Vec<TransClifford> {
    gates.iter().filter_map(|g| TransClifford::from_qasm_gate(g, n_qubits)).collect()
}

#[cfg(test)]
fn optimize_clifford_sequence(gates: &[QasmGate], n_qubits: usize) -> Vec<TransClifford> {
    let (emitted, residual) = split_trailing_1q(gates, n_qubits);
    let mut all = emitted;
    all.extend(residual);
    to_trans_cliffords(&all, n_qubits)
}

/// Layer occupancy of each emitted product, mirroring the expansion in
/// `Circuit::estimate_num_layers`: a CX spans 2 logical cycles, an S/SX spans 3, and a
/// T or measurement spans 1. Set all four to 1 for an unweighted DAG-depth model.
const LAYER_COST_T: usize = 1;
const LAYER_COST_M: usize = 1;
const LAYER_COST_CX: usize = 2;
const LAYER_COST_S: usize = 3;

/// Predicts the layer each emitted product will land in, mirroring `Circuit::compute_layers`.
///
/// Downstream, `Circuit::gen_deps` makes a product's parents the last product on each of its
/// qubits and `compute_layers` runs Kahn's algorithm over that DAG, so a product's layer is
/// `1 + max(layer of parents)`. Tracking the layer of the last product on each qubit reduces
/// that to an O(|support|) update, letting the transpiler predict layering without the
/// scheduler.
#[derive(Debug, Clone)]
struct LayerModel {
    /// Last layer occupied by a product touching this qubit; 0 = untouched.
    depth: Vec<usize>,
}

impl LayerModel {
    fn new(n_qubits: usize) -> Self {
        LayerModel { depth: vec![0; n_qubits] }
    }

    /// The layer a product on `qubits` would land in, without placing it.
    fn peek(&self, qubits: &[usize]) -> usize {
        qubits.iter().map(|&q| self.depth[q]).max().unwrap_or(0) + 1
    }

    /// Places a product spanning `cost` layers and returns the layer it starts in.
    fn place(&mut self, qubits: &[usize], cost: usize) -> usize {
        let layer = self.peek(qubits);
        for &q in qubits {
            self.depth[q] = layer + cost - 1;
        }
        layer
    }

    /// Deepest layer reached on any qubit so far.
    fn max_depth(&self) -> usize {
        self.depth.iter().copied().max().unwrap_or(0)
    }
}

/// Qubits acted on by a Pauli product ('I' entries are identity).
fn pauli_support(p: &TransPauli) -> Vec<usize> {
    (0..p.ops.len()).filter(|&q| p.ops[q] != 'I').collect()
}

/// Qubits acted on by a Clifford, with its layer occupancy.
fn clifford_support(c: &TransClifford) -> (Vec<usize>, usize) {
    let qubits: Vec<usize> = (0..c.ops.len()).filter(|&q| c.ops[q] != '_').collect();
    let cost = if c.name == "CX" { LAYER_COST_CX } else { LAYER_COST_S };
    (qubits, cost)
}

/// Applies the Cliffords a flush would emit to `layers`, as `flush_clifford_queue` would,
/// returning the qubits they touch.
#[cfg(test)]
fn apply_flushed_cliffords(
    layers: &mut LayerModel, queue: &[QasmGate], n_qubits: usize,
) -> Vec<usize> {
    let mut touched = Vec::new();
    for tc in optimize_clifford_sequence(queue, n_qubits) {
        let (qubits, cost) = clifford_support(&tc);
        layers.place(&qubits, cost);
        touched.extend(qubits);
    }
    touched
}

#[cfg(test)]
fn transpile(n_qubits: usize, gates: &[QasmGate], max_weight: i32) -> Vec<TransItem> {
    transpile_full(n_qubits, gates, max_weight, false)
}

/// Transpiles at a fixed max Pauli-product weight.
///
/// `defer_trailing` holds back each qubit's trailing single-qubit Clifford run at every flush.
/// It is orthogonal to the weight threshold and reduces predicted depth by up to 32%, since
/// S/SX gates span 3 logical cycles apiece and buy no width.
#[cfg(test)]
fn transpile_full(
    n_qubits: usize, gates: &[QasmGate], max_weight: i32, defer_trailing: bool,
) -> Vec<TransItem> {
    transpile_opts(n_qubits, gates, max_weight, defer_trailing, ResetMode::Frame)
}

/// Pays off the Pauli the frame owes physical qubit `q`, returning it as an [`Event::Fix`].
///
/// After this, `q` carries no frame, which is what lets a *conditional* gate on `q` be a plain
/// physical operation: a conditional Clifford would otherwise conjugate the owed Pauli only when
/// its condition holds, and that makes the frame depend on outcomes through an AND.
fn pay_frame(frame: &mut PauliFrame, q: usize) -> Option<Event> {
    let mut x: Vec<usize> = std::mem::take(&mut frame.dx[q]).into_iter().collect();
    let mut z: Vec<usize> = std::mem::take(&mut frame.dz[q]).into_iter().collect();
    let x_const = take_const(&mut x);
    let z_const = take_const(&mut z);
    if x.is_empty() && z.is_empty() && !x_const && !z_const {
        return None;
    }
    Some(Event::Fix { qubit: q, x_deps: x, x_const, z_deps: z, z_const })
}

/// The outcome id that stands for "every term of `cond` holds".
///
/// A single positive term is just that measurement's outcome. Otherwise a fresh derived
/// outcome is defined (an [`Event::Derived`]), so a Pauli under this condition remains a linear
/// frame term: `P^(AND of terms)` is `P` owed whenever the derived bit is 1.
fn derived_outcome(
    cond: &[(usize, bool)], next: &mut usize, defs: &mut HashMap<usize, Vec<(usize, bool)>>,
    ops: &mut Vec<TransItem>,
) -> usize {
    if let [(m, true)] = cond {
        return *m;
    }
    let id = DERIVED_BASE + *next;
    *next += 1;
    defs.insert(id, cond.to_vec());
    ops.push(TransItem::Event(Event::Derived { id, terms: cond.to_vec() }));
    id
}

/// The real measurement ids behind `ids`: derived outcomes expand to their terms' measurements,
/// and the constant pseudo-outcome is dropped.
fn real_outcomes(ids: &[usize], defs: &HashMap<usize, Vec<(usize, bool)>>) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for &id in ids {
        if id == CONST_OUTCOME {
            continue;
        }
        match defs.get(&id) {
            Some(terms) => out.extend(terms.iter().map(|&(m, _)| m)),
            None => {
                out.insert(id);
            }
        }
    }
    out
}

/// [`transpile_full`] with an explicit [`ResetMode`] for mid-circuit `reset`s.
///
/// Mid-circuit measurement needs no special treatment in the Heisenberg picture used here: a
/// measurement of `Z_q` is the product `C^dagger Z_q C` emitted in program order, exactly like a
/// final measurement. What dynamic circuits add is classical bookkeeping:
///   * every `M` product gets an outcome id (`PauliMeta::meas_id`);
///   * a `reset` is that measurement plus the correction `X_q^outcome`, which is owed to the
///     physical state and so enters the classical [`PauliFrame`] instead of the schedule;
///   * a conditional Pauli (`if(c==v) x q[i];`) is the same kind of frame update;
///   * each later `T`/`M` product records which outcomes flip its sign (`frame_deps`);
///   * a Pauli conditioned on several bits is still a frame term, on a *derived* AND outcome;
///   * any other conditional gate (`if(c==1) h|s|sx|t ...`) is physical: pending Cliffords are
///     flushed, the qubit's owed Pauli is paid ([`Event::Fix`]), and the gate is emitted as
///     ordinary products tagged with their condition and the measurements they wait for.
fn transpile_opts(
    n_qubits: usize, gates: &[QasmGate], max_weight: i32, defer_trailing: bool,
    reset_mode: ResetMode,
) -> Vec<TransItem> {
    let effective_max_weight = if max_weight <= 0 { n_qubits + 1 } else { max_weight as usize };

    let resolved = resolve_static_conditions(gates);
    let gates = &resolved[..];

    let mut tableau = Tableau::new(n_qubits);
    let mut ops: Vec<TransItem> = Vec::new();
    let mut clifford_queue: Vec<QasmGate> = Vec::new();

    let has_t = gates.iter().any(|g| matches!(g, QasmGate::T { .. } | QasmGate::Tdg { .. }));
    if !has_t {
        eprintln!("Warning: circuit has no T gates");
    }

    // Append a final Z-basis measurement to every qubit whose last operation is not already a
    // measurement or a reset. (A qubit measured mid-circuit and then reused still needs one; a
    // qubit left in |0> by a trailing reset does not.)
    let mut gates_with_measurements = gates.to_vec();
    let mut ends_measured = vec![false; n_qubits];
    for g in gates {
        match g {
            QasmGate::Measure { qubit, .. } | QasmGate::Reset { qubit } => {
                ends_measured[*qubit] = true
            }
            // A conditional Pauli is a frame update, not a physical gate: it neither needs a final
            // measurement after it nor cancels one that already happened.
            QasmGate::CondPauli { .. } => {}
            // ... and neither is a Pauli conditioned on several bits.
            QasmGate::CondGate { inner, .. }
                if matches!(
                    inner.as_ref(),
                    QasmGate::Clifford1Q { gate: Gate1Q::X | Gate1Q::Y | Gate1Q::Z, .. }
                ) => {}
            other => {
                for q in other.qubits() {
                    ends_measured[q] = false;
                }
            }
        }
    }
    for (q, &done) in ends_measured.iter().enumerate() {
        if !done {
            gates_with_measurements.push(QasmGate::Measure { qubit: q, clbit: None });
        }
    }

    let mut frame = PauliFrame::new(n_qubits);
    let mut next_meas_id = 0usize;
    // Latest outcome stored in each classical bit.
    let mut clbit_meas: HashMap<usize, usize> = HashMap::new();
    // Derived (AND) outcomes: next index, and each one's terms.
    let mut next_derived = 0usize;
    let mut derived_defs: HashMap<usize, Vec<(usize, bool)>> = HashMap::new();

    let mut last_weight: Option<usize> = None;
    let mut layers = LayerModel::new(n_qubits);

    /// Emits the queue and resets the tableau to whatever stays pending.
    ///
    /// With `defer_trailing`, each qubit's trailing single-qubit run is held back rather than
    /// emitted. That layer is local, so deferring it keeps every later product weight 1 while
    /// avoiding S/SX gates that cost 3 logical cycles each and buy no width. The held-back gates
    /// stay queued and merge with the next segment's leading 1Q gates, where
    /// `optimize_single_qubit_sequence` can collapse the combined run.
    fn flush_clifford_queue(
        clifford_queue: &mut Vec<QasmGate>, ops: &mut Vec<TransItem>, tableau: &mut Tableau,
        layers: &mut LayerModel, frame: &mut PauliFrame, n_qubits: usize, defer_trailing: bool,
    ) {
        let (emitted, residual) = split_trailing_1q(clifford_queue, n_qubits);
        let mut to_emit = emitted;
        if !defer_trailing {
            to_emit.extend(residual.iter().cloned());
        }
        // The emitted gates become physical, which changes the coordinates products are written
        // in, so the classical frame (kept in those coordinates) is pushed through them.
        for g in &to_emit {
            frame.conj_gate(g);
        }
        // ...and the Pauli the emitted gates are missing joins it as a constant term.
        frame.add(&dropped_pauli(&to_emit, n_qubits), CONST_OUTCOME);
        for c in to_trans_cliffords(&to_emit, n_qubits) {
            let (qubits, cost) = clifford_support(&c);
            layers.place(&qubits, cost);
            ops.push(TransItem::Clifford(c));
        }
        *clifford_queue = if defer_trailing { residual } else { Vec::new() };
        *tableau = tableau_of(clifford_queue, n_qubits);
    }

    for gate in &gates_with_measurements {
        match gate {
            QasmGate::Clifford1Q { .. } | QasmGate::Clifford2Q { .. } => {
                clifford_queue.push(gate.clone());
                push_clifford(&mut tableau, gate);
            }
            QasmGate::T { qubit } | QasmGate::Tdg { qubit } => {
                // Tdg uses a negative-sign Z Pauli (−Z rotation).
                let negative = matches!(gate, QasmGate::Tdg { .. });
                let pre_pauli = make_z_pauli(n_qubits, *qubit, negative);
                let conjugated = tableau.conjugate(&pre_pauli);
                let weight = conjugated.weight();
                last_weight = Some(weight);
                let flush = weight > effective_max_weight;
                let ps = if flush {
                    flush_clifford_queue(
                        &mut clifford_queue,
                        &mut ops,
                        &mut tableau,
                        &mut layers,
                        &mut frame,
                        n_qubits,
                        defer_trailing,
                    );
                    tableau.conjugate(&make_z_pauli(n_qubits, *qubit, negative))
                } else {
                    conjugated
                };
                let mut emitted = TransPauli::from_pauli_string(&ps, "T");
                emitted.apply_frame(frame.deps(&ps));
                layers.place(&pauli_support(&emitted), LAYER_COST_T);
                ops.push(TransItem::Pauli(emitted));
            }
            QasmGate::Measure { qubit, .. } | QasmGate::Reset { qubit } => {
                let is_reset = matches!(gate, QasmGate::Reset { .. });
                if last_weight.is_none() && !is_reset {
                    eprintln!("Warning: measurement on qubit {} before any T gate", qubit);
                }
                let pre_pauli = make_z_pauli(n_qubits, *qubit, false);
                let conjugated = tableau.conjugate(&pre_pauli);
                // Static policy flushes only when the preceding T gate's conjugated weight
                // exceeded max_weight; the measurement's own conjugated weight is not checked.
                // The dynamic policy never flushes here: there is no final flush after the loop
                // either, and that is sound because every qubit gets an appended measurement
                // whose Pauli is conjugated through the tableau, so the queued Cliffords' effect
                // is already captured in the measurement basis.
                // `--reset_mode flush|init` additionally flushes at every reset, which makes the
                // measurement weight 1 and leaves the qubit with an identity tableau row.
                let force_flush = is_reset && reset_mode != ResetMode::Frame;
                let should_flush =
                    force_flush || last_weight.map_or(false, |w| w > effective_max_weight);
                let ps = if should_flush {
                    flush_clifford_queue(
                        &mut clifford_queue,
                        &mut ops,
                        &mut tableau,
                        &mut layers,
                        &mut frame,
                        n_qubits,
                        false,
                    );
                    make_z_pauli(n_qubits, *qubit, false)
                } else {
                    conjugated
                };
                let mut emitted = TransPauli::from_pauli_string(&ps, "M");
                emitted.apply_frame(frame.deps(&ps));
                let id = next_meas_id;
                next_meas_id += 1;
                emitted.meta.meas_id = Some(id);
                emitted.meta.logical_qubit = Some(*qubit);
                emitted.meta.is_reset = is_reset;
                if let QasmGate::Measure { clbit: Some(c), .. } = gate {
                    emitted.meta.clbit = Some(*c);
                    clbit_meas.insert(*c, id);
                }
                layers.place(&pauli_support(&emitted), LAYER_COST_M);
                ops.push(TransItem::Pauli(emitted));

                if is_reset {
                    // The qubit now holds |outcome>; to make it |0> the state is owed X_q^outcome.
                    // That is a classical correction, so it joins the frame (written in the
                    // physical coordinates the tableau currently defines).
                    if reset_mode != ResetMode::Init {
                        let mut x_q = PauliString::identity(n_qubits);
                        x_q.x_bits[*qubit] = true;
                        frame.add(&tableau.conjugate(&x_q), id);
                    } else {
                        // Physically re-initialising the patch replaces the frame's correction:
                        // the flush made this patch exactly qubit `q`, so whatever Pauli was owed
                        // to it is gone along with the old state.
                        frame.dx[*qubit].clear();
                        frame.dz[*qubit].clear();
                    }
                    if reset_mode == ResetMode::Init {
                        // The flush above made the tableau the identity, so the patch holds
                        // this qubit alone and can be re-initialized without touching others.
                        let r = TransPauli::from_pauli_string(
                            &make_z_pauli(n_qubits, *qubit, false),
                            "R",
                        );
                        layers.place(&pauli_support(&r), LAYER_COST_M);
                        ops.push(TransItem::Pauli(r));
                    }
                }
            }
            QasmGate::CondPauli { gate: g, qubit, clbit, value } => {
                // An outcome of 1 owes the state the Pauli; an outcome of 0 owes it nothing.
                // `if(c==0)` is therefore the unconditional Pauli followed by `if(c==1)`.
                // A bit no measurement has written yet reads as 0.
                let m = clbit_meas.get(clbit).copied();
                if !*value {
                    let pauli = QasmGate::Clifford1Q { gate: *g, qubit: *qubit };
                    push_clifford(&mut tableau, &pauli);
                    clifford_queue.push(pauli);
                }
                if let Some(m) = m {
                    let mut p = PauliString::identity(n_qubits);
                    p.x_bits[*qubit] = matches!(g, Gate1Q::X | Gate1Q::Y);
                    p.z_bits[*qubit] = matches!(g, Gate1Q::Z | Gate1Q::Y);
                    frame.add(&tableau.conjugate(&p), m);
                }
            }
            QasmGate::CondGate { inner, terms } => {
                // `resolve_static_conditions` left only terms whose bit an earlier measurement
                // wrote, so each names a real outcome.
                let mut cond: Vec<(usize, bool)> = terms
                    .iter()
                    .map(|(cb, v)| (clbit_meas[cb], *v))
                    .collect();
                cond.sort_unstable();
                cond.dedup();
                let q = inner.qubits()[0];
                match inner.as_ref() {
                    QasmGate::Clifford1Q { gate: g, .. }
                        if matches!(g, Gate1Q::X | Gate1Q::Y | Gate1Q::Z) =>
                    {
                        // A Pauli under an AND of outcomes: a linear frame term on a derived bit.
                        let id = derived_outcome(&cond, &mut next_derived, &mut derived_defs, &mut ops);
                        let mut p = PauliString::identity(n_qubits);
                        p.x_bits[q] = matches!(g, Gate1Q::X | Gate1Q::Y);
                        p.z_bits[q] = matches!(g, Gate1Q::Z | Gate1Q::Y);
                        frame.add(&tableau.conjugate(&p), id);
                    }
                    _ => {
                        // A physical conditional gate. Flush so logical qubit q is exactly one
                        // physical patch, then settle the frame on q so that the gate sees none.
                        flush_clifford_queue(
                            &mut clifford_queue,
                            &mut ops,
                            &mut tableau,
                            &mut layers,
                            &mut frame,
                            n_qubits,
                            false,
                        );
                        let mut needs: BTreeSet<usize> = cond.iter().map(|&(m, _)| m).collect();
                        if let Some(fix) = pay_frame(&mut frame, q) {
                            if let Event::Fix { x_deps, z_deps, .. } = &fix {
                                let all: Vec<usize> =
                                    x_deps.iter().chain(z_deps.iter()).copied().collect();
                                needs.extend(real_outcomes(&all, &derived_defs));
                            }
                            ops.push(TransItem::Event(fix));
                        }
                        let needs: Vec<usize> = needs.into_iter().collect();
                        match inner.as_ref() {
                            QasmGate::T { .. } | QasmGate::Tdg { .. } => {
                                let negative = matches!(inner.as_ref(), QasmGate::Tdg { .. });
                                let ps = make_z_pauli(n_qubits, q, negative);
                                let mut emitted = TransPauli::from_pauli_string(&ps, "T");
                                emitted.apply_frame(frame.deps(&ps));
                                emitted.meta.cond = cond.clone();
                                emitted.meta.needs = needs;
                                layers.place(&pauli_support(&emitted), LAYER_COST_T);
                                ops.push(TransItem::Pauli(emitted));
                            }
                            QasmGate::Clifford1Q { gate: g, qubit } => {
                                // The lookup may realise the gate with X/Z, which are never
                                // emitted; whatever Pauli is missing is owed only if the gate ran.
                                let seq: Vec<QasmGate> = optimize_single_qubit_sequence(&[*g])
                                    .into_iter()
                                    .map(|gg| QasmGate::Clifford1Q { gate: gg, qubit: *qubit })
                                    .collect();
                                let owed = dropped_pauli(&seq, n_qubits);
                                for mut c in to_trans_cliffords(&seq, n_qubits) {
                                    c.meta.cond = cond.clone();
                                    c.meta.needs = needs.clone();
                                    let (qubits, cost) = clifford_support(&c);
                                    layers.place(&qubits, cost);
                                    ops.push(TransItem::Clifford(c));
                                }
                                if owed.weight() > 0 {
                                    let id = derived_outcome(
                                        &cond,
                                        &mut next_derived,
                                        &mut derived_defs,
                                        &mut ops,
                                    );
                                    frame.add(&owed, id);
                                }
                            }
                            other => unreachable!("conditional {:?} is not a single-qubit gate", other),
                        }
                    }
                }
            }
            QasmGate::Barrier => {}
        }
    }

    ops
}

fn make_z_pauli(n: usize, q: usize, negative: bool) -> PauliString {
    let mut ps = PauliString::identity(n);
    ps.z_bits[q] = true;
    ps.sign = negative;
    ps
}

/// Predicted depth of a transpiled circuit: the layering `Circuit::compute_layers` will derive
/// from it downstream, with each product charged its logical-cycle span.
fn predicted_depth(items: &[TransItem], n_qubits: usize) -> usize {
    let mut layers = LayerModel::new(n_qubits);
    for item in items {
        let (qubits, cost) = match item {
            TransItem::Pauli(p) => {
                (pauli_support(p), if p.label == "M" { LAYER_COST_M } else { LAYER_COST_T })
            }
            TransItem::Clifford(c) => clifford_support(c),
            TransItem::Event(_) => continue,
        };
        layers.place(&qubits, cost);
    }
    layers.max_depth()
}

/// Weight thresholds tried by [`transpile_best`].
///
/// With trailing deferral enabled these two reproduce the full 0..=10 sweep's best result exactly
/// on 8 of 11 benchmark circuits and within 1.5% on the rest: flushes are cheap enough that the
/// optimum collapses to either "never flush" or "keep every product weight 1".
const CANDIDATE_WEIGHTS: [i32; 2] = [0, 1];

/// Transpiles at each candidate weight and keeps the shallowest result.
///
/// Predicted depth is cheap to evaluate in-process and tracks scheduling well, so selecting
/// between whole transpilations replaces the external `max_width` sweep and the hand-maintained
/// table of per-circuit best weights.
fn transpile_best(
    n_qubits: usize, gates: &[QasmGate], reset_mode: ResetMode,
) -> (Vec<TransItem>, i32) {
    let mut best: Option<(Vec<TransItem>, i32, usize)> = None;
    for max_weight in CANDIDATE_WEIGHTS {
        let items = transpile_opts(n_qubits, gates, max_weight, true, reset_mode);
        let depth = predicted_depth(&items, n_qubits);
        println!("  candidate max_width {:<3} predicted depth {}", max_weight, depth);
        if best.as_ref().is_none_or(|(_, _, d)| depth < *d) {
            best = Some((items, max_weight, depth));
        }
    }
    let (items, weight, depth) = best.expect("at least one candidate");
    println!("Selected max_width {} (predicted depth {})", weight, depth);
    (items, weight)
}

fn write_trans(output_path: &str, items: &[TransItem]) -> io::Result<(usize, usize)> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let mut n_ts = 0usize;
    let mut n_cliffords = 0usize;

    // Line index of each measurement product: a product that needs an outcome lists that line as
    // `@line`, which the scheduler turns into a DAG edge. (Events have no line.)
    let mut meas_line: HashMap<usize, usize> = HashMap::new();
    let mut line = 0usize;
    for item in items {
        match item {
            TransItem::Pauli(p) => {
                if let Some(id) = p.meta.meas_id {
                    meas_line.insert(id, line);
                }
                line += 1;
            }
            TransItem::Clifford(_) => line += 1,
            TransItem::Event(_) => {}
        }
    }

    for item in items {
        let needs = match item {
            TransItem::Pauli(p) => {
                if p.label == "T" {
                    n_ts += 1;
                }
                &p.meta.needs
            }
            TransItem::Clifford(c) => {
                n_cliffords += 1;
                &c.meta.needs
            }
            TransItem::Event(_) => continue,
        };
        write!(writer, "{}", item)?;
        if !needs.is_empty() {
            let mut lines: Vec<usize> = needs
                .iter()
                .map(|m| *meas_line.get(m).expect("a needed outcome comes from an emitted M"))
                .collect();
            lines.sort_unstable();
            lines.dedup();
            let lines: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
            write!(writer, "@{}", lines.join(","))?;
        }
        writeln!(writer)?;
    }

    Ok((n_ts, n_cliffords))
}

/// True when the circuit has any classical dependence: a reset, a conditional Pauli, or a
/// measurement that stores its outcome.
fn is_dynamic(gates: &[QasmGate]) -> bool {
    gates.iter().any(|g| {
        matches!(
            g,
            QasmGate::Reset { .. }
                | QasmGate::CondPauli { .. }
                | QasmGate::CondGate { .. }
                | QasmGate::Measure { clbit: Some(_), .. }
        )
    })
}

fn json_usize_list(v: &[usize]) -> String {
    let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn json_terms(t: &[(usize, bool)]) -> String {
    let parts: Vec<String> = t.iter().map(|&(m, v)| format!("[{},{}]", m, v as u8)).collect();
    format!("[{}]", parts.join(","))
}

/// Writes the classical sidecar (version 2). Everything a downstream executor needs beyond the
/// `.trans` lines themselves:
///
///   * `products`: one record per `.trans` line, in file order (the line index is the product id
///     the scheduler assigns, because the transpiler never emits X/Z lines). T/M/R records carry
///     the sign and `frame_deps`; M records also `meas_id`, `logical_qubit`, `clbit`,
///     `is_reset`. A record of a *conditional* product also has `cond` (it runs only if every
///     `[outcome, value]` holds) and `needs` (the outcomes it waits for, i.e. its `@` edges).
///   * `events`: software-only actions with no line, each `before` the line of that index.
///     `fix` physically applies the Pauli the frame owed a qubit.
///   * `derived`: pseudo-outcomes, each the AND of `terms`, usable in any `frame_deps`.
///
/// Hand-rolled JSON keeps the crate free of a serde dependency.
fn write_sidecar(
    path: &str, n_qubits: usize, mode: ResetMode, items: &[TransItem],
) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let mode = match mode {
        ResetMode::Frame => "frame",
        ResetMode::Flush => "flush",
        ResetMode::Init => "init",
    };
    let cond_fields = |m: &PauliMeta| -> String {
        if m.cond.is_empty() {
            String::new()
        } else {
            format!(",\"cond\":{},\"needs\":{}", json_terms(&m.cond), json_usize_list(&m.needs))
        }
    };
    let (mut products, mut events, mut derived) = (Vec::new(), Vec::new(), Vec::new());
    let mut line = 0usize;
    for item in items {
        match item {
            TransItem::Clifford(c) => {
                products
                    .push(format!("{{\"i\":{},\"label\":\"C\"{}}}", line, cond_fields(&c.meta)));
                line += 1;
            }
            TransItem::Pauli(p) => {
                let m = &p.meta;
                let opt = |o: Option<usize>| o.map_or("null".to_string(), |v| v.to_string());
                products.push(format!(
                    "{{\"i\":{},\"label\":\"{}\",\"sign\":\"{}\",\"frame_deps\":{},\"meas_id\":{},\
                     \"logical_qubit\":{},\"clbit\":{},\"is_reset\":{}{}}}",
                    line,
                    p.label,
                    p.sign,
                    json_usize_list(&m.frame_deps),
                    opt(m.meas_id),
                    opt(m.logical_qubit),
                    opt(m.clbit),
                    m.is_reset,
                    cond_fields(m)
                ));
                line += 1;
            }
            TransItem::Event(Event::Fix { qubit, x_deps, x_const, z_deps, z_const }) => {
                events.push(format!(
                    "{{\"before\":{},\"op\":\"fix\",\"qubit\":{},\"x_deps\":{},\"x_const\":{},\
                     \"z_deps\":{},\"z_const\":{}}}",
                    line,
                    qubit,
                    json_usize_list(x_deps),
                    x_const,
                    json_usize_list(z_deps),
                    z_const
                ));
            }
            TransItem::Event(Event::Derived { id, terms }) => {
                derived.push(format!("{{\"id\":{},\"terms\":{}}}", id, json_terms(terms)));
            }
        }
    }
    writeln!(
        w,
        "{{\"version\":2,\"n_qubits\":{},\"reset_mode\":\"{}\",\"products\":[\n{}\n],\"events\":[\n{}\n],\"derived\":[\n{}\n]}}",
        n_qubits,
        mode,
        products.join(",\n"),
        events.join(",\n"),
        derived.join(",\n")
    )?;
    Ok(())
}

fn count_stats(items: &[TransItem]) -> (usize, usize, usize, f64) {
    let n_cliffords = items.iter().filter(|i| matches!(i, TransItem::Clifford(_))).count();
    let pps: Vec<&TransPauli> = items
        .iter()
        .filter_map(|i| {
            if let TransItem::Pauli(p) = i {
                if p.label == "T" { Some(p) } else { None }
            } else {
                None
            }
        })
        .collect();
    let n_pps = pps.len();
    let avg_weight = if n_pps > 0 {
        let tot_weight: usize =
            pps.iter().map(|p| p.ops.iter().filter(|&&c| c != 'I').count()).sum();
        tot_weight as f64 / n_pps as f64
    } else {
        0.0
    };
    let n_lines = items.iter().filter(|i| !matches!(i, TransItem::Event(_))).count();
    (n_cliffords, n_pps, n_lines, avg_weight)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    utils::print_banner("transpile");
    let _total_timer = Timer::new("transpilation");

    let input_file = &args.input_file;
    if !input_file.ends_with(".cliffordt.qasm") {
        return Err(format!(
            "Input file must be a .cliffordt.qasm file produced by a Clifford+T compiler in data_processing/, got: {}",
            input_file
        )
        .into());
    }

    println!("Loading compiled circuit from {}", input_file);
    let _load_timer = Timer::new("load circuit");
    let (n_qubits, gates) = parse_qasm(input_file)?;
    drop(_load_timer);

    let tot_gates = gates.iter().filter(|g| !matches!(g, QasmGate::Measure { .. })).count();
    let clifford_gates = gates
        .iter()
        .filter(|g| matches!(g, QasmGate::Clifford1Q { .. } | QasmGate::Clifford2Q { .. }))
        .count();

    println!("Circuit has {} gates on {} qubits", tot_gates, n_qubits);

    let mut gate_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for g in &gates {
        let name = match g {
            QasmGate::Clifford1Q { gate, .. } => format!("{:?}", gate),
            QasmGate::Clifford2Q { gate, .. } => format!("{:?}", gate),
            QasmGate::T { .. } => "T".to_string(),
            QasmGate::Tdg { .. } => "Tdg".to_string(),
            QasmGate::Measure { .. } => "Measure".to_string(),
            QasmGate::Reset { .. } => "Reset".to_string(),
            QasmGate::CondPauli { gate, .. } => format!("if-{:?}", gate),
            QasmGate::CondGate { inner, .. } => match inner.as_ref() {
                QasmGate::Clifford1Q { gate, .. } => format!("if-{:?}", gate),
                QasmGate::T { .. } => "if-T".to_string(),
                QasmGate::Tdg { .. } => "if-Tdg".to_string(),
                _ => "if-?".to_string(),
            },
            QasmGate::Barrier => continue,
        };
        gate_names.insert(name);
    }
    println!("Gate set: {}", gate_names.iter().cloned().collect::<Vec<_>>().join(", "));

    if args.auto {
        println!("Weight selection: auto (lowest predicted depth across candidates)");
    } else {
        println!("Weight selection: fixed (max weight {})", args.max_width);
    }

    let _tableau_timer = Timer::new("tableau");
    let items = if args.auto {
        transpile_best(n_qubits, &gates, args.reset_mode).0
    } else {
        transpile_opts(n_qubits, &gates, args.max_width, args.defer_trailing, args.reset_mode)
    };
    drop(_tableau_timer);

    let (n_cliffords, n_pps, tot_post, avg_weight) = count_stats(&items);
    let tot_delta = tot_gates as i64 - tot_post as i64;
    let clifford_delta = clifford_gates as i64 - n_cliffords as i64;

    if tot_gates > 0 {
        println!("Circuit length:    {} (before) -> {} (after transpilation)", tot_gates, tot_post);
        println!(
            "  Overall reduction: {} operations removed ({:.1}% reduction)",
            tot_delta,
            100.0 * tot_delta as f64 / tot_gates as f64
        );
    } else {
        println!("  Overall reduction: N/A (empty circuit)");
    }
    println!(
        "  Clifford gates:    {} (before) -> {} (after transpilation)",
        clifford_gates, n_cliffords
    );
    if clifford_gates > 0 {
        println!(
            "  Clifford reduction: {} gates removed ({:.1}% reduction)",
            clifford_delta,
            100.0 * clifford_delta as f64 / clifford_gates as f64
        );
    } else {
        println!("  Clifford reduction: N/A (no Cliffords before transpilation)");
    }
    println!("  Non-Clifford Pauli products: {}", n_pps);
    println!("  Average Pauli product weight: {:.2}", avg_weight);

    let output_stem = if args.output_file.is_empty() {
        let p = Path::new(input_file);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        // stem is "foo.cliffordt" — strip the inner extension
        if stem.ends_with(".cliffordt") {
            stem[..stem.len() - ".cliffordt".len()].to_string()
        } else {
            stem.to_string()
        }
    } else {
        args.output_file.clone()
    };
    let output_path = if args.output_file.is_empty() {
        // Auto-selected output must not collide with the .trans<N> sweep artifacts.
        if args.auto {
            format!("{}.transauto", output_stem)
        } else {
            format!("{}.trans{}", output_stem, args.max_width)
        }
    } else {
        args.output_file
    };

    let (n_ts, n_cliffords_written) = write_trans(&output_path, &items)?;
    println!("Wrote transpiled circuit to {}", output_path);
    println!("Wrote {} T gates and {} Cliffords", n_ts, n_cliffords_written);

    // Dynamic circuits (reset / classical feed-forward) also get the classical sidecar. The
    // scheduler never reads it; it exists so an executor or `validate_dynamic.py` can recover the
    // true measurement outcomes and the sign of every rotation.
    if args.sidecar || is_dynamic(&gates) {
        let sidecar_path = format!("{}.dyn.json", output_path);
        write_sidecar(&sidecar_path, n_qubits, args.reset_mode, &items)?;
        let n_meas = items
            .iter()
            .filter(|i| matches!(i, TransItem::Pauli(p) if p.meta.meas_id.is_some()))
            .count();
        println!("Wrote classical sidecar ({} measurement outcomes) to {}", n_meas, sidecar_path);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_qasm(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::with_suffix(".cliffordt.qasm").unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f
    }

    #[test]
    fn parse_qasm_basic() {
        let f = write_qasm(&[
            "OPENQASM 2.0;",
            "include \"qelib1.inc\";",
            "qreg q[2];",
            "h q[0];",
            "t q[1];",
            "cx q[0], q[1];",
        ]);
        let (n, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(n, 2);
        assert_eq!(gates.len(), 3);
        assert!(matches!(gates[0], QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 0 }));
        assert!(matches!(gates[1], QasmGate::T { qubit: 1 }));
        assert!(matches!(
            gates[2],
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 }
        ));
    }

    #[test]
    fn parse_qasm_sdg_tdg() {
        let f = write_qasm(&["qreg q[1];", "sdg q[0];", "tdg q[0];"]);
        let (_, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        assert!(matches!(gates[0], QasmGate::Clifford1Q { gate: Gate1Q::Sdg, qubit: 0 }));
        assert!(matches!(gates[1], QasmGate::Tdg { qubit: 0 }));
    }

    #[test]
    fn make_z_pauli_positive() {
        let p = make_z_pauli(3, 1, false);
        assert_eq!(p.pauli_at(0), 'I');
        assert_eq!(p.pauli_at(1), 'Z');
        assert_eq!(p.pauli_at(2), 'I');
        assert!(!p.sign);
    }

    #[test]
    fn make_z_pauli_negative() {
        let p = make_z_pauli(2, 0, true);
        assert_eq!(p.pauli_at(0), 'Z');
        assert!(p.sign);
    }

    #[test]
    fn trans_pauli_display_t_gate() {
        let ps = PauliString::from_str("+ZII");
        let tp = TransPauli::from_pauli_string(&ps, "T");
        assert_eq!(format!("{}", tp), "+Z__<T>");
    }

    #[test]
    fn trans_pauli_display_measurement() {
        let ps = PauliString::from_str("-IZI");
        let tp = TransPauli::from_pauli_string(&ps, "M");
        assert_eq!(format!("{}", tp), "-_Z_<M>");
    }

    #[test]
    fn trans_clifford_cx_display() {
        let gate = QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 };
        let tc = TransClifford::from_qasm_gate(&gate, 2).unwrap();
        assert_eq!(format!("{}", tc), "+ZX<CX>");
    }

    #[test]
    fn trans_clifford_s_display() {
        let gate = QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: 1 };
        let tc = TransClifford::from_qasm_gate(&gate, 3).unwrap();
        assert_eq!(format!("{}", tc), "+_Z_<S>");
    }

    #[test]
    fn trans_clifford_x_is_dropped() {
        let gate = QasmGate::Clifford1Q { gate: Gate1Q::X, qubit: 0 };
        assert!(TransClifford::from_qasm_gate(&gate, 2).is_none());
    }

    #[test]
    fn trans_clifford_z_is_dropped() {
        let gate = QasmGate::Clifford1Q { gate: Gate1Q::Z, qubit: 0 };
        assert!(TransClifford::from_qasm_gate(&gate, 2).is_none());
    }

    #[test]
    fn transpile_single_t_gate_no_cliffords() {
        let gates = vec![QasmGate::T { qubit: 0 }];
        let items = transpile(2, &gates, 0);
        let t_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "T")).collect();
        assert_eq!(t_items.len(), 1);
        if let TransItem::Pauli(p) = &t_items[0] {
            assert_eq!(p.sign, Sign::Plus);
            assert_eq!(p.ops[0], 'Z');
            assert_eq!(p.ops[1], 'I');
        }
    }

    #[test]
    fn transpile_h_then_t_gives_x_pauli() {
        // H maps Z→X, so T (a Z rotation) becomes an X rotation after conjugation.
        let gates =
            vec![QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 0 }, QasmGate::T { qubit: 0 }];
        let items = transpile(2, &gates, 0);
        let t_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "T")).collect();
        assert_eq!(t_items.len(), 1);
        if let TransItem::Pauli(p) = &t_items[0] {
            assert_eq!(p.ops[0], 'X');
        }
    }

    #[test]
    fn transpile_tdg_gives_negative_sign() {
        let gates = vec![QasmGate::Tdg { qubit: 0 }];
        let items = transpile(1, &gates, 0);
        let t_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "T")).collect();
        assert_eq!(t_items.len(), 1);
        if let TransItem::Pauli(p) = &t_items[0] {
            assert_eq!(p.sign, Sign::Minus);
            assert_eq!(p.ops[0], 'Z');
        }
    }

    #[test]
    fn transpile_measurements_appended_for_all_qubits() {
        let gates = vec![QasmGate::T { qubit: 0 }];
        let items = transpile(2, &gates, 0);
        let m_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "M")).collect();
        assert_eq!(m_items.len(), 2);
    }

    #[test]
    fn transpile_max_weight_1_flushes_cliffords() {
        let gates =
            vec![QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: 0 }, QasmGate::T { qubit: 0 }];
        let items_unlimited = transpile(2, &gates, 0);
        let cliffords_unlimited: Vec<_> =
            items_unlimited.iter().filter(|i| matches!(i, TransItem::Clifford(_))).collect();
        assert_eq!(cliffords_unlimited.len(), 0);
    }

    #[test]
    fn count_stats_basic() {
        let items = vec![
            TransItem::Pauli(TransPauli {
                sign: Sign::Plus,
                ops: vec!['Z', 'I'],
                label: "T".to_string(),
                meta: PauliMeta::default(),
            }),
            TransItem::Pauli(TransPauli {
                sign: Sign::Plus,
                ops: vec!['Z', 'Z'],
                label: "T".to_string(),
                meta: PauliMeta::default(),
            }),
            TransItem::Clifford(TransClifford { ops: vec!['Z', '_'], name: "S".to_string(), meta: PauliMeta::default() }),
        ];
        let (n_cliffords, n_paulis, tot, avg_weight) = count_stats(&items);
        assert_eq!(n_cliffords, 1);
        assert_eq!(n_paulis, 2);
        assert_eq!(tot, 3);
        assert!((avg_weight - 1.5).abs() < 1e-9);
    }

    #[test]
    fn count_stats_all_cliffords() {
        let items = vec![
            TransItem::Clifford(TransClifford { ops: vec!['X', '_'], name: "CX".to_string(), meta: PauliMeta::default() }),
            TransItem::Clifford(TransClifford { ops: vec!['Z', '_'], name: "S".to_string(), meta: PauliMeta::default() }),
        ];
        let (n_cliffords, n_paulis, tot, _avg_weight) = count_stats(&items);
        assert_eq!(n_cliffords, 2);
        assert_eq!(n_paulis, 0);
        assert_eq!(tot, 2);
    }

    #[test]
    fn count_stats_empty() {
        let items: Vec<TransItem> = vec![];
        let (n_cliffords, n_paulis, tot, avg_weight) = count_stats(&items);
        assert_eq!(n_cliffords, 0);
        assert_eq!(n_paulis, 0);
        assert_eq!(tot, 0);
        assert_eq!(avg_weight, 0.0);
    }

    #[test]
    fn make_z_pauli_has_weight_one() {
        let ps = make_z_pauli(4, 2, false);
        assert_eq!(ps.weight(), 1);
    }

    #[test]
    fn make_z_pauli_correct_qubit() {
        let ps = make_z_pauli(4, 1, false);
        assert_eq!(ps.pauli_at(1), 'Z');
        assert_eq!(ps.pauli_at(0), 'I');
        assert_eq!(ps.pauli_at(2), 'I');
    }

    #[test]
    fn parse_qasm_sdg_and_tdg_gates() {
        let f = write_qasm(&["OPENQASM 2.0;", "qreg q[2];", "sdg q[0];", "tdg q[1];"]);
        let (n_qubits, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(n_qubits, 2);
        assert_eq!(gates.len(), 2);
    }

    #[test]
    fn transpile_sx_gate_produces_clifford() {
        // Clifford gates are only flushed when a T gate's conjugated weight exceeds
        // max_weight. Use a 2-qubit circuit: cx entangles qubits so that T on q[1]
        // has weight 2 (> max_weight=1), triggering a flush of the clifford queue
        // which contains the preceding sx gate.
        let f =
            write_qasm(&["OPENQASM 2.0;", "qreg q[2];", "cx q[0],q[1];", "sx q[0];", "t q[1];"]);
        let (n_qubits, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        let items = transpile(n_qubits, &gates, 1);
        let has_clifford = items.iter().any(|i| matches!(i, TransItem::Clifford(_)));
        assert!(has_clifford, "SX gate should produce a Clifford TransItem");
    }

    #[test]
    fn transpile_two_qubit_circuit_appends_two_measurements() {
        let f = write_qasm(&["OPENQASM 2.0;", "qreg q[2];", "t q[0];"]);
        let (n_qubits, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        let items = transpile(n_qubits, &gates, 0);
        let measurement_count = items
            .iter()
            .filter(|i| matches!(i, TransItem::Pauli(p) if p.label.contains('M')))
            .count();
        assert_eq!(measurement_count, 2, "should append one measurement per qubit");
    }

    #[test]
    fn trans_pauli_display_negative_sign() {
        let ps = make_z_pauli(2, 0, true); // negative
        let tp = TransPauli::from_pauli_string(&ps, "T");
        let s = format!("{}", tp);
        assert!(s.starts_with('-'), "negative pauli should start with '-': {}", s);
    }

    #[test]
    fn trans_clifford_cx_has_two_qubit_paulis() {
        // Clifford gates are only flushed when a T gate's conjugated weight exceeds
        // max_weight. After cx q[0],q[1], Z on q[1] maps to Z_0*Z_1 (weight 2).
        // With max_weight=1, weight 2 > 1 triggers a flush of the clifford queue
        // which contains the CX gate.
        let f = write_qasm(&["OPENQASM 2.0;", "qreg q[2];", "cx q[0],q[1];", "t q[1];"]);
        let (n_qubits, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        let items = transpile(n_qubits, &gates, 1);
        let cx_item = items
            .iter()
            .find(|i| if let TransItem::Clifford(c) = i { c.name == "CX" } else { false });
        assert!(cx_item.is_some(), "CX gate should produce a CX Clifford item");
        if let Some(TransItem::Clifford(c)) = cx_item {
            assert_eq!(c.ops.len(), 2, "CX should have 2 qubit entries");
        }
    }

    // --- Dynamic (layer-aware) flush policy ---

    //fn n_cliffords(items: &[TransItem]) -> usize {
    //    items.iter().filter(|i| matches!(i, TransItem::Clifford(_))).count()
    //}

    /// Reference layer count: rebuilds the qubit-overlap DAG over the emitted stream and runs
    /// Kahn's algorithm, exactly as `Circuit::gen_deps` + `Circuit::compute_layers` do.
    fn layers_via_kahn(items: &[TransItem], n_qubits: usize) -> usize {
        // Events are not scheduled products, so they have no place in the DAG.
        let items: Vec<&TransItem> =
            items.iter().filter(|i| !matches!(i, TransItem::Event(_))).collect();
        let supports: Vec<Vec<usize>> = items
            .iter()
            .map(|i| match i {
                TransItem::Pauli(p) => pauli_support(p),
                TransItem::Clifford(c) => clifford_support(c).0,
                TransItem::Event(_) => unreachable!(),
            })
            .collect();
        let mut parents: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
        let mut last: Vec<Option<usize>> = vec![None; n_qubits];
        for (id, support) in supports.iter().enumerate() {
            for &q in support {
                if let Some(prev) = last[q] {
                    parents[id].push(prev);
                    children[prev].push(id);
                }
                last[q] = Some(id);
            }
        }
        let mut in_degrees: Vec<usize> = parents.iter().map(|p| p.len()).collect();
        let mut ready: Vec<usize> = (0..items.len()).filter(|&i| in_degrees[i] == 0).collect();
        let mut n_layers = 0;
        while !ready.is_empty() {
            n_layers += 1;
            let mut next = Vec::new();
            for &cur in &ready {
                for &child in &children[cur] {
                    in_degrees[child] -= 1;
                    if in_degrees[child] == 0 {
                        next.push(child);
                    }
                }
            }
            ready = next;
        }
        n_layers
    }

    /// Layer count the `LayerModel` predicts for an emitted stream, at unit cost per product.
    fn layers_via_model(items: &[TransItem], n_qubits: usize) -> usize {
        let mut lm = LayerModel::new(n_qubits);
        let mut max_layer = 0;
        for item in items {
            let support = match item {
                TransItem::Pauli(p) => pauli_support(p),
                TransItem::Clifford(c) => clifford_support(c).0,
                TransItem::Event(_) => continue,
            };
            max_layer = max_layer.max(lm.place(&support, 1));
        }
        max_layer
    }

    #[test]
    fn layer_model_matches_compute_layers() {
        // Correctness anchor: at unit cost the model must reproduce the Kahn layering that
        // Circuit::compute_layers performs downstream on the same emitted stream.
        let gates = vec![
            QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 0 },
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 },
            QasmGate::T { qubit: 1 },
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 1, target: 2 },
            QasmGate::T { qubit: 2 },
            QasmGate::T { qubit: 0 },
            QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: 3 },
            QasmGate::T { qubit: 3 },
        ];
        for max_width in [0, 1, 2] {
            let items = transpile(4, &gates, max_width);
            assert_eq!(
                layers_via_model(&items, 4),
                layers_via_kahn(&items, 4),
                "model and Kahn layering disagree at max_width {}",
                max_width
            );
        }
    }

    #[test]
    fn layer_model_peek_place_basics() {
        let mut lm = LayerModel::new(4);
        // Disjoint qubits share a layer.
        assert_eq!(lm.place(&[0], 1), 1);
        assert_eq!(lm.place(&[1], 1), 1);
        // Same qubit serialises.
        assert_eq!(lm.place(&[0], 1), 2);
        // peek does not mutate.
        assert_eq!(lm.peek(&[0]), 3);
        assert_eq!(lm.peek(&[0]), 3);
        // A multi-layer product advances depth by its cost.
        let mut lm2 = LayerModel::new(2);
        assert_eq!(lm2.place(&[0, 1], 3), 1);
        assert_eq!(lm2.peek(&[0]), 4, "a 3-layer product should occupy layers 1..=3");
        // An empty support still consumes a layer slot.
        assert_eq!(LayerModel::new(1).peek(&[]), 1);
    }

    /// Depth added to an empty layer model by flushing `queue`.
    fn flush_depth(queue: &[QasmGate], n_qubits: usize) -> usize {
        let mut lm = LayerModel::new(n_qubits);
        apply_flushed_cliffords(&mut lm, queue, n_qubits);
        lm.max_depth()
    }

    #[test]
    fn flush_cost_accounts_for_clifford_parallelism() {
        let one_cx = vec![QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 }];
        let disjoint_cx = vec![
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 },
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 2, target: 3 },
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 4, target: 5 },
        ];
        assert_eq!(flush_depth(&one_cx, 8), LAYER_COST_CX);
        assert_eq!(
            flush_depth(&disjoint_cx, 8),
            LAYER_COST_CX,
            "CX gates on disjoint qubits run in parallel and must not be charged cumulatively"
        );
    }

    #[test]
    fn flush_cost_serialises_overlapping_cliffords() {
        // Two CX gates sharing a qubit cannot run in parallel, so they cost twice as much.
        let chained = vec![
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 },
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 1, target: 2 },
        ];
        assert_eq!(flush_depth(&chained, 4), 2 * LAYER_COST_CX);
    }

    /// Compares two tableaus by their action on every basis Pauli, ignoring sign. Signs are
    /// discarded downstream (`PauliProduct::set_from_str` skips the sign character) and X/Y/Z
    /// gates are never emitted, being tracked in the Pauli frame.
    fn same_up_to_sign(a: &Tableau, b: &Tableau, n: usize) -> bool {
        (0..n).all(|q| {
            [false, true].iter().all(|&use_x| {
                let mut p = PauliString::identity(n);
                if use_x {
                    p.x_bits[q] = true;
                } else {
                    p.z_bits[q] = true;
                }
                let (ia, ib) = (a.conjugate(&p), b.conjugate(&p));
                ia.x_bits == ib.x_bits && ia.z_bits == ib.z_bits
            })
        })
    }

    /// The single-qubit lookup (`apply_1q_to_state`) and the tableau must agree on every gate's
    /// action on X, Y and Z, signs included. The lookup once had the Y rows of SX/SXdg flipped,
    /// which only changed signs (invisible to the scheduler) but corrupted outcome bits.
    #[test]
    fn single_qubit_lookup_rules_agree_with_tableau() {
        let code = |ps: &PauliString| pauli_code(
            match (ps.x_bits[0], ps.z_bits[0]) {
                (true, false) => 0,
                (true, true) => 1,
                _ => 2,
            },
            ps.sign,
        );
        for g in [Gate1Q::H, Gate1Q::S, Gate1Q::Sdg, Gate1Q::SX, Gate1Q::SXdg,
                  Gate1Q::X, Gate1Q::Y, Gate1Q::Z] {
            let mut t = Tableau::new(1);
            t.prepend_1q_correct(g, 0);
            let mut x = PauliString::identity(1);
            x.x_bits[0] = true;
            let mut z = PauliString::identity(1);
            z.z_bits[0] = true;
            let mut y = PauliString::identity(1);
            y.x_bits[0] = true;
            y.z_bits[0] = true;
            // The lookup is applied to arbitrary signed Paulis, so feed it each one in turn.
            let from_table = |p: u8| apply_1q_to_state(g, p, p).0;
            assert_eq!(from_table(pauli_code(0, false)), code(&t.conjugate(&x)), "{:?} on X", g);
            assert_eq!(from_table(pauli_code(2, false)), code(&t.conjugate(&z)), "{:?} on Z", g);
            let y_img = t.conjugate(&y);
            // Y carries a sign convention of its own (Y = iXZ), so compare the unsigned part and
            // the sign through the group law: image(Y) = i * image(X) * image(Z).
            let (ix, iz) = (t.conjugate(&x), t.conjugate(&z));
            let prod = PauliString::identity(1).mul_with_y_phase(&ix, &iz);
            assert_eq!(y_img.x_bits, prod.x_bits, "{:?} on Y (support)", g);
            assert_eq!(y_img.z_bits, prod.z_bits, "{:?} on Y (support)", g);
            assert_eq!(y_img.sign, prod.sign, "{:?} on Y (group law)", g);
            assert_eq!(from_table(pauli_code(1, false)), code(&y_img), "{:?} on Y", g);
        }
    }

    #[test]
    fn predicted_depth_matches_kahn_when_every_product_spans_one_layer() {
        // T and M products span a single layer each, so with no Cliffords emitted the scorer must
        // agree exactly with the Kahn layering the model is validated against.
        let gates =
            vec![QasmGate::T { qubit: 0 }, QasmGate::T { qubit: 1 }, QasmGate::T { qubit: 0 }];
        let items = transpile_full(2, &gates, 0, false);
        assert!(items.iter().all(|i| matches!(i, TransItem::Pauli(_))));
        assert_eq!(predicted_depth(&items, 2), layers_via_kahn(&items, 2));
    }

    #[test]
    fn predicted_depth_charges_cliffords_their_full_span() {
        // A CX spans 2 logical cycles and an S/SX 3, so an emitted Clifford must cost more than
        // the single layer a unit-cost layering would assign it.
        let gates = vec![
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 },
            QasmGate::T { qubit: 1 },
        ];
        let items = transpile_full(2, &gates, 1, false);
        assert!(items.iter().any(|i| matches!(i, TransItem::Clifford(_))));
        assert!(predicted_depth(&items, 2) > layers_via_kahn(&items, 2));
    }

    #[test]
    fn transpile_best_never_loses_to_any_candidate() {
        // Selection is by predicted depth, so the chosen transpilation is at least as shallow as
        // every candidate weight — including 0, the never-flush policy that is optimal on
        // circuits where emitting Cliffords never pays.
        let mut gates = vec![QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 }];
        for _ in 0..8 {
            gates.push(QasmGate::T { qubit: 1 });
        }
        let (best, _) = transpile_best(2, &gates, ResetMode::Frame);
        let best_depth = predicted_depth(&best, 2);
        for w in CANDIDATE_WEIGHTS {
            let candidate = transpile_full(2, &gates, w, true);
            assert!(best_depth <= predicted_depth(&candidate, 2), "lost to max_width {}", w);
        }
    }

    // ---- conditional gates -------------------------------------------------------------

    fn parse_lines(lines: &[&str]) -> (usize, Vec<QasmGate>) {
        let f = write_qasm(lines);
        parse_qasm(&f.path().to_string_lossy()).unwrap()
    }

    fn items_text(items: &[TransItem]) -> String {
        let f = tempfile::NamedTempFile::new().unwrap();
        let path = f.path().to_string_lossy().to_string();
        write_trans(&path, items).unwrap();
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn parse_conditionals_classify_by_gate_and_term_count() {
        let (_, gates) = parse_lines(&[
            "qreg q[2];",
            "creg a[1];",
            "creg b[1];",
            "measure q[0] -> a[0];",
            "measure q[1] -> b[0];",
            "if(a==1) h q[0];",
            "if(a==1 && b==0) t q[1];",
            "if(a==1 && b==1) x q[0];",
            "if(a==1) z q[0];",
        ]);
        let cond_h = gates.iter().filter(|g| matches!(g,
            QasmGate::CondGate { inner, terms } if terms.len() == 1
                && matches!(**inner, QasmGate::Clifford1Q { gate: Gate1Q::H, .. }))).count();
        let cond_t2 = gates.iter().filter(|g| matches!(g,
            QasmGate::CondGate { inner, terms } if terms.len() == 2
                && matches!(**inner, QasmGate::T { .. }))).count();
        let cond_x2 = gates.iter().filter(|g| matches!(g,
            QasmGate::CondGate { inner, terms } if terms.len() == 2
                && matches!(**inner, QasmGate::Clifford1Q { gate: Gate1Q::X, .. }))).count();
        let cond_z1 = gates.iter().filter(|g| matches!(g,
            QasmGate::CondPauli { gate: Gate1Q::Z, value: true, .. })).count();
        assert_eq!((cond_h, cond_t2, cond_x2, cond_z1), (1, 1, 1, 1));
    }

    #[test]
    fn parse_conditional_rejects_what_it_cannot_express() {
        for (bad, why) in [
            ("if(c==1) cx q[0], q[1];", "single-qubit"),
            ("if(c==2) h q[0];", "not 0 or 1"),
            ("if(w==1) h q[0];", "more than one bit"),
            ("if(c) h q[0];", "creg==value"),
        ] {
            let f = write_qasm(&["qreg q[2];", "creg c[1];", "creg w[2];", bad]);
            let err = parse_qasm(&f.path().to_string_lossy()).unwrap_err().to_string();
            assert!(err.contains(why), "`{}` should fail with `{}`, got: {}", bad, why, err);
        }
    }

    #[test]
    fn static_resolution_settles_conditions_on_unwritten_bits() {
        let h = Box::new(QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 0 });
        let cond = |terms: Vec<(usize, bool)>| QasmGate::CondGate { inner: h.clone(), terms };
        let gates = vec![
            cond(vec![(0, true)]),             // bit 0 unwritten: reads 0, so never runs
            cond(vec![(0, false)]),            // ...and `==0` always holds: plain H
            QasmGate::Measure { qubit: 1, clbit: Some(1) },
            cond(vec![(1, true), (0, false)]), // the unwritten term drops out
            cond(vec![(1, true), (1, false)]), // contradictory: never runs
        ];
        let out = resolve_static_conditions(&gates);
        assert_eq!(out.len(), 3, "{:?}", out);
        assert!(matches!(out[0], QasmGate::Clifford1Q { gate: Gate1Q::H, .. }));
        assert!(matches!(out[1], QasmGate::Measure { .. }));
        assert!(matches!(&out[2], QasmGate::CondGate { terms, .. } if terms == &vec![(1, true)]));
    }

    #[test]
    fn conditional_clifford_is_physical_and_waits_for_its_measurement() {
        let (n, gates) = parse_lines(&[
            "qreg q[2];",
            "creg m[1];",
            "h q[1];",
            "t q[1];",
            "measure q[1] -> m[0];",
            "if(m==1) s q[0];",
            "t q[0];",
        ]);
        let items = transpile_opts(n, &gates, 0, false, ResetMode::Frame);
        let conds: Vec<&TransClifford> = items
            .iter()
            .filter_map(|i| if let TransItem::Clifford(c) = i { Some(c) } else { None })
            .filter(|c| !c.meta.cond.is_empty())
            .collect();
        assert_eq!(conds.len(), 1, "one conditional S line");
        assert_eq!(conds[0].name, "S");
        assert_eq!(conds[0].meta.cond, vec![(0, true)]);
        assert_eq!(conds[0].meta.needs, vec![0]);
        // The measurement is the first `<M>` line; the conditional S names exactly that line.
        let text = items_text(&items);
        let lines: Vec<&str> = text.lines().collect();
        let m_line = lines.iter().position(|l| l.ends_with("<M>")).unwrap();
        // An unconditional `<S>` may precede it (a flushed H), so pick the line with an edge.
        let s_line = lines.iter().find(|l| l.contains("<S>@")).expect("a conditional S line");
        assert!(s_line.ends_with(&format!("<S>@{}", m_line)), "{} vs M at {}", s_line, m_line);
    }

    #[test]
    fn conditional_t_keeps_its_sign_and_condition() {
        let (n, gates) = parse_lines(&[
            "qreg q[2];",
            "creg m[1];",
            "t q[1];",
            "measure q[1] -> m[0];",
            "if(m==0) tdg q[0];",
        ]);
        let items = transpile_opts(n, &gates, 0, false, ResetMode::Frame);
        let t: Vec<&TransPauli> = items
            .iter()
            .filter_map(|i| if let TransItem::Pauli(p) = i { Some(p) } else { None })
            .filter(|p| p.label == "T" && !p.meta.cond.is_empty())
            .collect();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].sign, Sign::Minus, "tdg is a negative rotation");
        assert_eq!(t[0].meta.cond, vec![(0, false)], "`m==0` is a condition on the outcome being 0");
    }

    #[test]
    fn pauli_on_two_bits_is_a_derived_frame_term_not_a_line() {
        let base = &["qreg q[2];", "creg a[1];", "creg b[1];", "t q[0];", "measure q[0] -> a[0];",
                     "t q[1];", "measure q[1] -> b[0];"];
        let mut with = base.to_vec();
        with.push("if(a==1 && b==0) x q[0];");
        with.push("t q[0];");
        let mut without = base.to_vec();
        without.push("t q[0];");
        let (n, g1) = parse_lines(&with);
        let (_, g0) = parse_lines(&without);
        let a = transpile_opts(n, &g1, 0, false, ResetMode::Frame);
        let b = transpile_opts(n, &g0, 0, false, ResetMode::Frame);
        let lines = |v: &[TransItem]| v.iter().filter(|i| !matches!(i, TransItem::Event(_))).count();
        assert_eq!(lines(&a), lines(&b), "a conditional Pauli adds no scheduled product");
        let derived: Vec<usize> = a
            .iter()
            .filter_map(|i| if let TransItem::Event(Event::Derived { id, .. }) = i { Some(*id) } else { None })
            .collect();
        assert_eq!(derived.len(), 1);
        let used = a.iter().any(|i| matches!(i, TransItem::Pauli(p)
            if p.meta.frame_deps.contains(&derived[0])));
        assert!(used, "a later product's sign depends on the derived outcome");
    }

    #[test]
    fn conditional_gate_pays_the_owed_frame_first() {
        let (n, gates) = parse_lines(&[
            "qreg q[1];",
            "creg m[1];",
            "h q[0];",
            "t q[0];",
            "measure q[0] -> m[0];",
            "reset q[0];",
            "if(m==1) s q[0];",
            "t q[0];",
        ]);
        let items = transpile_opts(n, &gates, 0, false, ResetMode::Frame);
        // The reset owes q0 an X (its measurement is outcome 1); the conditional S would turn
        // that into a Y only if it runs, so the owed X is paid physically before it.
        let fix = items.iter().find_map(|i| match i {
            TransItem::Event(Event::Fix { qubit, x_deps, .. }) => Some((*qubit, x_deps.clone())),
            _ => None,
        });
        assert_eq!(fix, Some((0, vec![1])));
        let text = items_text(&items);
        let lines: Vec<&str> = text.lines().collect();
        let s_line = lines.iter().find(|l| l.contains("<S>@")).expect("a conditional S line");
        // needs both the condition's measurement (line of outcome 0) and the fix's (outcome 1)
        assert!(s_line.contains('@') && s_line.split('@').nth(1).unwrap().contains(','), "{}", s_line);
    }

    #[test]
    fn trailing_deferral_does_not_change_product_count() {
        // Deferring the trailing single-qubit layer changes which Cliffords are emitted, never
        // how many T products or measurements the circuit has.
        let mut gates = vec![QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 }];
        for _ in 0..8 {
            gates.push(QasmGate::T { qubit: 0 });
            gates.push(QasmGate::T { qubit: 1 });
        }
        let paulis =
            |items: &[TransItem]| items.iter().filter(|i| matches!(i, TransItem::Pauli(_))).count();
        assert_eq!(
            paulis(&transpile_full(2, &gates, 1, true)),
            paulis(&transpile_full(2, &gates, 1, false))
        );
    }

    #[test]
    fn split_trailing_1q_preserves_the_clifford() {
        // U = R · E: emitting E and keeping R pending must act exactly as the original queue.
        let queue = vec![
            QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 0 },
            QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: 1 },
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 },
            QasmGate::Clifford1Q { gate: Gate1Q::SX, qubit: 2 },
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 1, target: 2 },
            QasmGate::Clifford1Q { gate: Gate1Q::Sdg, qubit: 1 },
            QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 2 },
        ];
        let (emitted, residual) = split_trailing_1q(&queue, 3);
        let mut recombined = emitted.clone();
        recombined.extend(residual.iter().cloned());
        assert!(
            same_up_to_sign(&tableau_of(&recombined, 3), &tableau_of(&queue, 3), 3),
            "split must preserve the accumulated Clifford"
        );
    }

    #[test]
    fn split_trailing_1q_residual_is_local() {
        // The residual must be a tensor product of single-qubit Cliffords: at most one wire each,
        // no 2Q gates. That is what keeps later products weight 1.
        let queue = vec![
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 },
            QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: 0 },
            QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 1 },
            QasmGate::Clifford1Q { gate: Gate1Q::SX, qubit: 2 },
        ];
        let (_, residual) = split_trailing_1q(&queue, 3);
        assert!(!residual.is_empty());
        assert!(residual.iter().all(|g| matches!(g, QasmGate::Clifford1Q { .. })));
    }

    #[test]
    fn local_residual_keeps_products_weight_one() {
        // Every product conjugated through a purely single-qubit tableau stays weight 1.
        let queue = vec![
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 },
            QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 0 },
            QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: 1 },
            QasmGate::Clifford1Q { gate: Gate1Q::SX, qubit: 2 },
        ];
        let (_, residual) = split_trailing_1q(&queue, 3);
        let tab = tableau_of(&residual, 3);
        for q in 0..3 {
            assert_eq!(
                tab.conjugate(&make_z_pauli(3, q, false)).weight(),
                1,
                "a local residual must not spread support on qubit {}",
                q
            );
        }
    }
}
