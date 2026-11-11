"""Assemble a dcsynth dynamic circuit for the dynamic-circuit tutorial notebook.

Runs in dcsynth's environment (the DynamicCircuitPlay Artifact's .venv), not PureMagic's: the
notebook calls it as a subprocess. Given a circuit and its compiled block library it writes

  OUT/orig.qpy     the input circuit (final measurements removed)
  OUT/dynamic.qpy  dcsynth's dynamic version: dynamic blocks where they lower CNOT depth, on the
                   fewest ancillas that reach that depth, with ancillas no gate touches removed
  OUT/summary.json CNOT depths, qubit counts, which form each block took, and the realized
                   synthesis error of the blocks (half diamond distance; see
                   docs/mcm-changes.md, section 6a)

usage: python dcsynth_assemble.py CIRCUIT.qasm LIBRARY_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator

from dcsynth.assembly import BlockLibrary, Variant, assemble, cnot_depth
from dcsynth.assembly.schedule import fewest_ancillas


def strip_idle_ancillas(qc, n):
    """Drop qubits past the first n that no instruction touches."""
    used = {qc.find_bit(q).index for ins in qc.data for q in ins.qubits}
    idle = [qc.qubits[i] for i in range(n, qc.num_qubits) if i not in used]
    if not idle:
        return qc
    dag = circuit_to_dag(qc)
    dag.remove_qubits(*idle)
    return dag_to_circuit(dag)


def aligned(circ, indices, order):
    """Unitary of `circ` (on global qubits `indices`) with its qubits put in `order`."""
    if list(indices) == list(order):
        return Operator(circ).data
    qc = QuantumCircuit(len(order))
    qc.compose(circ, qubits=[list(indices).index(g) for g in order], inplace=True)
    return Operator(qc).data


def dynamic_block_error(version, v):
    """Bound on half the diamond distance between a dynamic version and target unitary v.

    Kraus operators K_j = <j| B_j P_j U_main |0> on the data qubits; with K_j = c_j V + E_j,
    the error is at most delta + delta^2/2, delta = sqrt(||sum E_j^dag E_j||)."""
    k, a = len(version.data_indices), version.num_ancillas
    D = 2 ** k
    u = Operator(version.main).data
    kraus = []
    for j in range(2 ** a):
        bits = [(j >> (a - 1 - i)) & 1 for i in range(a)]
        idx = sum(b << i for i, b in enumerate(bits))
        br = version.branches[j]
        full = Operator(br).data if br.num_qubits > k else np.kron(np.eye(2 ** a), Operator(br).data)
        proj = np.zeros(D * 2 ** a)
        proj[idx * D:(idx + 1) * D] = 1
        kraus.append((full @ (proj[:, None] * u))[idx * D:(idx + 1) * D, 0:D])
    t = np.array([np.trace(v.conj().T @ kj) / D for kj in kraus])
    c = t / np.linalg.norm(t)
    m = sum((kj - cj * v).conj().T @ (kj - cj * v) for kj, cj in zip(kraus, c))
    delta = float(np.sqrt(np.linalg.norm(m, 2)))
    return delta + delta ** 2 / 2, float(1 - np.sum(np.abs(t) ** 2))


def main():
    circuit, lib_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    original = QuantumCircuit.from_qasm_file(str(circuit))
    original.remove_final_measurements()
    n = original.num_qubits
    library = BlockLibrary(lib_dir)
    budget, _ = fewest_ancillas(library, n, 'perqubit')
    res = assemble(library, n, budget, Variant.DC)
    dynamic = strip_idle_ancillas(res.circuit, n)

    blocks = []
    for i, step in enumerate(res.steps):
        entry = {'block': i, 'form': step.mode}
        if step.mode == 'dc':
            ver = library.dynamic_versions(i)[step.version or 0]
            ori, ori_idx = library.original(i)
            e, cost = dynamic_block_error(ver, aligned(ori, ori_idx, ver.data_indices))
            entry.update(ancillas=ver.num_ancillas, error=e, dcsynth_cost=cost)
        blocks.append(entry)

    for name, qc in (('orig', original), ('dynamic', dynamic)):
        with open(out / f'{name}.qpy', 'wb') as f:
            qpy.dump(qc, f)
    summary = {
        'data_qubits': n,
        'ancilla_budget': budget,
        'orig': {'qubits': original.num_qubits, 'cnot_depth': cnot_depth(original)},
        'dynamic': {'qubits': dynamic.num_qubits, 'cnot_depth': res.cnot_depth,
                    'ancillas_used': dynamic.num_qubits - n,
                    'dynamic_blocks': res.num_dynamic_blocks},
        'blocks': blocks,
        'block_error': float(sum(b.get('error', 0.0) for b in blocks)),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
