OPENQASM 2.0;
include "qelib1.inc";
// Two data qubits (0,1) and one reusable ancilla (2), in the shape dcsynth emits:
// compute onto the ancilla, measure it, fix the data with Paulis, reset, reuse.
qreg q[3];
creg m0[1];
creg m1[1];
h q[0];
t q[0];
cx q[0], q[2];
h q[2];
t q[2];
cx q[2], q[1];
measure q[2] -> m0[0];
if(m0==1) z q[0];
if(m0==1) z q[1];
reset q[2];
h q[2];
t q[2];
cx q[2], q[0];
h q[2];
measure q[2] -> m1[0];
if(m1==1) x q[1];
reset q[2];
t q[1];
h q[1];
t q[0];
