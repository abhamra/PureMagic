OPENQASM 2.0;
include "qelib1.inc";
// Feed-forward on the critical path: each measurement immediately gates a Clifford on the other
// qubit, whose own T gates then feed the next measurement. Reaction latency is paid at every hop.
qreg q[2];
creg a[1];
creg b[1];
creg c[1];
creg d[1];
h q[0];
t q[0];
h q[0];
t q[0];
measure q[0] -> a[0];
if(a==1) h q[1];
t q[1];
h q[1];
t q[1];
measure q[1] -> b[0];
if(b==1) h q[0];
t q[0];
h q[0];
t q[0];
measure q[0] -> c[0];
if(c==1) h q[1];
t q[1];
h q[1];
t q[1];
measure q[1] -> d[0];
