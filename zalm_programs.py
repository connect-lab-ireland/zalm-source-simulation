# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# File: programs.py
# Jerry Horgan, 2025
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
This module contains the NetSquid QuantumProgram definitions used in the simulation.
These represent the low-level instruction sequences executed by QuantumProcessors.
"""
from netsquid.qubits import operators as ops
from netsquid.components.qprogram import QuantumProgram
from netsquid.components import instructions as instr
from custom_qubits import create_flying_qubits


class CreatePsiPlusPairProgram(QuantumProgram):
    """Quantum program to create a |Ψ+> Bell pair. |01> + |10>"""
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        # Apply X to q2 to put it in |1> state initially
        self.apply(instr.INSTR_INIT, [q1, q2])
        self.apply(instr.INSTR_X, q2)
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        yield self.run()

class BSMProgram(QuantumProgram):
    """Quantum program to perform a Bell State Measurement."""
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_H, q1)
        self.apply(instr.INSTR_MEASURE, q1, output_key="M1")
        self.apply(instr.INSTR_MEASURE, q2, output_key="M2")
        yield self.run()

class CorrectionProgram(QuantumProgram):
    """Applies a specific Pauli correction gate to a single qubit."""
    default_num_qubits = 1

    def program(self, gate_str: str):
        q1, = self.get_qubit_indices(1)
        if gate_str == "I": pass
        elif gate_str == "X": self.apply(instr.INSTR_X, q1)
        elif gate_str == "Z": self.apply(instr.INSTR_Z, q1)
        elif gate_str == "Y":
            self.apply(instr.INSTR_Z, q1)
            self.apply(instr.INSTR_X, q1)
        yield self.run()

class PostProcessProgram(QuantumProgram):
    default_num_qubits = 1

    def program(self, gate):
        q1 = self.get_qubit_indices(1)

        if gate == "None":
            #do nothing
            pass
        elif gate == "Z":
            # Apply a Z gate
            self.apply(instr.INSTR_Z, q1)
        elif gate == "X":
            # Apply a X gate
            self.apply(instr.INSTR_X, q1)

        yield self.run()


class BellMeasurementProgram(QuantumProgram):
    """Program to perform a Bell measurement on two qubits.

    Measurement results are stored in output keys "M1" and "M2"

    """
    default_num_qubits = 2

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_MEASURE, q2, observable=ops.Z, output_key="M1", discard=True)
        self.apply(instr.INSTR_MEASURE, q1, observable=ops.Z, output_key="M2", discard=True)
        yield self.run(parallel=False)

class CNOTQubits(QuantumProgram):
    default_num_qubits = -1

    def program(self):
        q1, q2 = self.get_qubit_indices(2)
        self.apply(instr.INSTR_CNOT, [q1, q2])
        self.apply(instr.INSTR_H, q1)

        yield self.run(parallel=False)

class SPDCInitQubits(QuantumProgram):
    """
    A quantum program to create and initialize two FlyingQubits
    in a quantum processor's memory.
    """
    default_num_qubits = 2

    def program(self):
        s_index, i_index = self.get_qubit_indices(2)
        # Create 2 flying qubits
        signal_qubit, idler_qubit = create_flying_qubits(2)
        self.apply(instr.INSTR_INIT, s_index, qubit=signal_qubit)
        self.apply(instr.INSTR_INIT, i_index, qubit=idler_qubit)
        yield self.run()

class SPDCProgram(QuantumProgram):
    default_num_qubits = 2 #4

    def program(self, vert_index, horz_index):
        # Instructions applied this way, to assign 1 to the Vertical Polarised Qubit]
        vert_index = vert_index
        horz_index = horz_index
        #log.debug(f"Vertical index : {vert_index} , Horz index : {horz_index}")
        self.apply(instr.INSTR_X, vert_index)  # Flip the state of the vertically polarized qubit

        self.apply(instr.INSTR_H, horz_index)  # Create a superposition
        self.apply(instr.INSTR_CNOT, [horz_index, vert_index])  # Entangle the qubits

        yield self.run(parallel=False)

class PhaseFlipQubits(QuantumProgram):
    default_num_qubits = 2

    def program(self, flipQubits):
        for qubit in flipQubits:
            self.apply(instr.INSTR_Z, qubit)

        yield self.run()
