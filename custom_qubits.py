# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# File: custom_qubits.py
# Jerry Horgan, 2025
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from netsquid.qubits.qubit import Qubit
from netsquid.qubits.qubitapi import _system_name_counter, get_qstate_formalism
from netsquid.qubits.qstate import QState

class FlyingQubit(Qubit):
    """
    A custom qubit class that extends NetSquid's default Qubit to include
    physical properties relevant for photonic network simulations, such as
    frequency and polarization.

    These properties are stored as simple Python attributes and do not directly
    affect the quantum state itself, but are used by component protocols
    to model physical interactions.
    """

    # Define a class-level set of valid polarization states for efficient checking.
    _VALID_POLARISATIONS = {"H", "V", "A", "D"}
    
    def __init__(self, name, **kwargs):
        """
        Initializes a FlyingQubit.

        Args:
            name (str): The name of the qubit.
            **kwargs: Keyword arguments passed to the parent Qubit class.
        """
        super().__init__(name, **kwargs)
        # Initialize physical properties to None by default.
        self._frequency = None      # in Hz
        self._polarisation = None   # Values should be "H", "V", "A", "D"
      
    @property
    def frequency(self):
        """
        The center frequency of the photon's wave packet, in Hz.
        
        This is a classical property used to calculate interference visibility.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value: float):
        """
        Sets the center frequency of the photon.

        Args:
            value (float): The frequency in Hertz.
        """
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Frequency must be a non-negative number.")
        self._frequency = value
    
    @property
    def polarisation(self) -> str:
        """
        A classical label for the intended polarization state of the photon.
        Valid states are 'H' (Horizontal), 'V' (Vertical), 'A' (Anti-diagonal),
        and 'D' (Diagonal).
        """
        return self._polarisation

    @polarisation.setter
    def polarisation(self, value: str):
        """
        Sets the polarization label, ensuring it is one of the valid states.
        """
        # Check if the provided value is in our set of valid states.
        if value not in self._VALID_POLARISATIONS:
            # If not, raise a descriptive ValueError to stop execution.
            raise ValueError(
                f"Invalid polarisation state '{value}'. "
                f"Must be one of {self._VALID_POLARISATIONS}."
            )
        # If it is valid, set the internal attribute.
        self._polarisation = value

def create_flying_qubits(num_qubits, system_name=None, no_state=False):
    """
    Creates a system of FlyingQubits.

    This is a custom factory function that mirrors netsquid.qubits.create_qubits
    but instantiates the `FlyingQubit` class instead of the standard `Qubit` class.

    Args:
        num_qubits (int): Number of qubits to create.
        system_name (str, optional): A name for the qubit system.
        no_state (bool, optional): If True, do not initialize the qubits' quantum state.

    Returns:
        list of :obj:`FlyingQubit`: The created flying qubits.
    """
    global _system_name_counter
    if system_name is None:
        # Use the same global counter as NetSquid's default for consistency
        system_name = f"FQS#{_system_name_counter}-" # FQS for Flying Qubit System
        _system_name_counter += 1
    
    # The key change is here: instantiate `FlyingQubit`
    qubits = [FlyingQubit(system_name + str(i)) for i in range(num_qubits)]
    
    if not no_state:
        # Initialize quantum states in the standard |0> state
        for i in range(num_qubits):
            qrepr_class = get_qstate_formalism()
            qrepr = qrepr_class(num_qubits=1)
            QState(qubits[i:i + 1], qrepr)
            
    return qubits