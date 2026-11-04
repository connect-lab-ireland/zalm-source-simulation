"""
Fiber example
----------------------------
It requires length and refractive index parameters.
It returns a Kraus operators which attenuate the state of the given mode,
according to the given refractive index and length.
"""
from qsi.qsi import QSI
from qsi.helpers import numpy_to_json, pretty_print_dict
from qsi.state import State, StateProp
import time
import numpy as np
import uuid
from scipy.linalg import expm

# Initiate QSI object instance
qsi = QSI()

# Store the parameters
LENGTH = None
REFRACTIVE_INDEX = None
C0 = 299999999 # Speed of light in vacuum
# Added
LOSS_DB_PER_KM = 0.2  # Standard SMF-28 fiber loss at 1550 nm

@qsi.on_message("state_init")
def state_init(msg):
    """
    Fiber doesn't hold any internal state, thus it responds with
    empty reponse.
    """
    return {
        "msg_type": "state_init_response",
        "states": [],
        "state_ids": []
    }

@qsi.on_message("param_query")
def param_query(msg):
    """
    Fiber expects two number (float) parameters: length and n (refractive index)
    """
    return {
        "msg_type": "param_query_response",
        "params" : {
            "length": "number",
            "n": "number"
        }
    }

@qsi.on_message("param_set")
def param_set(msg):
    """
    Single photon source doesn't require any parameters
    """
    global LENGTH
    global REFRACTIVE_INDEX
    params = msg["params"]
    if "length" in params:
        LENGTH = float(params["length"].get("value"))
    if "n" in params:
        REFRACTIVE_INDEX = float(params["n"].get("value"))
    return {
        "msg_type": "param_set_response",
    }

@qsi.on_message("channel_query")
def channel_query(msg):
    global LENGTH
    global REFRACTIVE_INDEX
    global C0
    #Added
    global LOSS_DB_PER_KM
    state = State.from_message(msg)
    uuid = msg["ports"]["input"]
    prop = state.get_props(uuid)

    # Do not compute channel, if mode with wrong properties is given
    #if prop.wavelength != 1550:
    #    return {
    #        "msg_type": "channel_query_response",
    #        "message": f"This component only interacts with 1550 nm modes, received {prop.wavelength}, {type(prop.wavelength)}"
    #    }

    # If length is not given
    if LENGTH is None or REFRACTIVE_INDEX is None:
        return {
            "msg_type": "channel_query_response",
            "message": f"This component requires 'length' [m] and 'n' (refractive index) parameters to be set."
        }

    # Changed eta calculation
    #eta = 10**((-20*0.01)/(LENGTH))
    #LENGTH is in KM
    eta = np.exp(-((LOSS_DB_PER_KM *np.log(10))/10) * LENGTH)
    phi = (2*np.pi*REFRACTIVE_INDEX*LENGTH*1000)/(prop.wavelength*10**(-9))
    n_max = prop.truncation-1 # Maximum number of photons in the system
    N = np.diag(np.arange(prop.truncation))
    U_phi = expm(-1j * phi * N )
    U_phi_half = expm(-1j * phi * N / 2)

    a = np.zeros((prop.truncation, prop.truncation))
    for n in range(prop.truncation-1):
        a[n, n+1] = 1 # original values
        #a[n, n+1] = np.sqrt(n+1) # new extinction operator

    kraus_operators = []
    for k in range(prop.truncation):
        if k == 0:
            # No photon loss case
            K = np.sqrt(eta**n_max) * U_phi
            kraus_operators.append(K)
        else:
            # Photon loss case
            factor = np.sqrt((1-eta)**k * eta**(n_max - k))
            a_k = np.linalg.matrix_power(a, k)
            K_k = factor * U_phi_half @ a_k @ U_phi_half
            kraus_operators.append(K_k)

    operating_time = LENGTH / (REFRACTIVE_INDEX * C0)

    return {
        "msg_type": "channel_query_response",
        "kraus_operators": [numpy_to_json(x) for x in kraus_operators],
        "kraus_state_indices": [uuid],
        "error": 0,
        "retrigger": False,
        "retrigger_time": 0,
        "operating_time": operating_time
    }

@qsi.on_message("terminate")
def terminate(msg):
    qsi.terminate()

qsi.run()
time.sleep(1)
