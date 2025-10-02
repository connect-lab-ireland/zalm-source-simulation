# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# File: zalm_source.py
# Jerry Horgan, 2025
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from qsi.qsi import QSI
from qsi.helpers import numpy_to_json
import time
import uuid
import netsquid as ns
from netsquid import sim_run, sim_reset
from netsquid.qubits import QFormalism
import config
import numbers
from system_setup import network_setup, protocols_setup

# --- Global state for our service ---
qsi = QSI()
# All configurable parameters should be managed via the config module.

# --- Define the API handlers ---
@qsi.on_message("state_init")
def state_init(msg):
    """
    This ZALM Source implementation has no initial state. It creates FlyingQubits (state) on demand.
    """
    return {
        "msg_type": "state_init_response",
        "states": [],
        "state_ids": []
    }


@qsi.on_message("param_query")
def param_query(msg):
    """
    Dynamically generates the parameter query response by introspecting
    the `config.py` module.

    It finds all global variables in UPPERCASE and reports their names
    and inferred data types.

    ports numbers as 'number' and all other types (string, boolean, list)
    as 'string', to conform with the QSI schema.
    """
    params_dict = {}

    try:
        # Get all names defined in the config module
        for name in dir(config):
            # Convention: Only expose variables in all uppercase
            if name.isupper():
                value = getattr(config, name)

                # Booleans and lists will be treated as strings.
                if isinstance(value, numbers.Number):
                    param_type = "number"
                else:
                    param_type = "string"

                params_dict[name] = param_type
    except (ValueError, TypeError) as e:
                print(f"WARNING: Could not get parameters. Invalid value. Error: {e}")

    return {
        "msg_type": "param_query_response",
        "params": params_dict
    }


@qsi.on_message("param_set")
def param_set(msg):
    """
    Dynamically sets parameters in the `config.py` module based on the
    incoming message.
    """
    params_to_set = msg.get("params", {})

    for key, value_dict in params_to_set.items():
        # Check if the parameter exists in our config to avoid setting arbitrary variables
        if hasattr(config, key) and key.isupper():
            new_value = value_dict["value"]

            print(f"Value BEFORE change {getattr(config, key)}")
            setattr(config, key, new_value)
            print(f"INFO: Config parameter '{key}' set to '{new_value}'.")
            print(f"Value AFTER change {getattr(config, key)}")

    #if "SIM_MODE" in params_to_set:
    #    config.apply_config_mode()
    #    print("Changed parameters")

    return {"msg_type": "param_set_response"}

@qsi.on_message("channel_query")
def channel_query(msg):
    """
    Custom message to trigger a single simulation run.
    """

    ns.sim_reset()
    network = network_setup()
    protocols = protocols_setup(network)
    controller = protocols[0]

    for p in protocols:
        p.start()
    ns.sim_run()

    result = controller.result

    return {"msg_type": "channel_query_response", "message": str(result), 'error': 0, 'operation_time': 1, 'retrigger': False,}

@qsi.on_message("result_query")
def result_query(msg):
    """
    Custom message to ask for the result of a run.
    """
    run_id = msg.get("run_id")
    result = results_store.get(run_id, {"status": "PENDING"})
    return {"msg_type": "result_query_response", "run_id": run_id, "result": result}

@qsi.on_message("terminate")
def terminate(msg):
    qsi.terminate()

# --- Start the service ---
def main():
    ns.set_qstate_formalism(QFormalism.DM)
    print("INFO: Starting ZALM Source QSI Service...")
    qsi.run()
    # Keep the main thread alive for the background server
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("INFO: Shutting down.")
        qsi.terminate()

if __name__ == '__main__':
    main()
