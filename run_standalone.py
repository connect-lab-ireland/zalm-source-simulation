# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# File: run_standalone.py
# Jerry Horgan, 2026
#
# Standalone (NetSquid-only) driver for the ZALM source simulation. No QSI.
#
# It calls the same network_setup() / protocols_setup() that zalm_source.py
# calls for each QSI "channel_query", and computes the same |Psi-> fidelity
# that qsi_source_fidelity.py computes (that driver runs at 0 km of fibre,
# where the QSI fibre channel is the identity).
#
# Memory: NetSquid 1.1.8 never frees QuantumProcessor / QuantumChannel objects,
# so every run leaks ~0.9 MB and a single process dies after ~2k runs. This
# driver therefore runs in chunks, each in a fresh worker process
# (--chunk runs per process), optionally in parallel (--workers).
#
# Examples:
#   python run_standalone.py --runs 10000 --workers 8
#   python run_standalone.py --runs 10000 --set DETECTOR_EFFICIENCY=0.5 --out det_eff_0.5.json
#   python run_standalone.py --runs 10000 --set TEMPORAL_JITTER_STDEV_PS=10 --no-processor-delays
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import argparse
import ast
import json
import multiprocessing as mp
import sys
import time
import warnings

import config  # safe: utils/system_setup are only imported inside workers


def parse_overrides(pairs):
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key.isupper() or not hasattr(config, key):
            sys.exit(f"Unknown config parameter: {key!r}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw  # plain strings, e.g. DETECTOR_TYPE=PNR
        out[key] = value
    return out


def run_chunk(task):
    """Run `n` simulations in this (fresh) process and return raw tallies."""
    n, seed, overrides, no_delays = task
    warnings.simplefilter("ignore")

    # Overrides must be applied before utils is imported: it builds the DWDM
    # grid from config.ENABLED_BANDS / GRID_GRANULARITY_GHZ at import time.
    for key, value in overrides.items():
        setattr(config, key, value)

    import random
    import numpy as np
    import netsquid as ns
    from netsquid.qubits import QFormalism
    import utils
    from system_setup import network_setup, protocols_setup, cleanup

    if no_delays:
        # Zero duration for every processor instruction (INIT/H/X/Z/CNOT/MEASURE)
        # without editing utils.create_processor.
        _PI = utils.PhysicalInstruction
        utils.PhysicalInstruction = lambda *a, **kw: _PI(*a, **{**kw, "duration": 0})

    if seed is not None:
        ns.set_random_state(seed=seed)
        np.random.seed(seed % 2**32)
        random.seed(seed)

    ns.set_qstate_formalism(QFormalism.DM)
    psi_minus = (1 / np.sqrt(2)) * np.array([0, 1, -1, 0]).reshape(-1, 1)

    fidelities, errors = [], {}
    for _ in range(n):
        # Same sequence as zalm_source.channel_query()
        ns.sim_reset()
        network = network_setup()
        protocols = protocols_setup(network)
        for p in protocols:
            p.start()
        ns.sim_run()
        result = protocols[0].result
        cleanup(network, protocols)

        if result is not None and result.get("status") == "SUCCESS":
            rho = np.array([[complex(c[0], c[1]) for c in row] for row in result["final_dm"]])
            fidelities.append(float(np.real((psi_minus.conj().T @ rho @ psi_minus)[0, 0])))
        else:
            kind = "no result" if result is None else result.get("error_type", result.get("status"))
            errors[kind] = errors.get(kind, 0) + 1
    return n, fidelities, errors


def main():
    ap = argparse.ArgumentParser(description="Run the ZALM source simulation with NetSquid only (no QSI).")
    ap.add_argument("--runs", type=int, default=1000, help="number of source triggers (default 1000)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a config.py parameter, e.g. --set DETECTOR_TYPE=PNR (repeatable)")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes (default 1)")
    ap.add_argument("--chunk", type=int, default=500,
                    help="runs per worker process before it is recycled (default 500, keeps RSS < ~600 MB)")
    ap.add_argument("--seed", type=int, default=None, help="base seed; chunk i uses seed+i")
    ap.add_argument("--no-processor-delays", action="store_true",
                    help="give all quantum-processor instructions zero duration (see README)")
    ap.add_argument("--label", default=None, help="value stored as configuration_name in the JSON")
    ap.add_argument("--out", default=None, help="write the JSON summary here")
    args = ap.parse_args()

    overrides = parse_overrides(args.set)
    if "SIM_MODE" in overrides:
        sys.exit("SIM_MODE is applied when config.py is imported; edit it in config.py instead.")

    sizes = [args.chunk] * (args.runs // args.chunk) + ([args.runs % args.chunk] if args.runs % args.chunk else [])
    tasks = [(n, None if args.seed is None else args.seed + i, overrides, args.no_processor_delays)
             for i, n in enumerate(sizes)]

    fidelities, errors, done = [], {}, 0
    t0 = time.time()
    ctx = mp.get_context("spawn")  # clean interpreter per worker on every OS
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for n, fids, errs in pool.imap_unordered(run_chunk, tasks):
            done += n
            fidelities += fids
            for k, v in errs.items():
                errors[k] = errors.get(k, 0) + v
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {done}/{args.runs} runs, "
                  f"{len(fidelities)} heralded, {time.time() - t0:.0f}s", flush=True)

    import numpy as np
    successes = len(fidelities)
    summary = {
        "configuration_name": args.label if args.label is not None else (overrides or "default"),
        "num_trials": args.runs,
        "overrides": overrides,
        "no_processor_delays": args.no_processor_delays,
        "seed": args.seed,
        "metrics": {
            "fidelity": {
                "mean": float(np.mean(fidelities)) if fidelities else None,
                "std_dev": float(np.std(fidelities)) if fidelities else None,
            },
            # Same quantity the QSI driver stored as "avg_photon_number"
            # (heralded successes / triggers); kept so the plot_*.py scripts work.
            "avg_photon_number": successes / args.runs,
        },
        "failure_breakdown": errors,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
