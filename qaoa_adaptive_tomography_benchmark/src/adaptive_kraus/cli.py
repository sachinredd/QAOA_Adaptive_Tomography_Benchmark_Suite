"""Command line entry points for complete runs and count-only refitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import load_config
from .runner import run_benchmark, write_json
from .experiments import make_pool, Observation
from .learning import fit_channel
from .manifold import random_isometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run paired adaptive-design comparisons")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    refit = sub.add_parser("refit", help="Fit saved/imported counts without a reference channel")
    refit.add_argument("--config", required=True)
    refit.add_argument("--observations", required=True)
    refit.add_argument("--initial", help="Optional NPZ containing a stack array")
    refit.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "run":
        run_benchmark(config, args.output)
    else:
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Use an empty output directory")
        output.mkdir(parents=True, exist_ok=True)
        pool = make_pool(config["channel"]["n_qubits"], config["pool_steps"], config["readout"])
        records = json.loads(Path(args.observations).read_text())
        observations = [Observation(r["experiment_id"], tuple(r["counts"]), r["shots"]) for r in records]
        if args.initial:
            with np.load(args.initial, allow_pickle=False) as saved:
                initial = saved["stack"]
        else:
            initial = random_isometry(2**config["channel"]["n_qubits"], config["rank"], config["seeds"][0]+4001)
        result = fit_channel(pool, observations, initial, **config["fit"])
        np.savez_compressed(output/"model.npz", stack=result.stack)
        write_json(output/"fit.json", result.history)
        write_json(output/"diagnostics.json", result.diagnostics)
        print(json.dumps(result.diagnostics, indent=2))
