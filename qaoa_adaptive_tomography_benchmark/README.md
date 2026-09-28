# QAOA adaptive tomography simulation benchmark

This suite executes a controlled benchmark of QAOA for batch selection in adaptive one-qubit process tomography. It tests reuse of QAOA angles between rounds, candidate shortlist size, circuit depth and optimization budget. It also executes fresh adaptive trajectories using seven policies at the same unknown-channel application budget.

The definitive results are in `analysis/summary.json` and the accompanying Word report. The entire design is frozen in `protocol/PROTOCOL.json`. `protocol/TRAJECTORY_CONFIGURATION.json` records the configuration chosen using only the separate development set. The scripts retain all planned outcomes, including fitting flags.

## Executed design

- Development: six archived matched-cost trajectories, one per physical family, with three frozen learned states each. No historical channel enters the fresh primary analysis.
- Evaluation: thirty fresh physical channels, five in each of six families. One calibrated readout profile and one acquisition repetition per channel.
- Frozen-state problems: select 3 of 8, 10, 12 or 14 experiments, plus a separate select-5-of-14 track. The latter uses a larger application budget and is not an equal-cost comparison with triples.
- QAOA: depths 1 and 2; objective-evaluation caps 20 and 80; cold or preceding-round angle initialization; three paired solver seeds; 128 samples per objective evaluation and 256 final samples. The first 8 and 32 final samples are nested checkpoints within that same final sample stream, not additional runs.
- Classical controls: exact feasible quadratic optimization, exact local information optimization within each shortlist, uniform random search and swap-based local search. Full-pool greedy has a different search domain and is reported separately.
- Counts: 12,960 QAOA optimization executions and 9,720 classical heuristic runs across development and evaluation. The statistical independent units in the primary analysis are thirty physical channels, not these execution counts.
- Adaptive comparison: 210 trajectories, comprising seven policies on the same thirty fresh channels. Each uses 4056 unknown-channel applications and four-candidate channel fitting at every round. The common pilot is simulated and fitted once per channel and reused across its seven branches.

## Simulator and cost interpretation

`src/benchmark_solver.py` simulates the ideal QAOA circuit exactly within the fixed-Hamming-weight subspace. It implements the archived cost gates, ordered chain XY mixer, and two fixed initial XY sweeps. Its probabilities were checked against Qiskit's full circuit statevector for all five problem sizes and both depths. Finite objective and final measurements are sampled from those probabilities.

Enumerating amplitudes and Hamiltonian energies is part of classical simulation. The optimizer never receives the exact minimizing bitstring. Its chosen batch must have appeared in the final samples; no exact replacement or classical repair is performed. Nevertheless, this implementation is a classical simulation, and its elapsed time cannot demonstrate quantum speedup.

Tomography observations use the archived Qiskit Aer density-matrix oracle. The Kraus fitter and the observation model are unchanged. Readout assignment probabilities are known and fixed at 0.03 and 0.06. There is no QAOA gate noise or IBM hardware execution in this benchmark.

Report characterization shots, unknown-channel applications, QAOA objective shots, final shots and classical work separately. Equal counts of classical candidate evaluations and quantum measurements do not imply equal time, energy or economic cost. The complete 256-shot final stream is charged, including when a figure also shows results after its first 8 or 32 samples.

## Reproduce the results

Install the versions in `requirements.txt` in an appropriate Python environment. The recorded research runtime is also in the frozen protocol. For a clean replication, create a new directory; this command refuses to overwrite one:

```bash
python scripts/new_run.py --output /absolute/path/qaoa_replication
```

From the new directory, use the following sequence. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before running the worker commands to avoid oversubscription.

```bash
python tests/validate_solver.py
python scripts/prepare_problems.py --split development
python scripts/run_frozen.py --split development --workers 6
python scripts/choose_configuration.py
python scripts/run_trajectories.py --stage scaffold --workers 6
python scripts/prepare_problems.py --split evaluation
python scripts/run_frozen.py --split evaluation --workers 6
python scripts/run_trajectories.py --stage remaining --workers 6
python scripts/analyse.py
python scripts/distribution_diagnostics.py
python scripts/verify_results.py
python scripts/plot_results.py
python scripts/build_report.py
```

Worker counts affect scheduling, not scientific seeds. Completed trajectory rounds have checksums and can be resumed. A failed uncommitted round is repeated from its preceding checkpoint. Do not run two trajectory workers on the same task at once. Changing source code or the frozen scientific protocol requires a new experiment; the runner checks those hashes. The supplied `prepare_inputs.py` documents extraction from the earlier suite archive, but is unnecessary for replication because its compact source inputs are included here.

## Analysis and independent units

The primary endpoint is transferred minus cold expected normalized quadratic energy regret for select-3-of-14, depth 1, and cap 20. Average the three solver seeds and the two post-pilot checkpoint outcomes within each channel. The initial checkpoint is excluded because both initializations are identical there. Use a 95% equal-family stratified t interval with Welch-Satterthwaite degrees of freedom on the thirty channel means.

The energy regret divides the energy gap above the exact feasible minimum by the feasible energy range. This metric is distinct from prediction RMSE and from loss of local information gain. Exact distribution expectations and exact optima are offline diagnostics, not information available to the finite-shot optimizer.

Other contrasts, sampling success rates and end-to-end RMSE intervals are descriptive and are not adjusted for multiple comparisons. The `posthoc_distribution_*.csv` files explicitly contain mechanism analyses added after the frozen-state results were observed; they are not primary tests. Intervals crossing zero do not establish equivalence.

## File map

| Location | Purpose |
|---|---|
| `protocol/` | Scientific protocol, source hashes, prior schedule, input provenance and development selection |
| `data/development/` | Eighteen compact archived learned checkpoints |
| `data/evaluation_channels.json` | Fresh channel parameters and paired acquisition seeds |
| `data/problems/` | Exact frozen quadratic problems and separate audit reference arrays |
| `data/trajectories/` | Counts, learned models, fitting candidates, solver records and diagnostics |
| `results/frozen/` | Every QAOA optimization, angles, sampled energy trace and final samples |
| `results/classical/` | Every heuristic selection and resource record |
| `results/verification.json` | Independent numerical and integrity checks |
| `analysis/` | Flat CSV tables, primary analysis and descriptive summaries |
| `figures/` | Eight report figures in PNG and editable SVG formats |
| `docs/` | Final Word report and its source text |
| `src/adaptive_kraus/` | Unmodified archived production implementation |
| `src/benchmark_*.py` | New circuit, solver and benchmark execution modules |
| `scripts/` | Planning, execution, analysis, verification, plotting and report generation |

Physical channel truth is used only by the synthetic observation oracle and subsequent evaluation. It is never passed to the acquisition selector or used to choose a channel fit. Full candidate histories and flagged fits remain in the delivered package.
