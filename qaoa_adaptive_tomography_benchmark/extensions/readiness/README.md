# Write-up readiness extension

This is an additive study. The original `src/`, protocol and benchmark results remain the scientific baseline. The extension tests a specific mechanism rather than searching for a favourable quantum-advantage claim.

## Study and provenance

`protocol.json` was locally timestamped and hashed before extension outcomes were generated. This is not external preregistration. The two historical datasets together supply 36 development channels; the previous 30-channel evaluation set is explicitly reused as **development only**. `selection.json` locks the intervention and fresh sample size before sampling new channels.

The development factorial crosses the existing swept-basis initialization with a uniform fixed-Hamming-weight (Dicke) initial state, mean-energy with lower-tail CVaR at alpha=0.2, and finite-shot with exact-expectation training. Depth, coefficient scaling, mixer order, optimizer, evaluation cap and final sampling are held fixed. Exact training is an offline diagnostic. Dicke state preparation is represented ideally; physical gate costs are not compiled and there is no hardware advantage claim.

One finite-shot variant is selected using expected best-of-256 normalized regret on development channels. Fresh confirmation uses the **actually sampled** best-of-256 regret, with seeds, checkpoints and two acquisition repetitions averaged within each physical channel. The fresh study also compares four complete adaptive policies at 4,056 channel applications per trajectory. RMSE, normalized Choi trace distance and all downstream comparisons are descriptive secondary endpoints.

Repetitions vary acquisition, fitting-initialization and solver streams while retaining common validation probes for a physical channel. Their within-channel variation is overall run variability, not an isolated measurement-noise variance estimate.

## Run

Use the parent `requirements.txt`. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before running. Commands are shown from the benchmark root.

```bash
python extensions/readiness/validate.py
python extensions/readiness/run.py freeze
python extensions/readiness/run.py development --workers 6
python extensions/readiness/run.py landscapes --workers 6
python extensions/readiness/run.py choose
python extensions/readiness/run.py scaffold --workers 6
python extensions/readiness/run.py prepare
python extensions/readiness/run.py confirmation --workers 6
python extensions/readiness/run.py adaptive --workers 6
python extensions/readiness/analyse.py --historical-only
python extensions/readiness/analyse.py
python extensions/readiness/verify.py
python extensions/readiness/report.py --output-dir readiness_report
```

Completed solver problems and trajectory rounds are resumable. Do not execute two workers on the same trajectory. Acquisition branches can run after their shared pilot is complete; this changes scheduling only, not seeds or scientific choices. The supplied historical results must remain available for the development and retrospective stages.

For a clean replication, copy the repository and remove only the generated extension paths listed below in that separate copy: `protocol.json`, `selection.json`, `validation.json`, `verification.json`, `release_manifest.json`, `data/`, `results/`, `analysis/` and `figures/`. Do not delete the original benchmark data. Then run the sequence above. Source/protocol changes require a separately declared new study, not reuse of completed caches.

## Outputs

| Path | Use |
|---|---|
| `protocol.json`, `selection.json` | Fixed design, intervention choice and sample-size calculation |
| `validation.json` | Qiskit circuit equivalence, original-baseline reproduction, CVaR and sampling checks |
| `data/channels.json` | Fresh physical parameters |
| `data/trajectories/` | Counts, all fitting candidates, selected models, solver records and offline checks |
| `results/development/`, `results/confirmation/` | Complete solver traces and actual final samples |
| `results/landscapes/` | Bounded parameter grids, not global optimization certificates |
| `analysis/` | Channel-level means, paired contrasts and explanatory diagnostics |
| `THEORY_AND_CLAIMS.md` | Mathematical derivations, reference map and permitted claims |
| `THESIS_WRITEUP_PLAN.md` | Research questions, contribution statement and chapter/evidence map |
| `RESULTS_FOR_WRITEUP.md` | Draft abstract and results paragraphs tied to the completed analysis |
| `LITERATURE_NOTES.md` | Targeted research positioning and literature-review boundaries |
| `figures/`, `tables/` | Standalone PNG/SVG figures and native LaTeX tables |
| `references.bib` | Verified foundational literature |
| `verification.json`, `release_manifest.json` | Independent saved-result checks and release hashes |

The reachability calculation, uniform-distribution reference and historical utility/error association are labelled **post hoc**. They explain results and cannot replace the frozen primary comparison. The archived Choi-distance data already existed; this extension brings them into the analysis rather than claiming to invent a new metric.
