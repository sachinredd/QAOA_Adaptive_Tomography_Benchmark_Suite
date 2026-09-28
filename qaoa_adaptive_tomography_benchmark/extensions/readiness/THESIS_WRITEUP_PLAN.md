# Thesis writing plan

Working title: **Optimizer Quality and Reconstruction Accuracy in QAOA-Assisted Adaptive Process Tomography**.

## Central research question

When a shallow constrained quantum optimizer selects experiments for adaptive one-qubit process tomography, how do optimization quality, local information gain and final reconstruction accuracy relate, and which circuit/training choices explain the observed behaviour?

This formulation remains meaningful if quantum methods fail to outperform classical controls. It avoids promising a speedup or extending conclusions to arbitrary quantum processes.

## Research questions and evidence

| Question | Evidence | Status of inference |
|---|---|---|
| Does previous-round angle reuse improve the optimizer? | Original primary transfer contrast; configuration-specific secondary contrasts | Original frozen primary remains authoritative; null interval is not equivalence |
| Where does measurement-design quality get lost? | Full-pool versus shortlist optimum, exact surrogate and sampled solution | Exact local decomposition on bounded enumerable problems |
| Does optimization quality propagate to reconstruction? | Original and fresh paired trajectories, RMSE and normalized Choi trace distance | Downstream comparisons are descriptive; independent unit is a physical channel |
| Do initialization, objective statistic or shot noise explain solver performance? | 2 × 2 × 2 development factorial; exact-training and bounded-landscape diagnostics | Controlled development evidence; no test-set tuning |
| Does the selected intervention reproduce on fresh channels? | Frozen sampled-regret primary, with repetitions clustered within channel | Fresh confirmation with a declared practical margin |
| Why is the classical baseline strong? | Local monotone-submodular derivation and exact full-pool audits | Known greedy theory applied to this objective, not a new theorem |

## Contribution statement

This thesis develops and audits a computational framework that separates constrained quantum-optimizer performance, local experimental-design quality and final quantum-channel reconstruction. It evaluates sequential angle reuse against demanding classical controls, identifies initialization and distributional limitations through controlled experiments, and tests a development-selected intervention on untouched physical channels under a fixed channel-application budget.

The novel work to defend is the research integration, controlled comparisons, explanatory evidence and reproducible benchmark. Kraus-isometry learning, adaptive tomography, QAOA, XY mixing, CVaR and Dicke-state preparation each have published precedents. Avoid an unverified claim to be the first to combine them.

## Chapter map

### 1. Introduction

Explain process characterization and the cost of selecting measurements adaptively. State the central question and six subsidiary questions above. List bounded contributions and define which claims the thesis does not attempt. End with a chapter map, rather than presenting simulation volume as a contribution by itself.

### 2. Background and related work

Develop density operators, CPTP channels, Kraus nonuniqueness, Choi representations and repeated-channel observations. Explain finite-count likelihood, Stiefel constraints and gauge-independent design coordinates. Review adaptive process tomography, D-optimal design, submodular selection, QAOA and alternating-operator circuits, parameter transfer, initial-state/mixer choices and tail objectives. Use the verified bibliography as a foundation, then follow citation chains to establish the precise gap. The existing short report bibliography is not a complete thesis literature review.

### 3. Mathematical and computational framework

Derive the observation model and calibrated readout POVM, repeated-channel probability Jacobian, Fisher blocks, ridge and log-determinant utility. Present the submodularity derivation and its fixed-model scope. Derive singleton/pair matching and the three-part loss decomposition. State the exact ordered RXX/RYY circuit, coefficient scaling and initial-state construction. Explain noisy objective selection, actual final sampling, expected-best order statistics and CVaR. Separate physical channel applications, characterization shots, optimizer shots, classical energy evaluations and elapsed simulation time.

### 4. Original controlled benchmark

Present the original protocol exactly as executed: six historical development channels, thirty independent evaluation channels, frozen checkpoints, solver seeds, angle reuse, selection sizes and depth/budget choices. Keep the original primary result prominent. Explain why the favourable deeper secondary configuration does not replace it. Present strong swap/random/exact references, loss decomposition, adaptive outcomes and fit verification. Treat scope and finite-sample uncertainty explicitly.

### 5. Mechanism study and fresh confirmation

Identify the limited claims possible from the original post hoc distributions. Introduce the separately frozen extension and explicit reuse of all thirty prior evaluation channels as development. Show the controlled factorial, bounded landscapes and post hoc support analysis. Explain the locked intervention rule, sample-size rule and timestamped provenance. Present the new primary result before secondary outcomes. Follow with two-repetition adaptive trajectories, Choi-distance analysis, uncertainty, all fitting flags and cost limitations of ideal Dicke initialization.

### 6. Discussion and conclusion

Answer each question using the appropriate evidence. Discuss why a stronger batch optimizer need not improve finite-budget prediction and why local greedy optimization is a demanding control. Distinguish structural impossibility at a fixed circuit depth from local optimizer failure. Explain which limitations weaken generalization and which simply define a legitimate scope. Propose hardware compilation/noise, alternative mixers, CP-aware design and multi-qubit generalization as separate next studies. Conclude with what was learned, not with an unearned advantage claim.

### Appendices

- Frozen protocols, channel distributions and named seeds.
- Ising mapping, gate conventions, Jacobian derivation and numerical validation.
- Complete resource definitions and fitting diagnostics.
- Channel-level results and secondary contrasts.
- Repository version, environment, replication commands and release manifest.

## Claim-to-file mapping

| Thesis claim | Authoritative evidence |
|---|---|
| Original transfer result | `../../analysis/summary.json`, `../../results/frozen/` |
| Original source verified unchanged | `archived_reverification.json`, `protocol.json` |
| Finite-shot baseline preserved | `validation.json` |
| Intervention selected without fresh outcomes | `selection.json`, `results/development/` |
| Fresh primary estimate | `analysis/summary.json`, `analysis/confirmation_channel_means.csv`, `results/confirmation/` |
| Actual budgets and fit choices | `data/trajectories/`, `verification.json` |
| Structural support limitation | `analysis/development_reachability_posthoc.csv`, `THEORY_AND_CLAIMS.md` |
| Classical local-design quality | `analysis/historical_summary.json`, `../../data/evaluation_problems.json` |
| Reconstruction accuracy | `analysis/fresh_trajectory_contrasts.csv`, `analysis/fresh_curves.csv` |
| Foundational methods are established | `references.bib`, `THEORY_AND_CLAIMS.md` |

## Writing-stage checks

1. Agree the precise scope and contribution with the supervisor; institutional rules determine formal acceptance.
2. Read the cited papers in full and expand the literature review through citation chains. Do not infer priority from a sparse keyword search.
3. Keep original confirmation, reused development and fresh confirmation populations clearly separated in every table caption.
4. Use one consistent notation for physical-system qubits, optimizer-register qubits, evolution length, shots and application budget.
5. State the number of physical channels beside every uncertainty interval. Repetitions and checkpoints remain within-channel observations.
6. Report every fitting flag and the finite nature of the offline refit search. A numerical pass is not global optimality.
7. Keep exploratory support and uniform-reference analyses labelled post hoc. The protocol is locally frozen, not externally preregistered.
8. Never count ideal Dicke preparation as free hardware. No new hardware evidence was produced.
9. Freeze a release commit and use its hashes when producing the final thesis figures and tables.
10. Apply the university's authorship, tool-use, research-integrity, formatting and submission requirements to the thesis document.

These are writing and academic-synthesis tasks. They do not require automatically enlarging the project into a hardware or multi-qubit thesis.
