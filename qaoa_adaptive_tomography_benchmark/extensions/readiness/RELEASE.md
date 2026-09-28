# Completed research release

This additive extension is based on repository commit `58f7a079c1e3fbbf8d98a9ef6272d2f4973d7e0d`. It preserves the original scientific implementation and benchmark outcomes. Keep the original and extension confirmation populations separate.

## Read first

- `RESULTS_FOR_WRITEUP.md`: draft abstract and evidence-based results text.
- `THESIS_WRITEUP_PLAN.md`: research questions, chapter plan and claim-to-file map.
- `THEORY_AND_CLAIMS.md`: derivations, interpretation and limits.
- `analysis/summary.json`: machine-readable final results.
- `verification.json`: completed independent saved-result audit.
- `README.md`: methods, output map and replication sequence.

## Main outcome

The development-selected Dicke initialization reduced actual best-of-256 normalized quadratic regret from 0.048136 to 0.006177 on 30 new physical channels: paired difference -0.041959, 95% stratified interval [-0.050926, -0.032992]. The intervention passed the locally frozen practical margin of 0.01. The unit of independence is the physical channel.

Final reconstruction RMSE did not show an established advantage over original QAOA or swap search. The selected-minus-swap contrast was -0.000940 [-0.005913, +0.004034]. These are unadjusted secondary intervals, not equivalence tests. No hardware or quantum computational advantage is claimed.

The release contains 1,728 development and 720 fresh frozen-state solver runs, 240 complete adaptive trajectories, bounded landscape diagnostics, all raw fitting candidates and acquisition counts, editable figures, three native LaTeX tables, and verified foundational references. Validation includes Qiskit circuit checks, original-baseline reproduction and independent recalculation of the primary result and every committed trajectory endpoint.

## Reuse without rerunning experiments

From the benchmark root, after installing `requirements.txt`:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python extensions/readiness/verify.py
python extensions/readiness/report.py --output-dir readiness_report
```

The report command regenerates the Word report, figures, LaTeX tables and drafting fragments from saved results. Running verification or report generation updates timestamps or generated files, so verify release hashes before regeneration if you need to confirm the original delivered bytes. `release_manifest.json` lists SHA-256 hashes for every delivered extension file other than the manifest itself.

The logical-resource counts exclude abandoned incomplete simulation work; execution interruptions and deterministic resumptions are documented in `execution_notes.json`. No physical hardware shots were taken.

The remaining thesis work is literature synthesis, academic writing, supervisor review and institutional presentation requirements. This release supports a bounded computational thesis; it is not a completed thesis manuscript or a guarantee of institutional acceptance.
