# Results material for thesis drafting

## Draft abstract

Adaptive quantum process tomography requires selecting informative experiments while accounting for the computational cost of that selection. This study separates quadratic batch-optimization quality, local Fisher-information gain and downstream channel-reconstruction accuracy in a reproducible one-qubit simulation framework. An initial paired benchmark on thirty independently sampled channels did not establish a benefit from previous-round QAOA angle reuse at its prespecified primary configuration. A subsequent controlled development study varied feasible-state initialization, mean-energy versus CVaR training, and finite-shot versus exact-expectation objectives. The selected initialization was tested on 30 untouched channels with two acquisition repetitions. Its sampled best-of-256 normalized regret difference from the original QAOA was -0.041959 (95% interval [-0.050926, -0.032992]), which established a reduction. At 4,056 unknown-channel applications, the paired final prediction-RMSE difference was -0.002838 [-0.009191, +0.003514]. A structural support analysis identifies optimal batches inaccessible to some shallow-circuit starting states, while classical greedy and swap controls remain demanding. The findings concern ideal simulated optimizer circuits under known readout calibration; quantum hardware or computational advantage is not demonstrated.

## Primary-result paragraph

The fresh confirmation analysis established a reduction in sampled quadratic regret for Dicke + mean relative to the original initialization. The paired channel-level effect was -0.041959 [-0.050926, -0.032992]. Each of the 30 independent channels contributes one average over two acquisition repetitions, two checkpoints and three solver seeds. Execution counts are not independent sample sizes. The practical margin of 0.01 refers to the feasible energy range, not prediction error.

## Downstream-result paragraph

The final prediction-RMSE contrast was -0.002838 [-0.009191, +0.003514] and did not establish the direction of the mean accuracy difference. This secondary analysis includes policy-dependent measurements, refitting and adaptive feedback. It should be reported alongside normalized Choi trace distance and resource use; it is not a multiplicity-adjusted discovery claim.

## Classical-reference qualification

The post hoc expected-best-regret contrast for the selected circuit against uniform feasible sampling was -0.000525 [-0.001009, -0.000041]. This compares exact distribution expectations for the same final-sample count. The circuit also incurs objective-evaluation shots and uncompiled state-preparation cost. It is not an equal-runtime comparison.

## Essential caveat

Retain the original negative transfer result and the separately frozen extension as distinct experiments. Use the baseline report and the release CSVs for tables; do not pool historical development channels into fresh confirmation.
