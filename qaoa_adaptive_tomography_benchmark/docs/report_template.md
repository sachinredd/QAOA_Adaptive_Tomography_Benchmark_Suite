# QAOA for adaptive quantum process tomography
Simulation benchmark of parameter reuse and measurement selection
Research report prepared 27 September 2026

This report evaluates QAOA as the optimizer that chooses the next measurement batch in adaptive quantum process tomography. It separates circuit optimization, the information value of the selected measurements, and the accuracy of the learned channel after new data are acquired.

The completed frozen-state benchmark contains 12,960 QAOA optimizations and 9,720 classical heuristic runs. The independent evaluation set contains thirty fresh physical channels. A further 210 adaptive trajectories compare seven policies at the same unknown-channel application budget, including 180 additional QAOA optimization calls within the two quantum-selector policies.

The primary test did not establish an advantage from reusing angles from the previous round. Transferred minus cold expected normalized energy regret was {{primary}}, with a 95% channel-level interval of {{primary_ci}}. Negative values favor transfer. This is an optimizer diagnostic, not a difference in tomography prediction error.

Classical swap search provided a demanding control. For select-3-of-14 problems at the selected QAOA configuration, its quadratic-optimum success rate was {{swap_success}} compared with {{cold_success}} for cold QAOA and {{transfer_success}} for transferred QAOA, using 256 candidate evaluations or final samples respectively. QAOA additionally used objective-evaluation shots. These are descriptive success rates over paired frozen states and solver seeds.

In the fresh adaptive comparison, transferred minus cold final prediction RMSE was {{trajectory_difference}}, with a descriptive 95% interval of {{trajectory_ci}}. This comparison did not establish an accuracy benefit from reusing angles. Cold QAOA had the smallest observed mean RMSE, but its paired difference from greedy also included zero.

The benchmark establishes a reproducible QAOA-focused investigation. It does not demonstrate quantum advantage, and no IBM hardware or QAOA gate-noise experiment was performed. All statistical conclusions remain scoped to the declared one-qubit channel distribution and known readout calibration.

@page
# 1 Study design and independent evidence
The protocol was frozen before the new benchmark outcomes were examined. The development set consists of six archived greedy trajectories, one from each physical family, with the learned model after the pilot and after each of the first two adaptive rounds. These eighteen checkpoints are used for implementation development and for selecting the depth and evaluation budget used in the subsequent adaptive comparison.

The evaluation set uses thirty newly sampled channels: five amplitude-damping, five thermal-relaxation, five dephasing, five coherent-rotation, five driven-dissipation and five depolarizing channels. The broad parameter distributions are inherited from the earlier generalisation study. New named random streams control channel draws, acquisition, initialization, validation and refitting. Historical outcomes are not pooled into the fresh primary analysis.

@table design

Each fresh channel supplies three frozen learned states along a full-pool greedy acquisition trajectory. Every frozen solver sees the same learned model, accumulated data, candidate shortlist and quadratic objective for its problem. The first state provides no preceding angles, so cold and transferred QAOA are deliberately identical there. The primary comparison uses only the two subsequent states.

The primary independent units are the thirty physical channels. Solver seeds and checkpoints are averaged within each channel. Thousands of optimization executions improve characterization of the algorithms on these inputs, but do not create thousands of independent physical channels. The end-to-end comparison likewise uses one paired outcome per policy and physical channel.

@page
# 2 Channel learning and the selection objective
The characterized system has one qubit. Its reference evolution is a stationary channel generated from the declared Markovian family. The learned model has up to four Kraus operators. Stacking those operators into an isometry ensures trace preservation; the operator-sum representation ensures complete positivity. The QAOA register has one qubit per shortlisted experiment and is separate from the characterized one-qubit system.

The candidate pool contains ninety settings: six Pauli-eigenstate preparations, five evolution lengths and three measurement bases. The common pilot uses twelve one-step settings with thirty-two shots each. Assignment errors of 0.03 and 0.06 are included in both the observation oracle and the known fitting POVM. State preparation and measurement rotations remain ideal.

Channel learning uses the archived finite-count likelihood and analytic Stiefel descent. At every acquisition round, the incumbent and three independent global starts are fitted for at most 2000 iterations each, with tolerance 10⁻⁶. Training likelihood alone chooses the model; held-out truth never chooses the fit or the acquisition policy.

The design calculation uses twelve free Pauli transfer coordinates. Prospective Fisher information includes the actual number of shots available to each setting. An identity ridge with coefficient one stabilizes the accumulated information. Local batch utility is the increase in log determinant after adding the candidate information blocks.

@eq gain

The shortlist retains the largest singleton information values. The quadratic model rewards those values and subtracts pairwise redundancy. It matches singleton and pair utilities by construction, while batches of three or five omit higher-order interactions.

@eq qubo

Three distinctions remain essential: solving the quadratic objective well, selecting a batch with high actual local information gain, and improving prediction after refitting are different achievements. Exact quadratic optimization is a solver reference; it is not a certificate of optimal tomography.

@page
# 3 QAOA circuit and parameter reuse
The circuit uses the archived RZ and RZZ cost gates with a connected, ordered chain of paired RXX and RYY operations. These mixer operations preserve the required number of selected experiments in ideal simulation. Two fixed mixer sweeps spread an initially feasible computational-basis state before the alternating layers. The order of the mixer gates is preserved exactly; it is not replaced by an exponential of the sum of overlapping terms.

Cold initialization draws the circuit angles independently at each checkpoint. The transfer variant starts from the preceding checkpoint's selected angles for the same channel, shortlist size, batch size, depth, evaluation cap and solver seed. Both variants use paired feasible starting bitstrings and named random streams. Only angles transfer: neither the previous model nor an exact solution replaces the current optimization problem.

The shortlist can change between rounds, and its ordering follows current singleton scores. Angle reuse therefore assumes some useful similarity between successive normalized objectives; the benchmark tests that assumption rather than enforcing it. QAOA Hamiltonian coefficients use the same normalization as the archived implementation.

COBYLA optimizes a sampled mean energy using 128 shots per evaluation, with caps of twenty or eighty evaluations. The returned angles are those with the best observed objective value, as in the archived solver. The final batch is the lowest-energy feasible outcome actually sampled in the final 256-shot stream. The first eight and thirty-two samples provide nested budget checkpoints; they are not separate optimization runs.

The new simulator represents only the feasible Hamming-weight subspace. It computes the same ideal circuit distribution more efficiently on a classical computer. Twenty comparisons against Qiskit's full statevector covered all five selection sizes, both depths and two angle vectors; the largest probability discrepancy was {{circuit_error}}. Reproducibility, sample-only selection and resource accounting checks also passed.

Enumeration in this simulator constructs amplitudes and phases; exact minimizing bitstrings are not passed to the optimizer. Nevertheless, it remains classical simulation. Its execution time is not evidence of quantum speedup, and its ideal feasible-sample rate says nothing about leakage under device noise.

@page
# 4 Primary parameter transfer result
The prespecified endpoint is transferred minus cold expected normalized quadratic energy regret at fourteen candidates, batch size three, depth one and a twenty-evaluation cap. Expected regret is evaluated from the final ideal circuit distribution after optimization. It is an offline diagnostic and was not available to the finite-shot objective optimizer.

@eq regret

The three solver seeds and the two post-pilot checkpoints are averaged within each channel. Equal weighting of the six physical families gives the overall estimate. The interval uses within-family channel variances and Welch-Satterthwaite degrees of freedom.

@fig fig01_transfer_scaling|Figure 1. Expected normalized energy regret for depth one and cap twenty. Error bars are approximate 95% channel-level intervals for each policy mean. Lower is better. The prespecified paired test concerns the fourteen-candidate point, rather than overlap between these separate mean intervals.

The primary estimate was {{primary}} {{primary_ci}}. The interval crosses zero, so this experiment did not establish either an improvement or a disadvantage from angle transfer under the primary configuration. It also does not establish equivalence. The point estimate corresponds to {{primary_points}} percentage points of the feasible energy range, not probability prediction error.

Changing shortlist size changes both the search space and the energy normalization. Comparisons across sizes are descriptive diagnostics of this implementation; they are not a universal scaling law or evidence that a particular size is computationally hard.

@page
# 5 Depth and optimization budget
The benchmark also evaluated depth two and the eighty-evaluation cap. These comparisons were planned as descriptive analyses; they do not replace the primary test or carry a multiplicity-adjusted discovery claim.

@fig fig02_transfer_contrasts|Figure 2. Paired transfer effects at fourteen candidates and batch size three. Negative differences favor angle reuse. Each interval uses thirty independent channel means. Only depth one with cap twenty is the primary comparison.

@table transfer

The depth-two, eighty-evaluation comparison favored transfer, with difference {{deep_difference}} and descriptive interval {{deep_ci}}. This is evidence of configuration dependence worth testing on a further untouched set; it does not turn the nonsignificant primary outcome into a general claim that transfer helps.

The configuration used for the adaptive trajectories was selected before evaluating those trajectories. On the separate archived development set, depth one with cap eighty had the lowest mean expected regret averaged across cold and transferred starts. The selection rule therefore retained that configuration even after the evaluation benchmark showed a more favorable transfer contrast elsewhere.

An eighty-evaluation cap is an upper bound, not an assertion that every optimization uses eighty evaluations. Actual objective counts and shots are recorded. Comparisons across budgets describe observed quality and cost; the study did not prespecify a noninferiority margin for claiming that fewer shots preserve performance.

@page
# 6 Finding the best batch from sampled outcomes
Expected energy and best-sampled energy answer different questions. A distribution can concentrate on moderately good solutions and lower its mean energy, while assigning very little probability to the best solution. Uniform random search samples a broader set and can be competitive when the feasible space is small.

@fig fig03_sampling_success|Figure 3. Descriptive fractions attaining the exact quadratic optimum after thirty-two or 256 final QAOA samples. QAOA uses the development-selected configuration and also incurs its objective-evaluation shots. Random search uses the panel's sample count; swap search uses 256 candidate evaluations in both panels. Five-setting batches are a separate, higher-application-cost track.

@table success

The original select-3-of-8 problem has only fifty-six feasible batches. A uniform search with 256 independent draws includes any particular optimum with probability approximately 99.0%. For select-3-of-14 the space grows to 364, and that probability is approximately 50.6%. These are analytical probabilities for a particular optimum; ties can increase success.

The executed select-3-of-14 results confirm that strong inexpensive classical controls are necessary. Swap search reached the exact quadratic optimum on {{swap_success}} of the displayed problems and seeds, substantially more often than either QAOA variant. Equal final-sample counts also omit QAOA's preceding optimization measurements, so they should not be interpreted as equal total cost.

@page
# 7 Shortlist and surrogate losses
The solver receives a restricted candidate set and an approximate batch objective. To identify where performance is lost, each frozen triple-selection problem was also compared with the exact local information optimum over the full ninety-setting pool and with the exact optimum within the shortlist.

The total local utility loss separates into exclusion by shortlisting, replacement of the true local utility by the quadratic surrogate, and the difference between the exact surrogate solution and the sampled QAOA output. The last component can be negative: a surrogate solver error can accidentally choose a better batch under the true local objective.

@fig fig04_approximation_losses|Figure 4. Mean losses in local log-determinant information gain for transferred QAOA at the development-selected configuration. The QAOA component is signed. These quantities are not prediction RMSE. Full-pool exact enumeration is restricted to triples; select-five audits retain only shortlist references.

At fourteen candidates, the mean shortlist loss was {{shortlist_loss}}, the quadratic-approximation loss was {{surrogate_loss}}, and the signed QAOA selection loss was {{solver_loss}}. These results identify costs of the complete selection procedure that would be hidden by reporting only its quadratic energy.

Adding candidates provides a richer choice of measurements, but it also changes the difficulty faced by the finite-depth circuit. Improving the solver alone cannot recover experiments excluded from the shortlist. Conversely, eliminating a quadratic energy gap cannot remove higher-order approximation error. These mechanisms should remain separate in a QAOA-focused thesis.

@page
# 8 Fresh adaptive trajectory comparison
The final stage compares seven complete acquisition policies on the thirty fresh channels: full-pool greedy, the previously frozen independent-prior schedule, exact quadratic optimization, uniform random quadratic search, swap search, cold QAOA and transferred QAOA. The quadratic policies all use fourteen shortlisted candidates and select three experiments per round.

The common pilot is sampled and fitted once for each channel, then reused across its seven paired policy branches. The branches make their own subsequent choices and receive setting-and-visit-specific observation streams. QAOA angle transfer now follows each policy's own evolving trajectory, rather than the shared greedy checkpoints used by the isolated solver benchmark.

Each selected setting receives 408 unknown-channel applications. Shots equal 408 divided by the evolution length. Three rounds of three settings, together with the 384-application pilot, give exactly 4056 applications for every trajectory. Characterization shot counts can differ because policies choose different evolution lengths.

@table development

The independent development rule selected depth {{selected_depth}} with cap {{selected_cap}}. Both QAOA policies use that configuration and 256 final samples. Random search uses 256 feasible proposals, and swap search uses 256 candidate-energy evaluations. The fixed-prior schedule is reused unchanged from its independent thirty-channel prior ensemble.

Final prediction RMSE compares the learned channel with exact synthetic probabilities on 144 held-out setting-outcome entries: eight random input states, three unseen evolution lengths and three measurement bases. These truth-based scores are evaluated only after each acquisition trajectory is frozen. No held-out score changes a fit, a batch or the selected QAOA configuration.

All end-to-end contrasts in this benchmark are descriptive secondary analyses. The one prespecified primary endpoint remains the frozen-state transfer diagnostic. One acquisition repetition per physical channel limits precision for the downstream policy comparisons.

@page
# 9 Final tomography accuracy
@fig fig05_trajectory_accuracy|Figure 5. Mean final prediction RMSE at exactly 4056 unknown-channel applications. Error bars show approximate 95% channel-level intervals for policy means. Smaller is better. Paired differences, rather than overlap between mean intervals, are used to compare policies.

@table accuracy

Transferred QAOA had mean final RMSE {{transfer_rmse}}, compared with {{cold_rmse}} for cold QAOA and {{greedy_rmse}} for full-pool greedy. The best observed policy mean in this fresh set was {{best_policy}} at {{best_rmse}}; identifying that smallest mean is descriptive and does not establish a statistically ordered ranking of all seven methods.

An RMSE of 0.05 corresponds to about five percentage points of root mean squared probability prediction error. It is not a device failure probability, a count of wrongly classified experiments or a process infidelity. The displayed means concern a fixed endpoint budget and do not directly quantify shots saved to reach a target accuracy.

Swap search selected the same batch as exhaustive quadratic optimization in all ninety adaptive decisions. The paired observation streams and fitting procedure therefore produced identical trajectory outcomes for those two policies. This is an observed result on this finite set, not a guarantee that local search always finds the optimum.

@page
# 10 Paired downstream comparisons
Each difference compares two policies on the same physical channel before averaging across channels and families. Negative values favor transferred QAOA in the figure. The paired design removes some between-channel variation without treating the seven policy branches as independent channel draws.

@fig fig06_trajectory_contrasts|Figure 6. Descriptive paired final prediction differences for transferred QAOA against the other six policies. The intervals are not adjusted for multiple comparisons. An interval crossing zero is inconclusive about direction and does not prove equivalence.

Transferred minus cold QAOA was {{trajectory_difference}} {{trajectory_ci}}. Transferred minus greedy was {{transfer_greedy}} {{transfer_greedy_ci}}, and transferred minus swap search was {{transfer_swap}} {{transfer_swap_ci}}.

These comparisons ask whether the complete adaptive workflow benefits from the selected quantum solver. They include feedback: a different early batch changes the fitted channel and hence later shortlists and objectives. This is why a frozen-state optimizer advantage, if present, would not by itself establish a final tomography advantage.

The exact quadratic policy also provides a useful interpretive control. Its difference from full-pool greedy was {{exact_greedy}} {{exact_greedy_ci}}. That contrast combines the effects of shortlisting, the quadratic surrogate and subsequent adaptive feedback; it is not a test of a quantum circuit.

@page
# 11 Resource accounting
@fig fig07_resources|Figure 7. Mean characterization shots and total attempted shots per complete adaptive trajectory. Every policy uses exactly 4056 unknown-channel applications. QAOA's optimizer measurements are counted separately before being added to the displayed bookkeeping total.

@table resources

Transferred QAOA used a mean of {{transfer_total_shots}} attempted shots, including {{transfer_objective_shots}} objective shots and 768 final optimizer shots. Cold QAOA used {{cold_total_shots}} total attempted shots. These totals include the pilot and every completed acquisition round; discarded infeasible samples are absent only because the QAOA register is simulated ideally within a feasible subspace.

The units are not interchangeable. A tomography shot can apply the unknown channel several times; an optimizer shot prepares and measures a different register. Classical energy evaluations, state preparation, readout, compilation, queueing and hardware gate duration are not given an invented exchange rate. Simulator runtime is retained for reproducibility but cannot demonstrate quantum hardware speed.

The solver benchmark charges all 256 final samples. Its smaller sample-prefix results are checkpoints within that stream. Computing exact reference optima, full-pool audit gains and diagnostic distributions is offline analysis and is not presented as free production-policy capability.

@page
# 12 Learning over successive acquisitions
@fig fig08_learning|Figure 8. Mean prediction error over the common pilot and three completed acquisition rounds for selected policies. Intermediate curves are descriptive. Policies share the pilot within each physical channel, then follow their own choices.

The curves show how much of the final behavior emerges early and whether later rounds reduce the remaining prediction error. They summarize the same thirty physical channels throughout; each checkpoint is not a new independent replicate. They should be read alongside the endpoint comparisons and fitting diagnostics.

Four-candidate fitting remains important because a poor local likelihood solution can distort both acquisition decisions and the final policy ranking. The completed benchmark retained {{stationarity_flags}} final trajectories above the diagnostic gradient threshold and {{material_flags}} with material improvement in the prescribed short offline refit. There were {{distinct_flags}} distinct flagged final trajectories; none was removed or replaced in the main results.

The offline check runs four hundred continuation iterations and one fresh four-hundred-iteration fit. A likelihood improvement greater than 10⁻⁵ per shot is considered material. Passing this finite search is not a certificate of global optimality. The largest observed offline improvement was {{max_refit_gain}} per shot.

The original and diagnostic models remain in the suite. The main comparisons use the acquisition models, because replacing only flagged endpoints after the experiment would change the analysis while leaving earlier decisions based on the original fits.

@page
# 13 Verification and mechanism diagnostics
The delivered results were checked independently of the summary-generation code. Every completed trajectory round has file checksums. Candidate likelihoods were recomputed from saved observation counts, the likelihood-winning candidate was compared with the selected model, and application and shot totals were rebuilt from the observations.

@table verification

Pilot counts were identical across all seven policies for each channel. Every logical fit candidate was re-evaluated, including the shared pilot copies. The 3360 logical candidates correspond to 2640 distinct executed acquisition fits because each channel's pilot is fitted once. A further 420 short offline fitting branches support the final convergence diagnostics.

A post hoc analysis inspected the already saved ideal QAOA distributions. For select-3-of-14 at the selected configuration, cold QAOA's mean effective support was {{cold_support}} of 364 feasible batches; transfer gave {{transfer_support}}. Effective support is the exponential of distribution entropy, not a count of observed unique batches.

Mean expected normalized regret was {{cold_expected}} for cold QAOA and {{transfer_expected}} for transfer, versus {{uniform_expected}} for a uniform feasible distribution on the same problems. Thus improving average energy can coexist with poor best-batch discovery. This observation motivates checking circuit expressiveness, feasible initialization and the optimization objective, but does not by itself identify which mechanism is causal.

These distribution diagnostics were added after the frozen-state results were inspected and are explicitly labeled post hoc. They are useful for selecting a new hypothesis, not for replacing the prespecified primary test or claiming a verified mechanism.

@page
# 14 Implications for a QAOA focused thesis
The benchmark supplies a concrete thesis contribution: an audited separation of QAOA optimization quality, information-design approximation and downstream tomography accuracy. The tested angle-reuse method did not establish a benefit under the primary configuration, while one deeper, larger-budget configuration showed a favorable descriptive transfer effect. That combination supports a narrower configuration-dependent question for future work.

The strong swap-search control changes the immediate research priority. At these sizes, the quadratic problems are readily solved classically. A defensible next experiment would compare feasible-state initialization and mixer structure on the same frozen problems, with circuit depth and shot budgets accounted for. A uniform feasible initial state or a different constraint-preserving mixer should be evaluated as a new intervention, not assumed to improve performance.

Another possible intervention is to change how the circuit distribution is trained when the production decision uses the best sampled outcome. Such a change must be evaluated against the current expected-energy objective, with the choice of training rule made on development channels and then frozen. A fresh held-out set is required after examining the present evaluation outcomes.

The current results do not justify enlarging the problem solely until a classical comparator becomes inconvenient. Larger shortlists or batches should correspond to a scientifically useful tomography design, and stronger classical solvers should remain in the comparison. General computational hardness does not establish that these particular instances are hard.

IBM hardware would provide a separate feasibility study after a concrete circuit intervention is selected. It should measure compiled depth, feasible-sample rate, optimizer shots and noise sensitivity on identical instances. No hardware execution or gate-noise robustness claim is included here.

The thesis can therefore focus on when and why QAOA is effective or ineffective inside adaptive tomography, with a reproducible implementation and transparent negative results. A claim of quantum advantage would require additional evidence beyond this benchmark.

@page
# 15 Reproduction and limitations
The suite contains the frozen protocol, source snapshot, development selection, new physical channel parameters, counts, candidate models, QAOA traces, final samples, exact reference arrays, flat analysis tables and report figures. The README gives a complete replication sequence. A helper creates a clean replication directory without overwriting the delivered results.

@code python scripts/new_run.py --output /absolute/path/new_replication
@code python tests/validate_solver.py
@code python scripts/analyse.py
@code python scripts/verify_results.py
@code python scripts/plot_results.py
@code python scripts/build_report.py

The first two commands prepare a new location and validate its solver when run from the appropriate directories. Analysis and report commands require completed results. Follow the README for acquisition and benchmark execution order; the abbreviated list above is not a substitute for those stages. The development configuration must be selected before executing the remaining fresh policy trajectories.

Limitations include one-qubit characterized systems, a single known readout profile, one acquisition repetition per channel, five physical draws per family, small classically enumerable selection spaces and ideal QAOA gates. Frozen solver inputs arise from greedy trajectories. The separate adaptive stage tests feedback, but it does not remove those physical and statistical limitations. Approximate t intervals have limited finite-sample guarantees, and secondary contrasts are not multiplicity adjusted.

The prior thesis report and matched-cost suite supply the unchanged channel-learning and prior-design methods. This benchmark adds a circuit-equivalent feasible-subspace simulator, explicit angle transfer, classical local-search controls, fresh paired trajectories and a frozen independent analysis. Source and data checksums bind the delivered implementation to its results.

## Method references
Hadfield, S., Wang, Z., O'Gorman, B., Rieffel, E. G., Venturelli, D., and Biswas, R. The alternating-operator framework motivates constraint-preserving mixers. The original paper is From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz, Algorithms 12, 34 (2019), https://doi.org/10.3390/a12020034.

Shaydulin, R., Lotshaw, P. C., Larson, J., Ostrowski, J., and Humble, T. S. Parameter Transfer for Quantum Approximate Optimization of Weighted MaxCut, ACM Transactions on Quantum Computing 4 (2023), https://arxiv.org/abs/2201.11785. That result motivates testing parameter reuse; its MaxCut performance is not assumed to transfer to tomography.

Internal source: Adaptive Kraus Matched Cost Suite and Learned Quantum Channels Thesis Report, completed research archives. Full source archive hashes and extracted checkpoint identities are recorded in protocol/INPUT_PROVENANCE.json.

@page
# Appendix Physical channel distributions
The reference generator uses a Hamiltonian equal to one half of the Pauli vector weighted by the angular-frequency vector. Its dissipators are lowering, raising and Pauli Z, with nonnegative rates denoted down, up and z. The Z dissipator has no additional factor of one half. The time step is one in benchmark units; no mapping to a particular device's microseconds is asserted.

@table channels

Every interval in the table denotes a continuous uniform distribution. Rotation directions are uniform on the sphere, obtained by normalizing a three-dimensional standard normal vector. Unlisted frequencies and rates are zero. The driven-dissipation scalar draws are independent. Each family contributes five fresh parameter draws and receives one sixth of the weight in the primary population estimate.

The fresh physical-channel seed is 27190317. The acquisition-stream root is 27190329; stable hashes derive separate acquisition, initialization, validation and refit streams for each family and instance. Frozen solver seeds are 6101, 6152 and 6203. The adaptive solver root is 7101, with channel-and-round-specific seeds paired across the two QAOA variants.

The prior schedule was planned on thirty independent channels using the earlier seed 713903 and was not retuned on this evaluation set. The development checkpoints use instance zero, repetition 4001 and calibrated readout from the earlier matched-cost archive. These distinct roles prevent development scores or prior planning from being counted as fresh confirmation.

Exact sampled parameters and named seeds are saved in data/evaluation_channels.json. The original family sampler is included unchanged in src/adaptive_kraus/study_sampling.py. The reference channels obey the specified stationary generator model; the fitted discrete channel is constrained to be physical but is not required to have a unique or embeddable continuous-time generator.
