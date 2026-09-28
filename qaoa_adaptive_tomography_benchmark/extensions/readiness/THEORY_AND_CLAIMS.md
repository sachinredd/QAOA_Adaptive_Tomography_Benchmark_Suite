# Mathematical framing and claim boundaries

## Local experimental-design objective

At a fixed fitted channel, let F_0 be accumulated Fisher information plus the positive ridge, and let A_i be the positive-semidefinite Fisher block for setting i, including its prospective shot count. The implemented utility is

G(S) = log det(F_0 + sum_{i in S} A_i) - log det(F_0).

F_0 is positive definite, G(empty)=0, and G is nonnegative and monotone. Its marginal gain can be written

G(S union {e}) - G(S) = integral_0^1 Tr[(F_S + t A_e)^(-1) A_e] dt.

If S is a subset of T, then F_S is no larger than F_T in the positive-semidefinite order. Inversion reverses this order. Multiplication inside the trace by A_e preserves the nonnegative trace difference. Therefore the marginal gain of e cannot increase when S is enlarged: G is submodular. This argument does not require the matrices to commute.

The standard cardinality-constrained greedy theorem (Nemhauser, Wolsey and Fisher, 1978) therefore applies to the full-pool *local* design problem. For a batch of k settings the guarantee is 1 - (1 - 1/k)^k; for k=3 it is 19/27, with the familiar limiting bound 1 - 1/e. Here each selected setting costs 408 unknown-channel applications, so cardinality is consistent with the declared application budget despite different shot counts.

This guarantee does not apply directly to final prediction RMSE, to the quadratic approximation, to hardware runtime, or to a sequence in which the fitted channel changes after every round. It explains why full-pool greedy deserves to be a serious comparator. The proof is an application of known results, not a claim of a new submodular-optimization theorem.

Also distinguish the finite search-space size from asymptotic hardness. For fixed batch size k, exhaustive subset enumeration has binomial(n,k)=O(n^k) candidates. Thus the select-three and select-five tracks do not become exponentially large families merely by increasing n while keeping k fixed. This does not rule out a polynomial quantum speedup under a specified cost model; it rules out treating a generic NP-hardness label as evidence of such a speedup in this experiment.

## Quadratic approximation and loss decomposition

Let v_i=G({i}) and r_ij=v_i+v_j-G({i,j}). Submodularity gives r_ij >= 0. The code minimizes E(x)=-sum_i v_i x_i + sum_{i<j} r_ij x_i x_j at fixed sum_i x_i=k. This is exact for singleton and pair selections; it omits higher-order terms for k>=3. QUBO optimality consequently is not optimality for G.

If S_full, S_short, S_Q and S_alg denote the full-pool true optimum, shortlist true optimum, exact quadratic optimum and selected batch, then

G(S_full)-G(S_alg) = [G(S_full)-G(S_short)] + [G(S_short)-G(S_Q)] + [G(S_Q)-G(S_alg)].

The first two terms are nonnegative; the last is signed. A quadratic solver error can improve true utility when the surrogate misranks batches. Every comparison here uses a single fixed fitted model; quantities from different adaptive branches should not be mistaken for identical objectives.

The magnitude of a log-determinant difference also needs interpretation. For two positive-definite information matrices in the same d-dimensional chart, a log-determinant advantage Delta corresponds exactly to a ratio exp(Delta/d) of their geometric-mean eigenvalues. The original report's mean signed selection loss of 0.235315 across twelve coordinates corresponds to about 1.020 in the geometric average of these ratios. This is not a uniform improvement in every direction, and it does not predict a 2% change in held-out RMSE. It illustrates why optimum-hit percentages or normalized energy regret should not be read as proportional reconstruction improvements.

## Why best-sample selection is a different objective

The archived circuit minimizes expected energy but returns the lowest-energy sampled batch. Sort basis states by energy and then by integer index, with probabilities p_i and cumulative lower mass C_i. The probability that state i is selected from N independent samples is

w_i=(1-C_{i-1})^N-(1-C_i)^N.

Expected selected energy is sum_i w_i E_i, not sum_i p_i E_i. This identity is implemented and independently checked in the extension. It is an offline distribution diagnostic; finite-shot policies still return an actually sampled state. Lower-tail CVaR is an established alternative objective, not an invention of this study.

The benchmark deliberately chooses the deployed batch from a separate final sample stream. It does not retain the best bitstring seen during training as an eligible final answer. This makes trained-distribution quality at a fixed final-sample budget well defined, but it is not necessarily the most efficient operational use of all observed bitstrings. An all-observed-candidates deployment rule is an untested alternative and should not be ruled out by the present conclusions.

## Structural reachability at shallow depth

An ordered RXX/RYY pair on a chain edge acts as a two-state rotation between feasible strings differing by an adjacent 01/10 swap. It cannot move amplitude to any other string. The diagonal cost layer cannot expand support. Starting with the chosen feasible basis string, propagate a Boolean reachable set through the two fixed initial sweeps and p variational sweeps. On each edge, replace the membership of each swapped pair by their Boolean OR.

This produces a support upper bound valid for *all* variational angles. If every globally minimizing feasible string lies outside this set, no angle optimizer or increase in final shots can discover an optimum at this depth and initialization. Interference may reduce actual support, so inclusion in the reachable set is not a guarantee of nonzero probability or of discoverability.

This explains a ceiling on optimum recovery, not the original null transfer contrast by itself. Cold and transferred angles share the same feasible starting string within a paired frozen comparison; transfer could still improve energy within that common support. A causal account of the transfer contrast would require an additional intervention directed at objective drift, ordering or the angle-initialization procedure.

The retrospective application to saved development data is labelled post hoc. The Dicke initial-state intervention starts with nonzero amplitude on every feasible string. It does not prove that the resulting circuit beats uniform random search, or that its preparation cost is negligible. Initial-state and mixer alignment are established research topics; a uniform Dicke state is not assumed to be the ground state of the ordered chain mixer.

The register positions follow decreasing singleton information value. A chain mixer is not invariant under arbitrary permutations of this ordering: relabelling cost variables without correspondingly relabelling the mixer graph changes the ansatz. The reachability finding therefore concerns the implemented ordering as well as its depth and initial bitstring. Random-order or structure-aware ordering experiments are possible follow-ups; no causal ordering ablation was executed here.

## Physical model and uncertainty

The learned channel remains CPTP through a stacked Kraus isometry. Fisher information uses twelve free Pauli-transfer coordinates. Those coordinates remove Kraus gauge redundancy but include directions that need not stay inside the CP cone near a boundary. The ridge is an information-design regularizer, not a Bayesian posterior or a confidence region. The theorem above holds for the resulting fixed positive-semidefinite blocks; its interpretation as predictive accuracy remains approximate.

Truth enters only synthetic observation generation and frozen evaluation. Prediction RMSE is over declared held-out probabilities; normalized Choi trace distance is one half of the trace norm between trace-one Choi matrices. Neither is a hardware failure probability. The fresh study averages solver seeds, checkpoints and acquisition repetitions within each physical channel before calculating uncertainty. The six equally weighted families define the target population.

## Allowed conclusions

- The archived transfer test is inconclusive at its prespecified primary configuration.
- A controlled initialization intervention may improve this particular shallow QAOA solver; the fresh contrast, not development selection, determines that conclusion.
- Good quadratic optimization, high local information gain and accurate reconstructed channels are separate empirical endpoints.
- Strong classical controls and state-preparation cost are required before making any practical quantum-advantage claim.
- Simulation evidence is bounded to the declared channel distribution, calibration, circuit, budget and fitter.

## Conclusions that the evidence cannot support

- Quantum computational or hardware advantage; universal QAOA superiority or failure.
- An equivalence claim merely because an interval crosses zero.
- An externally preregistered study: the protocol is locally timestamped and hashed.
- A new invention of Kraus learning, adaptive tomography, XY mixing, Dicke preparation, CVaR or parameter transfer.
- General scaling hardness inferred by increasing the number of candidate settings alone.
- A globally optimal circuit-angle solution inferred from a finite grid or a local search.

## Reference map

| Component | Prior work | Contribution of this project |
|---|---|---|
| QAOA | Farhi, Goldstone and Gutmann (2014), arXiv:1411.4028 | Application and evaluation, not a new ansatz |
| Constraint-preserving mixing | Hadfield et al. (2019), doi:10.3390/a12020034 | Explicit ordered chain implementation and verified feasible-subspace simulation |
| Kraus-isometry learning | Ahmed, Quijandria and Kockum (2023), doi:10.1103/PhysRevLett.130.150402 | Reuse within a finite-count, matched-application acquisition study |
| Adaptive process tomography | Pogorelov et al. (2017), doi:10.1103/PhysRevA.95.012302 | Comparison of measurement-batch optimization policies |
| Parameter transfer | Shaydulin et al. (2023), doi:10.1145/3584706 | Tests sequential tomography instances rather than assuming MaxCut results transfer |
| Initial-state/mixer alignment | He et al. (2023), doi:10.1038/s41534-023-00787-5 | Controlled initialization change and reachability diagnostics on tomography objectives |
| CVaR | Barkoutsos et al. (2020), doi:10.22331/q-2020-04-20-256 | Controlled comparison against mean-energy training |
| Dicke preparation | Bartschi and Eidenbenz (2019), arXiv:1904.07358 | Ideal-state intervention; efficient physical preparation is not compiled or benchmarked here |
| Submodular greedy bound | Nemhauser, Wolsey and Fisher (1978), doi:10.1007/BF01588971 | Shows why the exact local design has a principled classical control |

The search establishes relevant precedents, not an exhaustive priority claim. Avoid 'first' claims without a systematic database review and supervisor approval of the precise novelty statement.
