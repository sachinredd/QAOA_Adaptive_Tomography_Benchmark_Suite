# Literature positioning for the thesis

The sources below were checked against primary publisher or author-preprint records. This is a targeted foundation for the literature chapter, not a systematic review or a priority certification. Citations are supplied in `references.bib`.

## The direct experimental-design precedents

**Gazit, Ng and Suzuki (2019), Quantum process tomography via optimal design of experiments.** [Publisher record](https://doi.org/10.1103/PhysRevA.100.012350). This is a direct precedent for framing process tomography through statistical experimental design. It discusses the structural differences between classical and quantum estimation and nuisance parameters. The thesis must compare its own local Fisher-coordinate and known-calibration choices against that perspective. Do not claim that formulating process tomography as experimental design is new.

**Nunn et al. (2010), Optimal experiment design for quantum state tomography: Fair, precise, and minimal tomography.** [Publisher record](https://doi.org/10.1103/PhysRevA.81.042109). This supplies earlier state-tomography context for optimizing measurement design under a fixed measurement budget. Distinguish state and process tomography and distinguish optimizing a design criterion from comparing downstream finite-count reconstruction.

**Pogorelov et al. (2017), Experimental adaptive process tomography.** [Publisher record](https://doi.org/10.1103/PhysRevA.95.012302). Adaptive process tomography itself has experimental precedent. This project's question concerns the algorithm used to select measurement batches and how its quality propagates through refitting, rather than the novelty of adapting measurements at all.

## Established computational components

**Ahmed, Quijandria and Kockum (2023), Gradient-Descent Quantum Process Tomography by Learning Kraus Operators.** [Author preprint](https://arxiv.org/abs/2208.00812). The Kraus/Stiefel approach is established. Attribute the physical parametrization correctly; describe the actual count likelihood, restarts and diagnostics used here instead of implying that this project originated physical channel learning.

**Hadfield et al. (2019)** and **Farhi, Goldstone and Gutmann (2014)** supply the constrained alternating-operator and original QAOA frameworks. The implemented ordered chain is a specific circuit within that literature. Distinguish the physical one-qubit system from the fourteen-qubit optimization register.

**Shaydulin et al. (2023), Parameter Transfer for Quantum Approximate Optimization of Weighted MaxCut.** [Author preprint](https://arxiv.org/abs/2201.11785). This motivates transfer, but transferring a typical vector between MaxCut instances and reusing preceding-round angles after a tomography shortlist changes are different questions. The original negative result is therefore informative within its own setting, not a refutation of the MaxCut result.

## Why the mechanism interventions are defensible

**He et al. (2023), Alignment between initial state and mixer improves QAOA performance for constrained optimization.** [Publisher article](https://doi.org/10.1038/s41534-023-00787-5). Initial-state choice and mixer structure are already known to matter. This project tests an initialization intervention on its own measurement-design instances. It does not assume that a Dicke state is aligned with the ordered chain mixer or that state preparation is free.

**Barkoutsos et al. (2020), Improving Variational Quantum Optimization Using CVaR.** [Author preprint](https://arxiv.org/abs/1907.04769). This motivates testing a lower-tail objective when deployment needs a good sampled bitstring. The contribution here is the controlled comparison against mean-energy training, including the interaction with initialization; CVaR is not a newly proposed training rule.

**Bartschi and Eidenbenz (2019), Deterministic Preparation of Dicke States.** [Author preprint](https://arxiv.org/abs/1904.07358). Efficient preparation constructions exist, but this study initializes ideal amplitudes. The paper is relevant background, not evidence that this benchmark implemented or timed its construction.

## Classical control and theoretical interpretation

**Nemhauser, Wolsey and Fisher (1978), An analysis of approximations for maximizing submodular set functions—I.** [Publisher record](https://doi.org/10.1007/BF01588971). Combine the known greedy cardinality guarantee with the local positive-semidefinite log-determinant derivation in `THEORY_AND_CLAIMS.md`. The resulting bound concerns a frozen local utility. It does not guarantee final prediction accuracy or make a statement about quantum computational speed.

## Defensible gap statement

The thesis investigates whether an optimizer's improvement on a constrained measurement-selection surrogate survives two further stages: actual local information gain and channel reconstruction after adaptive feedback. Its contribution combines auditable paired experiments, a loss decomposition, controlled circuit/training interventions and fresh confirmation. This is a narrower and more defensible claim than inventing a new tomography, QAOA or state-preparation method.

Before submission, expand the chapter through citation chains and the university's bibliographic databases, and compare the final scope with the closest full papers. A failed keyword search is not evidence that no related work exists.
