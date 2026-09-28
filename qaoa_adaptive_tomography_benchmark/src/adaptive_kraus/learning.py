"""Count likelihood and analytic Stiefel descent through repeated channels."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .channels import blocks, apply_channel, adjoint_channel
from .experiments import aggregate_counts
from .manifold import project_tangent, cayley_step, inner


class CountObjective:
    """Per-shot multinomial negative log likelihood, constants omitted.

    Complex-gradient convention: dL = Re Tr(G^dagger dK). A hard floor
    avoids log(0); below it the derivative is zero. Near-identity full-rank
    initialisation avoids exact zero probabilities in the supplied runs.
    """
    def __init__(self, pool, observations, probability_floor=1e-12):
        experiments, self.counts = aggregate_counts(observations, pool)
        self.rho = np.asarray([e.rho for e in experiments])
        self.effects = np.asarray([e.effects for e in experiments])
        self.steps = np.asarray([e.steps for e in experiments])
        self.total_shots = int(self.counts.sum())
        self.floor = float(probability_floor)
        self.evaluations = 0
        self.gradient_evaluations = 0

    def value_grad(self, stack, gradient=True):
        self.evaluations += 1
        self.gradient_evaluations += int(gradient)
        k = blocks(stack)
        states = [self.rho]
        for _ in range(self.steps.max()):
            states.append(apply_channel(k, states[-1]))
        selected = np.asarray(states)[self.steps, np.arange(len(self.steps))]
        p = np.einsum("soij,sji->so", self.effects, selected).real
        safe = np.maximum(p, self.floor)
        value = float(-np.sum(self.counts*np.log(safe))/self.total_shots)
        if not gradient:
            return value, None
        coeff = np.where(p > self.floor, -self.counts/safe/self.total_shots, 0.)
        source = np.einsum("so,soij->sij", coeff, self.effects)
        adjoint = np.zeros_like(self.rho)
        grad = np.zeros_like(k)
        for t in range(len(states)-1, 0, -1):
            mask = self.steps == t
            adjoint[mask] += source[mask]
            grad += 2*np.einsum("nab,kbc,ncd->kad", adjoint, k, states[t-1], optimize=True)
            adjoint = adjoint_channel(k, adjoint)
        return value, grad.reshape(stack.shape)

    def value(self, stack):
        return self.value_grad(stack, gradient=False)[0]


@dataclass
class FitResult:
    stack: np.ndarray
    history: list[dict]
    diagnostics: dict


def fit_channel(pool, observations, initial, iterations=200, learning_rate=1., tolerance=1e-7):
    """Cayley descent with Armijo backtracking; warm-start from previous round."""
    if iterations < 1 or learning_rate <= 0 or tolerance < 0:
        raise ValueError("Invalid fit budget")
    objective = CountObjective(pool, observations)
    stack = np.asarray(initial, complex).copy()
    if not np.allclose(stack.conj().T@stack, np.eye(stack.shape[1]), atol=1e-8):
        raise ValueError("Initial Kraus stack must be an isometry")
    history = []
    started = perf_counter()
    status = "iteration_limit"
    for it in range(iterations):
        value, g = objective.value_grad(stack)
        grad_norm = float(np.linalg.norm(project_tangent(stack, g)))
        if grad_norm < tolerance:
            status = "gradient_tolerance"
            break
        direction = -(g-stack@g.conj().T@stack)
        slope = inner(g, direction)
        step = learning_rate
        accepted = False
        for _ in range(30):
            proposal = cayley_step(stack, g, step)
            trial = objective.value(proposal)
            if np.isfinite(trial) and trial <= value+1e-4*step*slope:
                stack, value, accepted = proposal, trial, True
                break
            step *= .5
        history.append(dict(iteration=it, nll=value, gradient_norm=grad_norm,
                            step=step, accepted=accepted))
        if not accepted:
            status = "line_search_stalled"
            break
    return FitResult(stack, history, dict(status=status, seconds=perf_counter()-started,
        nll=objective.value(stack), evaluations=objective.evaluations,
        gradient_evaluations=objective.gradient_evaluations,
        tp_error=float(np.linalg.norm(stack.conj().T@stack-np.eye(stack.shape[1])))))
