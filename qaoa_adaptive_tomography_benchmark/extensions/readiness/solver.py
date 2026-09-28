"""Controlled initialization/objective interventions; no exact solution injection."""
from common import *
from benchmark_solver import Circuit, basis
from scipy.optimize import minimize
from time import perf_counter

VARIANTS = {
    'baseline': ('swept', 1.),
    'cvar': ('swept', .2),
    'dicke': ('dicke', 1.),
    'dicke_cvar': ('dicke', .2),
}

def tail_mean(values, probabilities, alpha):
    """Lower-tail CVaR, with fractional mass at the alpha quantile."""
    order = np.argsort(values, kind='stable')
    p = np.asarray(probabilities)[order]
    mass = np.clip(alpha - (np.cumsum(p)-p), 0., p)
    return float(mass @ np.asarray(values)[order] / alpha)

def sampled_tail(values, alpha):
    if alpha == 1.:
        return float(np.mean(values))
    s = np.sort(values); amount = alpha*len(s); whole = int(amount)
    return float((s[:whole].sum()+(amount-whole)*s[whole])/amount)

def best_distribution(energies, probabilities, count):
    """Distribution of best of N IID samples; ties use ascending basis index."""
    order = np.argsort(energies, kind='stable')
    tail = np.r_[np.cumsum(np.asarray(probabilities)[order][::-1])[::-1], 0.]
    tail = np.clip(tail, 0., 1.); tail[0] = 1.
    mass = tail[:-1]**count-tail[1:]**count
    result = np.zeros(len(order)); result[order] = mass
    return result

def make_circuit(model, initial, variant, depth=1):
    c = Circuit(model, depth, initial)
    if VARIANTS[variant][0] == 'dicke':
        # This ideal-state intervention REPLACES the initial basis + two sweeps.
        # Preparation is not free on hardware; no gate-cost advantage is claimed.
        c.initial[:] = 1/np.sqrt(len(c.initial))
    return c

def run_solver(model, seed, variant='baseline', exact=False, cap=80, shots=128, final_shots=256):
    start = perf_counter(); rng = np.random.default_rng(seed)
    initial = np.zeros(model.n_bits)
    initial[rng.choice(model.n_bits, model.batch_size, replace=False)] = 1
    start_angles = np.r_[rng.uniform(0., 1., 1), rng.uniform(.1, 1.2, 1)]
    c = make_circuit(model, initial, variant)
    alpha = VARIANTS[variant][1]
    best_value, best_angles = np.inf, None
    trace = []
    def objective(theta):
        nonlocal best_value, best_angles
        p = c.probabilities(theta)
        true_value = tail_mean(c.diagonal, p, alpha) if alpha < 1 else float(p@c.diagonal)
        if exact:
            value = true_value
        else:
            draws = rng.choice(len(p), size=shots, p=p)
            value = sampled_tail(c.diagonal[draws], alpha)
        trace.append(dict(angles=np.asarray(theta).tolist(), observed=value, exact=true_value))
        if value < best_value:
            best_value, best_angles = value, np.asarray(theta).copy()
        return value
    result = minimize(objective, start_angles, method='COBYLA',
                      options={'maxiter':cap, 'rhobeg':.35, 'tol':1e-3})
    p = c.probabilities(best_angles)
    draws = rng.choice(len(p), size=final_shots, p=p)
    winner = min(np.unique(draws), key=lambda i:(c.energy[i], i))
    best_mass = best_distribution(c.energy, p, final_shots)
    span = float(np.ptp(c.energy)); denom = span if span > 1e-12 else 1.
    gap = (c.energy-c.energy.min())/denom
    chosen_exact = tail_mean(c.diagonal,p,alpha) if alpha < 1 else float(p@c.diagonal)
    return dict(variant=variant, exact_training=exact, initial_bits=initial.astype(int).tolist(),
        start_angles=start_angles.tolist(), angles=best_angles.tolist(), trace=trace,
        probabilities=p.tolist(), final_sample_integers=c.integers[draws].tolist(),
        selected_bits=c.bits[winner].astype(int).tolist(), selected_index=int(winner),
        selected_energy=float(c.energy[winner]), sampled_regret=float(gap[winner]),
        expected_regret=float(p@gap), expected_best_regret=float(best_mass@gap),
        success_probability=float(1-(1-p[c.energy<=c.energy.min()+1e-9].sum())**final_shots),
        optimal=bool(c.energy[winner]<=c.energy.min()+1e-9),
        effective_support=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),
        selection_noise_penalty=float(chosen_exact-min(t['exact'] for t in trace)),
        objective_evaluations=len(trace), objective_shots=0 if exact else len(trace)*shots,
        final_shots=final_shots, total_shots=(0 if exact else len(trace)*shots)+final_shots,
        offline_exact_objective_evaluations=len(trace) if exact else 0,
        optimizer_status=int(result.status), seconds=perf_counter()-start)

def probabilities_many(c, angles):
    """Batched depth-one landscape; same ordered gates as Circuit.probabilities."""
    state = np.repeat(c.initial[None,:],len(angles),axis=0)
    state *= np.exp(-1j*np.asarray(angles)[:,0,None]*c.diagonal)
    co=np.cos(2*np.asarray(angles)[:,1,None]); si=-1j*np.sin(2*np.asarray(angles)[:,1,None])
    for a,b in c.edges:
        x,y=state[:,a].copy(),state[:,b].copy()
        state[:,a],state[:,b]=co*x+si*y,si*x+co*y
    p=abs(state)**2
    return p/p.sum(axis=1,keepdims=True)
