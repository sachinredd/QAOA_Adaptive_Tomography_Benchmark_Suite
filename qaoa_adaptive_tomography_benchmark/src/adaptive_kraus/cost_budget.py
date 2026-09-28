"""Exact application budgets and an outcome-independent prior design."""
import numpy as np
from .experiments import make_pool, pilot_ids, Observation
from .channels import gksl_channel
from .design import build_design, batch_gain

def prospective_shots(pool, config):
    cap = config.get("applications_per_setting", 0)
    if not cap:
        return config["round_shots"]
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1 or any(cap % e.steps for e in pool):
        raise ValueError("Budget must be a positive integer divisible by every candidate step")
    return np.asarray([cap//e.steps for e in pool], dtype=np.int64)

def measure_selected(oracle, pool, selected, shots):
    if np.ndim(shots) == 0:
        return oracle.measure([pool[i] for i in selected], int(shots))
    out = []
    for i in selected:
        out.extend(oracle.measure([pool[i]], int(shots[i])))
    return out

def prior_design(spec):
    """Greedily maximize mean local log-det gain on independent prior draws.

    Only the distribution, known readout, and budget enter. No test channel,
    acquired counts, fitted test model, or held-out score is accepted.
    Synthetic Observation counts are placeholders: build_design consumes only
    their setting IDs and shot totals when accumulating expected information.
    """
    from .study_protocol import family_parameters
    c=spec['base'];prior=spec['fixed_prior']
    parameters=[dict(family=f,instance=i,channel=family_parameters(f,i,1.,prior['seed'],'broad_v1'))
                for f in spec['families'] for i in range(prior['instances'])]
    channels=[gksl_channel(**p['channel']) for p in parameters]
    schedules={};gains={}
    for profile in spec['noise_profiles']:
        pool=make_pool(1,c['pool_steps'],profile['assumed_readout'])
        shots=prospective_shots(pool,c)
        obs=[Observation(i,(0,c['pilot_shots']),c['pilot_shots']) for i in pilot_ids(pool)]
        schedule=[];round_gains=[]
        for _ in range(c['rounds']):
            designs=[build_design(k,pool,obs,shots,c['ridge'],c['fisher_probability_floor']) for k in channels]
            chosen=[];remaining=list(range(len(pool)))
            for _ in range(c['batch_size']):
                values=[float(np.mean([batch_gain(d,chosen+[i]) for d in designs])) for i in remaining]
                pick=remaining[int(np.argmax(values))]
                chosen.append(pick);remaining.remove(pick)
            round_gains.append(float(np.mean([batch_gain(d,chosen) for d in designs])))
            schedule.append(chosen)
            obs.extend(Observation(i,(0,int(shots[i])),int(shots[i])) for i in chosen)
        schedules[profile['id']]=schedule;gains[profile['id']]=round_gains
    return dict(seed=prior['seed'],parameters=parameters,schedules=schedules,mean_local_gains=gains,
                note="Offline greedy average-logdet schedule on separate prior draws; not globally optimal and not adapted to test data.")
