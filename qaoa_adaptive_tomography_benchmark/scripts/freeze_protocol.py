from pathlib import Path
import sys,json,hashlib,platform
from datetime import datetime,timezone
from importlib.metadata import version
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from adaptive_kraus.study_sampling import RANGES
p={
 'name':'QAOA parameter reuse and scaling benchmark', 'frozen_utc':datetime.now(timezone.utc).isoformat(),
 'stage':'prospective protocol before new benchmark outcomes',
 'development':'Six archived greedy trajectories: one per family, instance 0, replicate 4001, calibrated readout, rounds 0,1,2 from the matched-cost archive. Used for implementation checks and trajectory configuration selection only.',
 'evaluation':{'channels_per_family':5,'families':list(RANGES),'channel_seed':27190317,'acquisition_seed':27190329,'solver_seeds':[6101,6152,6203],'readout':[.03,.06],'scaffold':'Fresh full-pool greedy trajectories; saved models after pilot and first two adaptive rounds. One acquisition repetition per physical channel.'},
 'selection_problems':[[8,3],[10,3],[12,3],[14,3],[14,5]],
 'qaoa':{'depths':[1,2],'evaluation_caps':[20,80],'initializations':['cold','transfer'],'objective_shots':128,'final_shots':256,'prefixes':[8,32,256],'mixer':'Same ordered chain XY circuit and two fixed initialization sweeps as archived implementation','transfer':'Only angles transfer between consecutive frozen checkpoints within the same channel, size, depth, cap and solver seed; the feasible initial bitstring and random draws are paired with cold initialization. Round 0 is identical. No later checkpoint or reference probability is used.'},
 'classical':{'exact':'Enumerate feasible quadratic objective for reference; enumerate true local utility within shortlist separately','random_budgets':[8,32,256],'swap_budgets':[20,80,256],'greedy':'Full-pool local information gain, shown separately because search domain and objective differ'},
 'primary':'Transferred minus cold normalized expected quadratic energy regret at n=14,b=3,p=1,cap=20, averaging solver seeds and checkpoints 1 and 2 within each fresh physical channel.',
 'normalization':'(energy minus feasible minimum)/(feasible maximum minus feasible minimum); zero for exactly constant objectives. Exact distribution expectation is an offline simulator diagnostic, not fed into optimization.',
 'interval':'95 percent equal-family stratified t interval using 30 independent channel means, five per family; Welch-Satterthwaite df. All other contrasts are descriptive, with no multiplicity adjustment.',
 'secondary':'Sampled best-of-8,32,256 energy regret and success; true local gain; approximation losses; size/depth/budget effects; fresh end-to-end prediction error and resource totals.',
 'application_cost':'408 unknown-channel applications per selected setting, as archived. Three-setting batches cost 1224; five-setting frozen audits cost 2040 and are a separate size track, not an equal-cost comparison.',
 'trajectory_stage':{'channels':'Same 30 fresh channels, common pilot and named observation streams across policies','policies':['greedy','fixed_prior','exact_qubo','random_qubo','swap','qaoa_cold','qaoa_transfer'],'shortlist':14,'batch':3,'rounds':3,'total_channel_applications':4056,'configuration_choice':'Choose depth and cap by lowest pooled cold/transfer development mean expected normalized energy regret at n=14,b=3 over checkpoints 1,2 and all three seeds. Ties within 1e-12 favor lower cap then depth. Both QAOA policies use the selected configuration. Test data cannot change the choice.','qaoa_seed':7101,'fixed_prior':'Reuse calibrated schedule frozen in the earlier matched-cost study with independently sampled prior channels; no new prior tuning','classical_random_samples':256,'swap_evaluations':256,'fit':{'candidates':4,'iterations':2000,'tolerance':1e-6},'offline_check':{'continuation_iterations':400,'fresh_iterations':400,'material_nll_gain':1e-5,'gradient_threshold':1e-4}},
 'analysis_limits':'One calibrated readout profile and one acquisition repetition per channel. Frozen checkpoints come from greedy, so the solver study measures those input distributions; fresh policy trajectories evaluate feedback separately. All simulations ideal on QAOA register. Simulator speed is not quantum hardware speed.',
 'stopping':'Fixed design; retain every completed planned result including fitting flags; no outcome-based extension. Record technical repairs and reruns. No historical pooling.',
 'environment':{'python':platform.python_version(),**{x:version(x) for x in ['numpy','scipy','pandas','qiskit','qiskit-aer']}}
}
path=B/'protocol/PROTOCOL.json'
if path.exists():raise SystemExit('Protocol already frozen')
path.write_text(json.dumps(p,indent=2)+'\n')
print('Protocol frozen',hashlib.sha256(path.read_bytes()).hexdigest())
