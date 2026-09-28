from pathlib import Path
import sys,json
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'src'))
from adaptive_kraus.design import BinaryDesign
from adaptive_kraus.qaoa import build_circuit
from benchmark_solver import Circuit,solve,random_search,swap_search
from qiskit.quantum_info import Statevector
rng=np.random.default_rng(97031);errors=[]
for n,b in [(8,3),(10,3),(12,3),(14,3),(14,5)]:
 model=BinaryDesign(-rng.uniform(.1,1,n),np.triu(rng.uniform(0,.3,(n,n)),1),b,np.arange(n))
 for depth in [1,2]:
  initial=np.zeros(n);initial[rng.choice(n,b,False)]=1
  circuit=Circuit(model,depth,initial);qc,params,_=build_circuit(model,depth,initial)
  for j in range(2):
   angles=rng.uniform(-1,2,2*depth)
   full=Statevector.from_instruction(qc.assign_parameters(dict(zip(params,angles)))).probabilities()
   compact=circuit.probabilities(angles)
   error=float(abs(full[circuit.integers]-compact).max());errors.append(error)
   assert error<1e-11
   assert abs(full[circuit.integers].sum()-1)<1e-11
 result=solve(model,1,20,721)
 again=solve(model,1,20,721)
 assert result['angles']==again['angles'] and result['final_sample_integers']==again['final_sample_integers']
 chosen=result['selected']['256']['bits'];integer=sum(int(v)<<i for i,v in enumerate(chosen))
 assert integer in result['final_sample_integers'] and sum(chosen)==b
 assert result['objective_evaluations']<=20 and result['objective_shots']==128*result['objective_evaluations']
 for f in [random_search,swap_search]:
  q,metadata=f(model,511,80)
  assert q.sum()==b and metadata['candidate_evaluations']==80
 print('Validated',n,b,flush=True)
report={'probability_vector_comparisons':len(errors),'maximum_probability_error':max(errors),'reproducibility':'passed','sampled_output_only':'passed','budget_accounting':'passed','classical_feasibility':'passed'}
(B/'results/solver_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
