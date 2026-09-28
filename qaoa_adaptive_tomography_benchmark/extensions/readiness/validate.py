"""Independent gate, objective, sampling, and archived-baseline checks."""
from common import *
from solver import *
from benchmark_solver import solve as archived_solve
from adaptive_kraus.design import BinaryDesign
from adaptive_kraus.qaoa import build_circuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def main():
    rng=np.random.default_rng(282611)
    errors=[];cases=0
    for n,b in [(8,3),(14,3)]:
        model=BinaryDesign(-rng.uniform(.1,1,n),np.triu(rng.uniform(0,.3,(n,n)),1),b,np.arange(n))
        initial=np.zeros(n);initial[rng.choice(n,b,False)]=1
        h,j,_=model.ising()
        for variant in VARIANTS:
            c=make_circuit(model,initial,variant)
            angles=rng.uniform(-1.,2.,(3,2))
            many=probabilities_many(c,angles)
            for k,(g,be) in enumerate(angles):
                qc=QuantumCircuit(n)
                for i in range(n):
                    qc.rz(2*g*h[i]/c.scale,i)
                    for other in range(i+1,n):qc.rzz(2*g*j[i,other]/c.scale,i,other)
                for parity in (0,1):
                    for i in range(parity,n-1,2):
                        qc.rxx(2*be,i,i+1);qc.ryy(2*be,i,i+1)
                v=np.zeros(2**n,complex);v[c.integers]=c.initial
                full=Statevector(v).evolve(qc).probabilities()
                expected=c.probabilities([g,be])
                errors.extend([float(np.max(abs(full[c.integers]-expected))),float(np.max(abs(many[k]-expected)))])
                assert abs(full[c.integers].sum()-1)<1e-11
                cases+=1
            r=run_solver(model,951,variant)
            assert r['total_shots']==128*r['objective_evaluations']+256
            assert r['objective_evaluations']<=80 and sum(r['selected_bits'])==b
            integer=sum(int(x)<<i for i,x in enumerate(r['selected_bits']))
            assert integer in r['final_sample_integers']
            assert abs(sum(r['probabilities'])-1)<1e-12
        old=archived_solve(model,1,80,951)
        new=run_solver(model,951)
        assert old['angles']==new['angles']
        assert old['final_sample_integers']==new['final_sample_integers']
        assert old['selected']['256']['bits']==new['selected_bits']
        circuit=make_circuit(model,initial,'baseline')
        qc,params,_=build_circuit(model,1,initial)
        a=[.47,.83]
        full=Statevector.from_instruction(qc.assign_parameters(dict(zip(params,a)))).probabilities()
        errors.append(float(np.max(abs(full[circuit.integers]-circuit.probabilities(a)))))
    assert max(errors)<1e-11
    assert abs(tail_mean([0,10],[.1,.9],.2)-5)<1e-12
    assert abs(sampled_tail(np.array([0.,10.,20.]),.5)-10/3)<1e-12
    assert np.allclose(best_distribution(np.array([0.,1.]),np.array([.25,.75]),2),[.4375,.5625])
    # Ties follow the documented integer-index tie break.
    assert np.allclose(best_distribution(np.array([0.,0.]),np.array([.25,.75]),2),[.4375,.5625])
    assert np.allclose(best_distribution(np.array([1.,0.]),np.array([0.,1.]),256),[0.,1.])
    result=dict(status='passed',qiskit_probability_cases=cases,maximum_probability_error=max(errors),
        archived_baseline_exact_reproductions=2,batched_landscapes='passed',
        fractional_CVaR='passed',best_sample_order_statistics='passed',sample_only_and_cost='passed',utc=now())
    write(HERE/'validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
