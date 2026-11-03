"""Generic four-variable LP envelope versus the specialized exact sweep."""
import time,csv
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from bounds import from_counts

R=Path(__file__).resolve().parents[1]/'results'; rows=[]
for m in [10,25,50,100,200]:
    for rep in range(5):
        rng=np.random.default_rng([20261022,m,rep])
        c=rng.multinomial(m,[.3,.1,.15,.45]);pa=int(c[2]+c[3]+50);pb=int(c[1]+c[3]+60)
        tic=time.perf_counter();r=from_counts(c,pa,pb,40,45,100);special=time.perf_counter()-tic
        tic=time.perf_counter();lo=1.;hi=-1.
        for q in range(m+1):
            ca=2/(pa+100+q);cb=2/(pb+100+q);coef=np.array([0,-cb,ca,ca-cb]);base=40*ca-45*cb
            for sign in [1,-1]:
                sol=linprog(sign*coef,A_eq=[[1,1,1,1]],b_eq=[q],bounds=[(0,int(v)) for v in c],method='highs')
                assert sol.success
                val=base+sign*sol.fun
                if sign==1:lo=min(lo,val)
                else:hi=max(hi,val)
        generic=time.perf_counter()-tic
        err=max(abs(lo-float(r['lower'])),abs(hi-float(r['upper'])))
        assert err<1e-10
        rows.append(dict(m=m,repetition=rep,specialized_seconds=special,generic_lp_seconds=generic,max_absolute_error=err))
with open(R/'lp_benchmark.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('PASS',len(rows),'whole-envelope LP comparisons')
