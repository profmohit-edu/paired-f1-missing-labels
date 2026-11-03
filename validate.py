"""Independent exhaustive label enumeration and linear programming checks."""
import itertools, json, time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from bounds import joint_bounds, summarize, f1, from_counts, independent_bounds


def exhaustive(a,b,y,qlo,qhi):
    u=np.flatnonzero(y == -1)
    values=[]
    for bits in itertools.product([0,1], repeat=len(u)):
        if not qlo <= sum(bits) <= qhi:
            continue
        yy=y.copy();yy[u]=bits
        values.append(f1(int((a*yy).sum()),int(a.sum()),int(yy.sum()))-
                      f1(int((b*yy).sum()),int(b.sum()),int(yy.sum())))
    return min(values),max(values)


def run(out):
    start=time.perf_counter(); rng=np.random.default_rng(22026)
    count=0; maxerr=0.; lpchecks=0
    # Entire paired-prediction space through n=3, all observed/missing labels.
    for n in range(0,4):
        for av in itertools.product([0,1],repeat=n):
            for bv in itertools.product([0,1],repeat=n):
                for yv in itertools.product([-1,0,1],repeat=n):
                    a,b,y=[np.array(v,dtype=int) for v in (av,bv,yv)]
                    m=int((y==-1).sum())
                    for lo in range(m+1):
                        for hi in range(lo,m+1):
                            r=joint_bounds(a,b,y,lo,hi); e=exhaustive(a,b,y,lo,hi)
                            assert (r['lower'],r['upper']) == e,(a,b,y,lo,hi,r,e)
                            il,iu=independent_bounds(a,b,y,lo,hi)
                            assert il <= r['lower'] <= r['upper'] <= iu
                            count+=1
    for _ in range(1000):
        n=int(rng.integers(1,13));a=rng.integers(0,2,n);b=rng.integers(0,2,n)
        y=rng.choice([-1,0,1],n,p=[.65,.175,.175]);m=int((y==-1).sum())
        lo=int(rng.integers(m+1));hi=int(rng.integers(lo,m+1))
        r=joint_bounds(a,b,y,lo,hi);e=exhaustive(a,b,y,lo,hi)
        assert (r['lower'],r['upper'])==e
        swap=joint_bounds(b,a,y,lo,hi)
        assert (r['lower'],r['upper']) == (-swap['upper'],-swap['lower'])
        for side in ['lower','upper']:
            w=r[side+'_witness'];d=summarize(a,b,y);x=w['x'];q=w['q']
            assert sum(x)==q and lo<=q<=hi
            assert all(0<=v<=c for v,c in zip(x,d['caps']))
            v=f1(d['ta']+x[2]+x[3],d['pa'],d['s']+q)-f1(d['tb']+x[1]+x[3],d['pb'],d['s']+q)
            assert v==r[side]
        count+=1
    # A generic LP baseline independently solves every fixed-count problem.
    for _ in range(200):
        n=int(rng.integers(10,101));a=rng.integers(0,2,n);b=rng.integers(0,2,n)
        y=rng.choice([-1,0,1],n);d=summarize(a,b,y);m=sum(d['caps'])
        q=int(rng.integers(m+1));p=d['s']+q
        ca=2/(d['pa']+p) if d['pa']+p else 0
        cb=2/(d['pb']+p) if d['pb']+p else 0
        c=np.array([0,-cb,ca,ca-cb]);base=ca*d['ta']-cb*d['tb']
        r=from_counts(**d,qlo=q,qhi=q)
        for sign,side in [(1,'lower'),(-1,'upper')]:
            lp=linprog(sign*c,A_eq=[[1,1,1,1]],b_eq=[q],bounds=[(0,v) for v in d['caps']],method='highs')
            assert lp.success
            err=abs(base+sign*lp.fun-float(r[side]));maxerr=max(maxerr,err)
            assert err<1e-10
            lpchecks+=1
    result=dict(exhaustive_instances=count,lp_optimizations=lpchecks,
                max_lp_absolute_error=maxerr,seconds=time.perf_counter()-start,
                status='PASS',definition='F1=0 when predicted-positive count plus actual-positive count is zero')
    Path(out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__':
    run(Path(__file__).resolve().parent/'validation.json')
