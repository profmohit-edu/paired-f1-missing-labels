"""Reproducible demonstration on three bundled scikit-learn datasets.

Missing labels are induced after fitting. Exact-count and centered count-band
regimes use hidden truth deliberately and are oracle sensitivity analyses.
"""
import csv,hashlib,itertools,json,math,platform,time
from pathlib import Path
import numpy as np
import scipy, sklearn
from sklearn.datasets import load_breast_cancer,load_wine,load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from bounds import joint_bounds,independent_bounds,certificate,f1,from_counts

ROOT=Path(__file__).resolve().parent


def dumpcsv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    start=time.perf_counter();rows=[]; fits=[]; provenance=[]
    pred_dir=ROOT/'prediction_records';pred_dir.mkdir(exist_ok=True)
    for name,loader in [('breast_cancer',load_breast_cancer),('wine',load_wine),('digits',load_digits)]:
        data=loader();X=data.data
        y=(data.target==0).astype(int) if name!='digits' else (data.target>=5).astype(int)
        h=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
        provenance.append(dict(dataset=name,n=len(y),features=X.shape[1],positives=int(y.sum()),array_sha256=h,
                               positive_definition='original class 0' if name!='digits' else 'digit >= 5',
                               loader=loader.__name__,source='bundled scikit-learn dataset; no feature-data download'))
        for seed in range(30):
            train,test=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=seed)
            yt=y[test]
            models=[('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=seed))),
                    ('tree',DecisionTreeClassifier(max_depth=3,random_state=seed)),
                    ('forest',RandomForestClassifier(n_estimators=100,max_depth=5,n_jobs=1,random_state=seed))]
            predictions={}
            for label,model in models:
                t=time.perf_counter();model.fit(X[train],y[train]);p=model.predict(X[test]);predictions[label]=p
                fits.append(dict(dataset=name,seed=seed,model=label,n_train=len(train),n_test=len(test),
                                 full_f1=float(f1(int((p*yt).sum()),int(p.sum()),int(yt.sum()))),
                                 fit_predict_seconds=time.perf_counter()-t))
            save=dict(test_indices=test,train_indices=train,y=yt,**predictions)
            for mechidx,mech in enumerate(['uniform','prediction_weighted','outcome_weighted']):
                for rateidx,rate in enumerate([.2,.5,.8]):
                    rng=np.random.default_rng(np.random.SeedSequence([20261022,seed,mechidx,rateidx]))
                    weights=np.ones(len(test))
                    if mech=='prediction_weighted':weights=1+3*predictions['logistic']
                    if mech=='outcome_weighted':weights=1+3*yt
                    m=int(math.floor(rate*len(test)))
                    u=rng.choice(len(test),size=m,replace=False,p=weights/weights.sum())
                    mask=np.zeros(len(test),dtype=bool);mask[u]=True
                    obs=yt.copy();obs[mask]=-1
                    save[f'mask_{mech}_{rateidx}']=mask
                    q=int(yt[mask].sum());band=math.ceil(.1*m)
                    regimes=[('unrestricted',0,m),('oracle_band',max(0,q-band),min(m,q+band)),('oracle_exact',q,q)]
                    for aa,bb in itertools.combinations(predictions,2):
                        a,b=predictions[aa],predictions[bb]
                        true=f1(int((a*yt).sum()),int(a.sum()),int(yt.sum()))-f1(int((b*yt).sum()),int(b.sum()),int(yt.sum()))
                        keep=~mask
                        cc=f1(int((a[keep]*yt[keep]).sum()),int(a[keep].sum()),int(yt[keep].sum()))-f1(int((b[keep]*yt[keep]).sum()),int(b[keep].sum()),int(yt[keep].sum()))
                        for regime,lo,hi in regimes:
                            tic=time.perf_counter();r=joint_bounds(a,b,obs,lo,hi);elapsed=time.perf_counter()-tic
                            il,iu=independent_bounds(a,b,obs,lo,hi)
                            assert il<=r['lower']<=true<=r['upper']<=iu
                            jc=certificate(r['lower'],r['upper']);ic=certificate(il,iu)
                            assert not jc or jc*(true)>0
                            assert not ic or ic==jc
                            rows.append(dict(dataset=name,seed=seed,model_a=aa,model_b=bb,mechanism=mech,
                                             missing_fraction=rate,missing_n=m,q_true=q,qlo=lo,qhi=hi,regime=regime,
                                             true_difference=float(true),complete_case_difference=float(cc),
                                             complete_case_wrong_sign=int(cc*true<0),
                                             lower=float(r['lower']),upper=float(r['upper']),
                                             independent_lower=float(il),independent_upper=float(iu),
                                             width=float(r['upper']-r['lower']),independent_width=float(iu-il),
                                             joint_certificate=jc,independent_certificate=ic,
                                             solver_seconds=elapsed,
                                             lower_witness=json.dumps(r['lower_witness'],sort_keys=True),
                                             upper_witness=json.dumps(r['upper_witness'],sort_keys=True)))
            np.savez_compressed(pred_dir/f'{name}_{seed:02d}.npz',**save)
        print('completed',name,flush=True)
    dumpcsv(ROOT/'comparisons.csv',rows);dumpcsv(ROOT/'fits.csv',fits)
    (ROOT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    timings=[]
    for m in [100,1000,10000,100000]:
        c=[3*m//10,m//10,15*m//100,45*m//100]
        for repetition in range(10):
            tic=time.perf_counter();from_counts(c,pa=c[2]+c[3]+50,pb=c[1]+c[3]+60,ta=40,tb=45,s=100)
            timings.append(dict(m=m,repetition=repetition,seconds=time.perf_counter()-tic))
    dumpcsv(ROOT/'scaling.csv',timings)
    (ROOT/'execution.json').write_text(json.dumps(dict(comparisons=len(rows),model_fits=len(fits),
        seconds=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        scikit_learn=sklearn.__version__,platform=platform.platform(),seed_range='0 through 29 inclusive',
        no_population_inference=True),indent=2))

if __name__=='__main__':main()
