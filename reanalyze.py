"""Reviewer-driven strengthened comparator, added after the initial run."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bounds import count_matched_bounds, certificate, joint_bounds

R=Path(__file__).resolve().parent
d=pd.read_csv(R/'comparisons.csv'); cache={};new=[]
for row in d.to_dict('records'):
    key=(row['dataset'],row['seed'])
    if key not in cache:cache[key]=dict(np.load(R/'prediction_records'/f'{key[0]}_{key[1]:02d}.npz'))
    z=cache[key];obs=z['y'].copy();idx=[.2,.5,.8].index(row['missing_fraction'])
    obs[z[f"mask_{row['mechanism']}_{idx}"]]=-1
    lo,hi=count_matched_bounds(z[row['model_a']],z[row['model_b']],obs,row['qlo'],row['qhi'])
    assert float(lo)<=row['lower']+1e-14 and float(hi)>=row['upper']-1e-14
    row.update(count_matched_lower=float(lo),count_matched_upper=float(hi),
               count_matched_width=float(hi-lo),count_matched_certificate=certificate(lo,hi))
    new.append(row)
pd.DataFrame(new).to_csv(R/'comparisons.csv',index=False)
summary=[]
for keys,g in pd.DataFrame(new).groupby(['dataset','regime','missing_fraction']):
    summary.append(dict(dataset=keys[0],regime=keys[1],missing_fraction=keys[2],n=len(g),
       mean_joint_width=g.width.mean(),mean_independent_width=g.independent_width.mean(),
       mean_count_matched_width=g.count_matched_width.mean(),
       joint_certificates=int(g.joint_certificate.ne(0).sum()),
       independent_certificates=int(g.independent_certificate.ne(0).sum()),
       count_matched_certificates=int(g.count_matched_certificate.ne(0).sum()),
       complete_case_wrong_signs=int(g.complete_case_wrong_sign.sum())))
pd.DataFrame(summary).to_csv(R/'summary.csv',index=False)
a=np.array([1,1,1,0,1,0,1,0]);b=np.array([1,0,1,0,0,1,0,0]);y=np.array([1,1,-1,-1,-1,0,-1,-1])
r=joint_bounds(a,b,y);l,h=count_matched_bounds(a,b,y)
(R/'worked_example.json').write_text(json.dumps(dict(a=a.tolist(),b=b.tolist(),y=y.tolist(),
    joint={k:str(v) if k in ['lower','upper'] else v for k,v in r.items()},
    count_matched=[str(l),str(h)]),indent=2))
print('Added strengthened comparator to',len(new),'rows')
