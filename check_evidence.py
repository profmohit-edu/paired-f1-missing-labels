"""Audit exported witnesses independently against reconstructed label vectors."""
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent
d=pd.read_csv(R/'comparisons.csv');cache={};witnesses=0
def score(a,y):
    den=int(a.sum()+y.sum())
    return Fraction(2*int((a*y).sum()),den) if den else Fraction(0)
for row in d.to_dict('records'):
    key=row['dataset'],row['seed']
    if key not in cache:cache[key]=dict(np.load(R/'prediction_records'/f'{key[0]}_{key[1]:02d}.npz'))
    z=cache[key];a=z[row['model_a']];b=z[row['model_b']];truth=z['y']
    mask=z[f"mask_{row['mechanism']}_{[.2,.5,.8].index(row['missing_fraction'])}"]
    assert len(set(z['test_indices']) & set(z['train_indices']))==0
    true=score(a,truth)-score(b,truth)
    assert abs(float(true)-row['true_difference'])<1e-14
    assert row['qlo']<=int(truth[mask].sum())<=row['qhi']
    for side in ['lower','upper']:
        w=json.loads(row[side+'_witness']);yy=truth.copy();yy[mask]=0
        assert sum(w['x'])==w['q'] and row['qlo']<=w['q']<=row['qhi']
        for cell in range(4):
            ids=np.flatnonzero(mask & (2*a+b==cell));n=w['x'][cell]
            assert 0<=n<=len(ids);yy[ids[:n]]=1
        val=score(a,yy)-score(b,yy)
        assert abs(float(val)-row[side])<1e-14
        witnesses+=1
    assert row['lower']<=float(true)+1e-14<=row['upper']+2e-14
    assert not row['joint_certificate'] or row['joint_certificate']*true>0
    assert not row['count_matched_certificate'] or row['joint_certificate']==row['count_matched_certificate']
    assert row['count_matched_lower']<=row['lower']+1e-14
    assert row['upper']<=row['count_matched_upper']+1e-14
assert len(d)==7290 and len(cache)==90
assert not d.duplicated(['dataset','seed','model_a','model_b','mechanism','missing_fraction','regime']).any()
for _,g in d.groupby(['dataset','seed','model_a','model_b','mechanism','missing_fraction']):
    h=g.set_index('regime')
    assert h.loc['unrestricted','lower']<=h.loc['oracle_band','lower']+1e-14<=h.loc['oracle_exact','lower']+2e-14
    assert h.loc['unrestricted','upper']+2e-14>=h.loc['oracle_band','upper']+1e-14>=h.loc['oracle_exact','upper']
out=dict(status='PASS',case_rows=len(d),reconstructed_witnesses=witnesses,prediction_records=len(cache),
         checked=['train/test disjointness','truth feasibility','endpoint witness scores',
                  'strict certificate signs','comparator containment','count-regime nesting','duplicate rows'])
(R/'evidence_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
