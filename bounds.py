"""Sharp finite-pool F1 difference bounds; no label model is fitted."""
from fractions import Fraction
import numpy as np


def f1(tp, predicted, positives):
    den = int(predicted) + int(positives)
    return Fraction(2 * int(tp), den) if den else Fraction(0)


def summarize(a, b, y_observed):
    a, b, y = [np.asarray(v, dtype=int) for v in (a, b, y_observed)]
    if a.ndim != 1 or a.shape != b.shape or a.shape != y.shape:
        raise ValueError('Inputs must be equally sized one-dimensional arrays')
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all() or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError('Predictions must be binary; labels must be -1, 0 or 1')
    u = y == -1
    caps = np.bincount(2*a[u]+b[u], minlength=4).astype(int)
    return dict(caps=caps.tolist(), pa=int(a.sum()), pb=int(b.sum()),
                ta=int(((a == 1) & (y == 1)).sum()),
                tb=int(((b == 1) & (y == 1)).sum()), s=int((y == 1).sum()))


def from_counts(caps, pa, pb, ta, tb, s, qlo=0, qhi=None):
    """Cell order 00,01,10,11. q bounds concern missing positives only.

    Returns sharp hull endpoints and attainable cell-count witnesses.
    All endpoint comparisons use Python arbitrary-precision integer products.
    """
    caps = np.asarray(caps, dtype=int)
    if caps.shape != (4,) or (caps < 0).any():
        raise ValueError('Four nonnegative capacities are required')
    m = int(caps.sum())
    if qhi is None:
        qhi = m
    if not (0 <= qlo <= qhi <= m):
        raise ValueError('Infeasible missing-positive count interval')
    q = np.arange(qlo, qhi+1, dtype=int)
    da, db = pa+s+q, pb+s+q
    max_order = [2,3,0,1] if pa < pb else [2,0,3,1]
    orders = np.tile(list(reversed(max_order)), (len(q),1))
    vals, witnesses = [], []
    for maximize in [False, True]:
        order = orders[:, ::-1] if maximize else orders
        rem = q.copy()
        x = np.zeros((len(q), 4), dtype=int)
        for j in range(4):
            cell = order[:, j]
            take = np.minimum(rem, caps[cell])
            x[np.arange(len(q)), cell] = take
            rem -= take
        # Exact rational comparisons avoid numerical certificates at ties.
        best_num, best_den, k = None, None, None
        for j in range(len(q)):
            xa,xb = int(x[j,2]+x[j,3]),int(x[j,1]+x[j,3])
            dna,dnb=int(da[j]),int(db[j])
            if not dna or not dnb:
                frac=f1(ta+xa,pa,s+int(q[j]))-f1(tb+xb,pb,s+int(q[j]))
                num,den=frac.numerator,frac.denominator
            else:
                num=2*(int(ta)+xa)*dnb-2*(int(tb)+xb)*dna
                den=dna*dnb
            improve=(best_num is None or
                     (num*best_den > best_num*den if maximize else num*best_den < best_num*den))
            if improve: best_num,best_den,k=num,den,j
        v=Fraction(best_num,best_den)
        vals.append(v)
        witnesses.append(dict(q=int(q[k]), x=x[k].tolist()))
    return dict(lower=vals[0], upper=vals[1], lower_witness=witnesses[0],
                upper_witness=witnesses[1])


def joint_bounds(a, b, y_observed, qlo=0, qhi=None):
    return from_counts(**summarize(a,b,y_observed), qlo=qlo, qhi=qhi)


def marginal_bounds(pred, y_observed, qlo=0, qhi=None):
    pred, y = np.asarray(pred), np.asarray(y_observed)
    u = y == -1
    m = int(u.sum())
    if qhi is None:
        qhi = m
    if not 0 <= qlo <= qhi <= m:
        raise ValueError('Infeasible count interval')
    pa, s = int(pred.sum()), int((y == 1).sum())
    t = int(((pred == 1) & (y == 1)).sum())
    up = int(pred[u].sum())
    lower = [f1(t+max(0,q-(m-up)),pa,s+q) for q in range(qlo,qhi+1)]
    upper = [f1(t+min(up,q),pa,s+q) for q in range(qlo,qhi+1)]
    return min(lower), max(upper)


def independent_bounds(a, b, y_observed, qlo=0, qhi=None):
    la,ua=marginal_bounds(a,y_observed,qlo,qhi)
    lb,ub=marginal_bounds(b,y_observed,qlo,qhi)
    return la-ub,ua-lb


def count_matched_bounds(a,b,y_observed,qlo=0,qhi=None):
    """Stronger marginal comparator: enforce the same q before subtraction."""
    m=int((np.asarray(y_observed)==-1).sum())
    if qhi is None:qhi=m
    lows=[];highs=[]
    for q in range(qlo,qhi+1):
        la,ua=marginal_bounds(a,y_observed,q,q)
        lb,ub=marginal_bounds(b,y_observed,q,q)
        lows.append(la-ub);highs.append(ua-lb)
    return min(lows),max(highs)


def certificate(lo, hi):
    return 1 if lo > 0 else (-1 if hi < 0 else 0)
