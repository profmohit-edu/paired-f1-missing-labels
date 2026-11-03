from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];d=pd.read_csv(R/'results/comparisons.csv')
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
names={'breast_cancer':'Breast cancer','wine':'Wine','digits':'Digits'}
colors=['#174a7e','#b55723','#666666']
u=d[d.regime=='unrestricted']
fig,axes=plt.subplots(1,3,figsize=(10,3.1),sharey=True)
for ax,(name,g) in zip(axes,u.groupby('dataset',sort=True)):
    s=g.groupby('missing_fraction')[['width','count_matched_width','independent_width']].mean()
    for (col,label),c,ls in zip([('width','Joint'),('count_matched_width','Count matched'),('independent_width','Separate')],colors,['-','--',':']):
        ax.plot(s.index,s[col],marker='o',label=label,color=c,ls=ls)
    ax.set(title=names[name],xlabel='Fraction of test labels hidden',xticks=[.2,.5,.8],ylim=(0,2.05))
axes[0].set_ylabel('Mean identification interval width');axes[2].legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(R/'figures/Fig1.pdf');fig.savefig(R/'figures/Fig1.png',dpi=300);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,3.1),sharey=True)
for ax,(name,g) in zip(axes,u.groupby('dataset',sort=True)):
    for col,label,c,ls in [('joint_certificate','Joint',colors[0],'-'),('count_matched_certificate','Count matched',colors[1],'--')]:
        s=g.assign(cert=g[col].ne(0)).groupby('missing_fraction').cert.mean()
        ax.plot(s.index,s.values,marker='o',label=label,color=c,ls=ls)
    ax.set(title=names[name],xlabel='Fraction of test labels hidden',xticks=[.2,.5,.8],ylim=(-.02,1.04))
axes[0].set_ylabel('Fraction with a strict ranking certificate');axes[2].legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(R/'figures/Fig2.pdf');fig.savefig(R/'figures/Fig2.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(5.8,3.4))
for (col,label,c),ls,mk in zip([('unrestricted','No count information',colors[0]),('oracle_band','Oracle count band',colors[1]),('oracle_exact','Oracle exact count',colors[2])],['-','--',':'],['o','s','^']):
    g=d[d.regime==col].groupby('missing_fraction').width.mean()
    ax.plot(g.index,g.values,marker=mk,label=label,color=c,ls=ls)
ax.set(xlabel='Fraction of test labels hidden',ylabel='Mean joint interval width',xticks=[.2,.5,.8],ylim=(0,.34))
ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(R/'figures/Fig3.pdf');fig.savefig(R/'figures/Fig3.png',dpi=300);plt.close(fig)
