from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from facetail import *
from numpy.polynomial import legendre as L

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42})
colors=['#215A8C','#C95B32','#27806B']

def finish(fig,name):
 fig.tight_layout();fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)

N,q=24,8; a=extremizer(N,q); b=np.zeros(N+1);b[9]=1;b[11]=-1;b/=np.sqrt(norm2(b))
x=np.linspace(-1,1,600)
fig,axs=plt.subplots(1,2,figsize=(6.8,2.55))
for v,n,c in zip([a,b],['Coherent tail','Cancelling tail'],colors):
 axs[0].plot(x,L.legval(x,v),label=n,color=c)
 axs[1].plot(x,L.legval(x,injection(v,q)),label=n,color=c)
axs[0].set(xlabel=r'Reference coordinate $\xi$',ylabel=r'Discarded field $r(\xi)$',title='(a) Equal unit tail norm')
axs[1].set(xlabel=r'Reference coordinate $\xi$',ylabel=r'Injected constraint $c_q(\xi)$',title='(b) Different constraint injection')
axs[0].legend(fontsize=8);finish(fig,'tail_cancellation')

sv=pd.read_csv(ROOT/'results/singular_values.csv')
fig,axs=plt.subplots(1,2,figsize=(6.8,2.6))
for N in [8,16,24,32]:
 d=sv[sv.N==N];axs[0].plot(d.q,d.predicted,label=f'N={N}')
axs[0].set(xlabel='Retained degree q',ylabel=r'Sharp amplification $G_{Nq}$',title='(a) Tail-to-constraint gain')
axs[0].legend(ncol=2,fontsize=8)
axs[1].loglog(sv.predicted,sv.svd,'.',color=colors[0],ms=2.5)
lo,hi=sv.predicted.min(),sv.predicted.max();axs[1].plot([lo,hi],[lo,hi],color='0.6',lw=.8)
axs[1].set(xlabel='Closed-form amplification',ylabel='Weighted matrix singular value',title='(b) Independent SVD check')
finish(fig,'sharp_gain')

d=pd.read_csv(ROOT/'results/gauge_snapshots.csv')
fig,axs=plt.subplots(1,2,figsize=(6.8,2.7))
for q in (2,4,5,6,8,10,12):
 z=d[d.q==q];axs[0].loglog(z.tail_l2,z.measured_constraint,'.',label=str(q),ms=3)
axs[0].axvline(1e-4,color='0.4',ls='--',lw=.8);axs[0].axhline(1e-3,color='0.4',ls='--',lw=.8)
axs[0].set(xlabel=r'Discarded field $L^2$ norm',ylabel=r'Post-transfer constraint $L^2$ norm',title='(a) Gauge-wave transfer candidates')
axs[0].legend(title='q',fontsize=7,ncol=2,title_fontsize=8)
z=d.groupby('q').max(numeric_only=True)
axs[1].semilogy(z.index,z.tail_l2,'o-',label='Ordinary field error',color=colors[0])
axs[1].semilogy(z.index,z.compatible_field_error,'s--',label='Compatible field error',color=colors[1])
axs[1].set(xlabel='Retained degree q',ylabel=r'Maximum field $L^2$ error',title='(b) Cost of endpoint preservation')
axs[1].legend(fontsize=8)
finish(fig,'gauge_transfer')

d=pd.read_csv(ROOT/'results/evolution.csv');d=d[(d.q==6)&(d.dt_factor==.5)]
cv=pd.read_csv(ROOT/'results/convergence.csv')
fig,axs=plt.subplots(1,2,figsize=(6.8,2.7))
for m,c in zip(['truncate','reset','compatible'],colors):
 z=d[d.method==m];axs[0].semilogy(z.time,np.maximum(z.constraint,1e-16),label=m.capitalize(),color=c,lw=1.1)
axs[0].axvline(.25,color='0.5',ls=':',lw=.8)
axs[0].set(xlabel='Time t',ylabel=r'Constraint $L^2$ norm',title='(a) Coarsening 14 to 6 at t=0.25',ylim=(1e-14,2e-3))
axs[0].legend(fontsize=8)
axs[1].semilogy(cv.p,cv.field_error,'o-',label='Field error',color=colors[0])
axs[1].semilogy(cv.p,cv.constraint,'s--',label='Constraint norm',color=colors[1])
axs[1].set(xlabel='Fixed polynomial degree',ylabel=r'$L^2$ error at t=1',title='(b) Independent DG convergence')
axs[1].legend(fontsize=8)
finish(fig,'wave_evolution')

d=pd.read_csv(ROOT/'results/mapping.csv')
fig,ax=plt.subplots(figsize=(5.4,2.2))
ax.plot(d.alpha,d.actual_l2,'o-',color=colors[0],label='Physical derivative transfer')
ax.plot(d.alpha,d.affine_prediction_l2,'s--',color=colors[1],label='Affine face-tail prediction')
ax.set(xlabel=r'Map parameter $\alpha$',ylabel=r'Constraint $L^2(d\xi)$ norm',title=r'Cancelling tail under $x=\xi+\alpha\xi^2/2$')
ax.legend(fontsize=8);finish(fig,'mapped_failure')

# Report all selected numerical statements from raw data.
a=pd.read_csv(ROOT/'results/algebra.csv'); s=pd.read_csv(ROOT/'results/singular_values.csv')
g=pd.read_csv(ROOT/'results/gauge_snapshots.csv')
e=pd.read_csv(ROOT/'results/evolution_summary.csv')
accepted=g.tail_l2<=1e-4; unsafe=g.measured_constraint>1e-3
summary={'random_cases':int(a['count'].sum()),'max_algebra_error':float(a.max_abs_error.max()),
 'svd_cases':len(s),'max_svd_relative_error':float(np.max(abs(s.predicted-s.svd)/s.predicted)),
 'gauge_cases':len(g),'gauge_identity_max_abs_error':float(abs(g.measured_constraint-g.predicted_constraint).max()),
 'gauge_compatible_constraint_max':float(g.compatible_constraint.max()),
 'gauge_compatible_trace_max':float(g.compatible_trace_error.max()),
 'field_gate_accepts':int(accepted.sum()),'field_gate_unsafe':int((accepted&unsafe).sum()),
 'guard_accepts':int((accepted&~unsafe).sum())}
import json
(ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
