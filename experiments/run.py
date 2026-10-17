"""Reproduce algebra, transfer, mapping, and time-dependent validation."""
import sys, json, csv, platform, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from numpy.polynomial import legendre as L
from scipy.special import roots_legendre
from facetail import *

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)

def save(name,rows):
    with (OUT/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def coefficients(f,N):
    x,w=roots_legendre(max(2*N+8,96))
    return (L.legvander(x,N).T@(w*f(x)))*(2*np.arange(N+1)+1)/2

def algebra():
    rng=np.random.default_rng(6602027); rows=[]
    for N in (8,16,24,32,48):
        for q in (2,4,6):
            errors=[]; bounds=[]
            for _ in range(200):
                a=rng.normal(size=N+1)/(1+np.arange(N+1))**2
                d=pad(L.legder(a[:q+1]),q+1)-pad(L.legder(a),N+1)[:q+1]
                errors.append(np.max(np.abs(d-injection(a,q))))
                tail=a.copy(); tail[:q+1]=0
                bounds.append(injection_norm(a,q)/(amplification(N,q)*np.sqrt(norm2(tail))))
            rows.append(dict(N=N,q=q,count=200,max_abs_error=max(errors),max_bound_fraction=max(bounds),
                             sharp_gain=amplification(N,q),attained_gain=injection_norm(extremizer(N,q),q)))
    save('algebra',rows)
    # Equal discarded energy, either coherent or cancelling parity sums.
    N,q=24,8
    coherent=extremizer(N,q)
    cancel=np.zeros(N+1); cancel[q+1]=1; cancel[q+3]=-1
    cancel/=np.sqrt(norm2(cancel))
    save('cancellation',[dict(case=n,tail_norm=np.sqrt(norm2(a)),constraint_norm=injection_norm(a,q))
                         for n,a in [('coherent',coherent),('cancelling',cancel)]])
    # SVD of the weighted commutator gives an independent operator-norm check.
    rows=[]
    for N in range(3,33):
        for q in range(1,N):
            eye=np.eye(N+1); D=np.column_stack([pad(L.legder(a),N+1) for a in eye])
            T=np.eye(N+1)[:q+1]
            Dq=np.column_stack([pad(L.legder(a),q+1) for a in np.eye(q+1)])
            B=(Dq@T-T@D)[:,q+1:]
            wm=np.sqrt(2/(2*np.arange(q+1)+1))
            wn=np.sqrt(2/(2*np.arange(q+1,N+1)+1))
            s=np.linalg.svd(wm[:,None]*B/wn[None,:],compute_uv=False)
            rows.append(dict(N=N,q=q,predicted=amplification(N,q),svd=s[0],rank=int(np.sum(s>1e-10))))
    save('singular_values',rows)

def snapshots():
    # Gauge-wave H=1-A sin(2pi(x-t)); g_xx=H is one exact metric component.
    # This is a transfer test of its reduction constraint, not an Einstein evolution.
    rows=[]; N=24; h=.5; A=.1
    for t in np.linspace(0,1,33)[:-1]:
        for center in np.linspace(-.875,.875,8):
            a=coefficients(lambda xi:1-A*np.sin(2*np.pi*(center+h*xi/2-t)),N)
            for q in (2,3,4,5,6,8,10,12):
                uc=a[:q+1]; phi=pad(L.legder(a),N+1)[:q+1]*2/h
                cc=pad(L.legder(uc),q+1)*2/h-phi
                b=compatible(a,q)
                # Compatible derivative uses degree q-1, not q.
                phi_b=pad(L.legder(a)[:q],q+1)*2/h
                cb=pad(L.legder(b),q+1)*2/h-phi_b
                tail=a.copy(); tail[:q+1]=0
                rows.append(dict(time=t,center=center,q=q,tail_l2=np.sqrt(h/2*norm2(tail)),
                    predicted_constraint=np.sqrt(2/h)*injection_norm(a,q),
                    measured_constraint=np.sqrt(h/2*norm2(cc)),
                    compatible_constraint=np.sqrt(h/2*norm2(cb)),
                    compatible_field_error=np.sqrt(h/2*norm2(pad(b,N+1)-a)),
                    trace_error=max(abs(L.legval(np.array([-1,1]),uc)-L.legval(np.array([-1,1]),a))),
                    compatible_trace_error=max(abs(L.legval(np.array([-1,1]),b)-L.legval(np.array([-1,1]),a)))))
    save('gauge_snapshots',rows)

def mapping():
    # x=xi+alpha xi^2/2; J=1+alpha xi. Rational physical derivative.
    # Project that derivative by high-order quadrature to separate mapping error
    # from interpolation aliasing.
    rows=[]; N=24; q=8
    a=np.zeros(N+1); a[9]=1; a[11]=-1
    x,w=roots_legendre(256); V=L.legvander(x,q)
    for alpha in (0,.1,.3,.6):
        v=1/(1+alpha*x)
        fine_phi=v*L.legval(x,L.legder(a))
        coarse_phi=(V.T@(w*fine_phi))*(2*np.arange(q+1)+1)/2
        actual=v*L.legval(x,L.legder(a[:q+1]))-L.legval(x,coarse_phi)
        affine=v*L.legval(x,injection(a,q))
        rows.append(dict(alpha=alpha,actual_l2=np.sqrt(np.sum(w*actual**2)),
            affine_prediction_l2=np.sqrt(np.sum(w*affine**2)),
            mapping_remainder_l2=np.sqrt(np.sum(w*(actual-affine)**2))))
    save('mapping',rows)


class DG:
    """Strong-form upwind LGL DG for a periodic flat-space scalar wave.

    psi_t=-Pi; Pi_t=-Phi_x; Phi_t=-Pi_x+gamma*(psi_x-Phi).
    psi has zero propagation speed and receives no numerical-flux lift.
    Continuous norms in reports use modal integration, not LGL quadrature.
    """
    def __init__(self,p,K=8,gamma=1.):
        self.p=p; self.K=K; self.gamma=gamma; self.h=2/K
        self.xi=nodes(p); self.V=L.legvander(self.xi,p); self.iV=np.linalg.inv(self.V)
        self.D=np.column_stack([L.legval(self.xi,L.legder(self.iV[:,i])) for i in range(p+1)])
        self.w=2/(p*(p+1)*L.legval(self.xi,L.Legendre.basis(p).coef)**2)
        self.x=(-1+(np.arange(K)+.5)*self.h)[:,None]+self.h*self.xi/2
        self.xq,self.wq=roots_legendre(max(48,2*p+8))
        self.Vq=L.legvander(self.xq,p)
        self.physq=(-1+(np.arange(K)+.5)*self.h)[:,None]+self.h*self.xq/2
    def initial(self):
        psi=np.sin(np.pi*self.x)+.2*np.cos(3*np.pi*self.x)
        phi=np.pi*np.cos(np.pi*self.x)-.6*np.pi*np.sin(3*np.pi*self.x)
        return np.array([psi,phi,phi]) # right-moving: Pi=Phi
    def rhs(self,U):
        psi,pi,phi=U; scale=2/self.h
        dx=lambda z: scale*(z@self.D.T)
        rpi=-dx(phi); rphi=-dx(pi)+self.gamma*(dx(psi)-phi)
        wp=pi+phi; wm=pi-phi
        # Upwind fluxes for the acoustic (pi,phi) subsystem. The gamma*Dpsi
        # term is evaluated in the volume; these are not the characteristic
        # combinations of the full principal matrix at nonzero gamma.
        flux_pi_L=(np.roll(wp[:,-1],1)-wm[:,0])/2
        flux_phi_L=(np.roll(wp[:,-1],1)+wm[:,0])/2
        flux_pi_R=(wp[:,-1]-np.roll(wm[:,0],-1))/2
        flux_phi_R=(wp[:,-1]+np.roll(wm[:,0],-1))/2
        rpi[:,0]+=scale*(flux_pi_L-phi[:,0])/self.w[0]
        rpi[:,-1]-=scale*(flux_pi_R-phi[:,-1])/self.w[-1]
        rphi[:,0]+=scale*(flux_phi_L-pi[:,0])/self.w[0]
        rphi[:,-1]-=scale*(flux_phi_R-pi[:,-1])/self.w[-1]
        return np.array([-pi,rpi,rphi])
    def step(self,U,dt):
        a=self.rhs(U); b=self.rhs(U+dt*a/2); c=self.rhs(U+dt*b/2); d=self.rhs(U+dt*c)
        return U+dt*(a+2*b+2*c+d)/6
    def metrics(self,U,t):
        psi,pi,phi=U; c=2/self.h*(psi@self.D.T)-phi
        def norm(z):
            modal=z@self.iV.T
            return np.sqrt(self.h/2*np.sum(modal**2*(2/(2*np.arange(self.p+1)+1))))
        exact=np.sin(np.pi*(self.physq-t))+.2*np.cos(3*np.pi*(self.physq-t))
        error=(psi@self.iV.T)@self.Vq.T-exact
        error_norm=np.sqrt(self.h/2*np.sum(error**2*self.wq))
        return dict(constraint=norm(c),field_error=error_norm,
                    energy=(norm(pi)**2+norm(phi)**2)/2)

def transfer(old,U,q,method):
    new=DG(q,K=old.K,gamma=old.gamma)
    modes=U@old.iV.T
    b=modes[:,:,:q+1].copy()
    if method=='reset':
        for e in range(old.K): b[2,e]=pad(L.legder(b[0,e]),q+1)*2/old.h
    elif method=='compatible':
        for e in range(old.K): b[0,e]=compatible(modes[0,e],q)
        b[2,:,-1]=0 # T_{q-1} Phi; retains inherited constraint in lower modes
    elif method!='truncate': raise ValueError(method)
    return new,b@new.V.T

def evolution():
    rows=[]; summary=[]
    for q in (4,6,8):
        for method in ('truncate','reset','compatible'):
            for factor in (1.,.5):
                dg=DG(14); U=dg.initial(); t=0.
                for end in (.25,1.):
                    dt=factor*.12*dg.h/(dg.p+1)**2
                    n=int(np.ceil((end-t)/dt)); dt=(end-t)/n
                    for i in range(n):
                        U=dg.step(U,dt); t+=dt
                        if i%max(1,n//80)==0:
                            rows.append(dict(q=q,method=method,dt_factor=factor,time=t,**dg.metrics(U,t)))
                    if end==.25:
                        before=dg.metrics(U,t)
                        modes=U@dg.iV.T
                        predicted=np.sqrt(sum(2/dg.h*injection_norm(a,q)**2 for a in modes[0]))
                        dg,U=transfer(dg,U,q,method)
                        after=dg.metrics(U,t)
                        summary.append(dict(q=q,method=method,dt_factor=factor,
                            before_constraint=before['constraint'],after_constraint=after['constraint'],
                            predicted_injection=predicted,energy_jump=after['energy']-before['energy'],
                            after_field_error=after['field_error']))
                        rows.append(dict(q=q,method=method,dt_factor=factor,time=t,**after))
                    else:
                        summary[-1].update({'final_'+k:v for k,v in dg.metrics(U,t).items()})
    # Uncoarsened spatial reference, plus coarse-grid initial convergence.
    conv=[]
    for p in (4,6,8,10,14):
        dg=DG(p); U=dg.initial(); dt=.06*dg.h/(p+1)**2
        n=int(np.ceil(1/dt));dt=1/n
        for _ in range(n): U=dg.step(U,dt)
        conv.append(dict(p=p,**dg.metrics(U,1.)))
    save('evolution',rows); save('evolution_summary',summary); save('convergence',conv)

if __name__=='__main__':
    start=time.perf_counter()
    algebra(); snapshots(); mapping(); evolution()
    (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
        numpy=np.__version__,seed=6602027,elapsed_seconds=time.perf_counter()-start),indent=2))
    print('Experiments complete:',OUT)
