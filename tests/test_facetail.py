"""Independent algebraic and nodal checks, including exact rational arithmetic."""
import unittest
from fractions import Fraction as F
import numpy as np
from numpy.polynomial import legendre as L
from facetail import *


class FaceTailTests(unittest.TestCase):
    def test_rational_commutator(self):
        # Derive derivatives in the monomial basis using Rodrigues' recurrence;
        # compare polynomial values, not the implementation's derivative matrix.
        from math import factorial
        def P(n):
            out = [F(0)]*(n+1)
            for k in range(n//2+1):
                out[n-2*k] = F((-1)**k*factorial(2*n-2*k),
                    2**n*factorial(k)*factorial(n-k)*factorial(n-2*k))
            return out
        for N in range(2, 16):
            for q in range(N):
                a = [F((n*7)%11-5, n+1) for n in range(N+1)]
                # Exact projected derivative coefficients via independent
                # integration by parts: <P_n',P_k> = 2 for n>k of odd gap.
                coeff = [-F(2*k+1)*sum(a[n] for n in range(q+1,N+1)
                    if (n-k)%2) for k in range(q+1)]
                poly = [F(0)]*(q+1)
                for k,c in enumerate(coeff):
                    for j,v in enumerate(P(k)): poly[j] += c*v
                # Monomial quadrature: integrate derivative of full tail times P_k.
                for k in range(q+1):
                    integral = F(0)
                    for n in range(q+1,N+1):
                        for i,v in enumerate(P(n)[1:],1):
                            for j,w in enumerate(P(k)):
                                power=i-1+j
                                if power%2==0: integral += a[n]*i*v*w*F(2,power+1)
                    self.assertEqual(coeff[k], -F(2*k+1,2)*integral)

    def test_nodal_all_quadratures(self):
        rng=np.random.default_rng(660)
        for N in (4,8,16,24):
            for q in (2,3):
                a=rng.normal(size=N+1)
                da=L.legder(a)
                for source in ('gauss','lobatto'):
                    for target in ('gauss','lobatto'):
                        R=nodal_restriction(N,q,source,target)
                        x=nodes(N,source); y=nodes(q,target)
                        ua=R@L.legval(x,a)
                        pa=R@L.legval(x,da)
                        bc=np.linalg.solve(L.legvander(y,q),ua)
                        residual=L.legval(y,L.legder(bc))-pa
                        np.testing.assert_allclose(residual,L.legval(y,injection(a,q)),atol=2e-10)

    def test_projection_properties(self):
        rng=np.random.default_rng(2027)
        for N in (4,8,16,32):
            for q in range(1,N):
                a=rng.normal(size=N+1)
                b=compatible(a,q)
                np.testing.assert_allclose(L.legval([-1,1],b),L.legval([-1,1],a),atol=2e-12)
                np.testing.assert_allclose(L.legder(b),pad(L.legder(a),q)[:q],atol=2e-12)
                self.assertLessEqual(norm2(L.legder(b)),norm2(L.legder(a))*(1+1e-12))
                np.testing.assert_allclose(compatible(pad(b,N+1),q),b)
                if q>=2:
                    np.testing.assert_allclose(compatible(pad(b,N+1),q-1),compatible(a,q-1),atol=2e-12)

    def test_sharpness(self):
        for N in range(2,33):
            for q in range(N):
                a=extremizer(N,q)
                self.assertAlmostEqual(injection_norm(a,q),amplification(N,q),places=10)

    def test_inherited_constraint(self):
        rng=np.random.default_rng(1720)
        for N,q in [(8,3),(16,6),(24,10)]:
            a=rng.normal(size=N+1); phi=rng.normal(size=N+1)
            fine=pad(L.legder(a),N+1)-phi
            coarse=pad(L.legder(a[:q+1]),q+1)-phi[:q+1]
            np.testing.assert_allclose(coarse,fine[:q+1]+injection(a,q),atol=1e-12)

    def test_guard_with_inherited_error(self):
        rng=np.random.default_rng(1721)
        for _ in range(100):
            N,q,h=16,6,.25
            a=rng.normal(size=N+1)/(1+np.arange(N+1))**3
            phi=pad(L.legder(a),N+1)*2/h+rng.normal(size=N+1)*1e-4
            direct=pad(L.legder(a[:q+1]),q+1)*2/h-phi[:q+1]
            np.testing.assert_allclose(predicted_constraint(a,phi,q,2/h),direct,atol=1e-12)
            tail=a.copy(); tail[:q+1]=0
            for eta,eps in [(1e-3,1e-2),(1e-4,1e-3),(1.,1.)]:
                expected=(h/2*norm2(tail)<=eta**2 and h/2*norm2(direct)<=eps**2)
                self.assertEqual(admissible(a,phi,q,eta,eps,h),expected)

    def test_tensor(self):
        rng=np.random.default_rng(42)
        for shape,qs in [((9,8),(3,4)),((8,7,6),(3,2,3))]:
            a=rng.normal(size=shape)
            for axis in range(a.ndim):
                small=restrict_tensor(a,qs)
                ds=differentiate_axis(small,axis)
                expanded=np.zeros_like(small)
                sl=[slice(None)]*a.ndim; sl[axis]=slice(0,qs[axis])
                expanded[tuple(sl)]=ds
                direct=expanded-restrict_tensor(differentiate_axis(a,axis),qs)
                np.testing.assert_allclose(direct,tensor_injection(a,qs,axis),atol=1e-12)

if __name__=='__main__': unittest.main()
