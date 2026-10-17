"""Legendre p-transfer diagnostics; ascending, unnormalised Legendre coefficients.

All norms are continuous unweighted L2 norms on [-1,1], not lumped LGL norms.
The sign convention is C = d_x psi - Phi. These are reference-element
operators; physical constraints require the inverse Jacobian.
"""
import numpy as np
from numpy.polynomial import legendre as L


def truncate(a, q):
    a = np.asarray(a, dtype=float)
    if q < 0 or q >= len(a):
        raise ValueError("Require 0 <= q < number of source coefficients")
    return a[:q + 1].copy()


def pad(a, n):
    return np.pad(a, (0, max(0, n - len(a))))[:n]


def norm2(a):
    a = np.asarray(a)
    return float(np.sum(a * a * (2 / (2 * np.arange(len(a)) + 1))))


def derivative(a):
    return L.legder(a)


def tails(a, q):
    """s[0],s[1]: sums of discarded even and odd coefficients."""
    n = np.arange(q + 1, len(a))
    return np.array([np.sum(np.asarray(a)[n[n % 2 == p]]) for p in (0, 1)])


def injection(a, q):
    """D T_q a - T_q D a, including the degree-q component."""
    s = tails(a, q)
    k = np.arange(q + 1)
    return -(2 * k + 1) * s[1 - k % 2]


def injection_norm(a, q):
    return np.sqrt(norm2(injection(a, q)))


def predicted_constraint(a, phi, q, inverse_jacobian=1.):
    """Exact coarse C for constant inverse Jacobian, including inherited C."""
    if len(a) != len(phi):
        raise ValueError("Fields must use the same source degree")
    inherited = inverse_jacobian * pad(derivative(a), len(a)) - phi
    return inherited[:q + 1] + inverse_jacobian * injection(a, q)


def admissible(a, phi, q, field_tolerance, constraint_tolerance, h=2.):
    """One affine 1D element. Both tolerances use physical continuous L2.

    This certifies the instant of transfer only. Future evolution, other
    constraints, boundary characteristic data, and mapping aliasing are outside
    this contract. No resetting or modification of fields is performed.
    """
    tail = np.asarray(a).copy(); tail[:q + 1] = 0
    c = predicted_constraint(a, phi, q, 2/h)
    return (h/2*norm2(tail) <= field_tolerance**2 and
            h/2*norm2(c) <= constraint_tolerance**2)


def amplification(N, q):
    """Sharp tail-L2 to injected-constraint-L2 operator norm."""
    if not 0 <= q < N:
        raise ValueError("Require 0 <= q < N")
    k, n = np.arange(q + 1), np.arange(q + 1, N + 1)
    return max(np.sqrt(np.sum(2 * (2 * k[k % 2 != p] + 1)) *
                       np.sum((2 * n[n % 2 == p] + 1) / 2)) for p in (0, 1))


def extremizer(N, q):
    k, n = np.arange(q + 1), np.arange(q + 1, N + 1)
    scores = [np.sum(2 * (2 * k[k % 2 != p] + 1)) *
              np.sum((2 * n[n % 2 == p] + 1) / 2) for p in (0, 1)]
    p = int(np.argmax(scores))
    a = np.zeros(N + 1)
    sel = n[n % 2 == p]
    a[sel] = (2 * sel + 1) / 2
    return a / np.sqrt(norm2(a))


def compatible(a, q):
    """Classical endpoint-preserving H1 projector, not a new projector.

    q>=1. Two highest parity modes receive the discarded endpoint sums.
    Its derivative is T_{q-1} D a. For q=1 the constant also changes.
    """
    if q < 1:
        raise ValueError("Endpoint-preserving projection needs q >= 1")
    b = truncate(a, q)
    s = tails(a, q)
    for p in (0, 1):
        r = q if q % 2 == p else q - 1
        b[r] += s[p]
    return b


def nodes(N, kind="lobatto"):
    if N < 1:
        raise ValueError("Degree must be >=1")
    if kind == "gauss":
        return L.leggauss(N + 1)[0]
    if kind == "lobatto":
        return np.r_[-1., L.Legendre.basis(N).deriv().roots(), 1.]
    raise ValueError(kind)


def nodal_restriction(N, q, source="lobatto", target="lobatto"):
    """V_target[:, :q+1] T V_source^-1: same full-segment mode transfer
    as the inspected SpECTRE branch, independently implemented in NumPy.
    """
    return L.legvander(nodes(q, target), q) @ np.linalg.inv(
        L.legvander(nodes(N, source), N))[:q + 1, :]


def differentiate_axis(a, axis):
    return L.legder(a, axis=axis)


def restrict_tensor(a, degrees):
    return a[tuple(slice(q + 1) for q in degrees)].copy()


def tensor_injection(a, degrees, axis):
    """Tensor identity, with other axes restricted before parity contraction."""
    slices = [slice(q + 1) for q in degrees]
    slices[axis] = slice(None)
    lines = np.moveaxis(a[tuple(slices)], axis, 0)
    q = degrees[axis]
    n = np.arange(q + 1, lines.shape[0])
    s = [np.sum(lines[n[n % 2 == p]], axis=0) for p in (0, 1)]
    c = np.stack([-(2*k + 1)*s[1-k % 2] for k in range(q + 1)])
    return np.moveaxis(c, 0, axis)
