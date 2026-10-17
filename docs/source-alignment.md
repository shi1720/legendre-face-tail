# Source alignment and implementation boundary

Inspected repository: https://github.com/sxs-collaboration/spectre

Pinned commit: `d57433e93984c238a144f83411eda9f125759812`

Inspection date: 17 October 2026.

## Projection

`src/NumericalAlgorithms/Spectral/Projection.cpp`, full-segment branch in
`projection_matrix_child_to_parent`, lines 168-194 in the pinned source, constructs
the transfer as modal-to-nodal times truncated modal coefficients times
nodal-to-modal. The `projection_matrices` pure-p branch at lines 511-519 delegates
to this route. This is the non-massive tensor-product full-element setting of
the one-dimensional analysis. Massive quantities, h-restriction, Fourier and
spherical-harmonic operators are different cases.

The unit tests construct source and target Vandermonde matrices independently
for Gauss and Gauss-Lobatto nodes. This checks the specified matrix operation;
it does not link or execute the production C++ implementation.

## Adaptation motivation

`src/ParallelAlgorithms/Amr/Criteria/Constraints.hpp`, documentation of the
`Constraints` criterion, discusses its coarsening factor and suggests evaluating
constraints after a lower-degree projection as a possible alternative.
Consequently the general idea of a prospective coarse-state constraint check
is explicitly credited to this existing implementation discussion. The paper
contributes a closed-form Legendre transfer diagnostic and its analysis, not
the invention of that general adaptation idea.

## Integration work still needed

Actual integration would require mapped inertial-frame tensor constraints,
existing observations and normalization conventions, boundary and mortar data,
parallel AMR lifecycle tests, and full-system regression runs. This repository
does not claim any of those steps are complete. The affine formula must not be
used alone on nonaffine maps.
