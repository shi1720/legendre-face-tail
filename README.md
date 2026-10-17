# Legendre face-tail diagnostics

Reference implementation for **Face-Tail Diagnostics for Constraint Injection
under Legendre p-Coarsening**, by Shivam Gupta (2026).

This is a research working paper, not a peer-reviewed publication. It studies
the instantaneous first-order reduction constraint introduced by a particular
spectral transfer. It does not implement an Einstein evolution or claim a new
constraint-projection method.

## Reproduce

Python 3.11 or later is recommended. Use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
python experiments/run.py
python experiments/figures.py
cd paper
tectonic main.tex
```

The recorded complete experiment took about 20 seconds on the workstation; runtime
depends on hardware and numerical libraries. No network calls or external data
are made by the tests or experiments. Tectonic may download standard TeX packages.

## Main result

For `u = sum(a[n] P_n)` and ordinary degree restriction `T_q`, the new logical
constraint is `(D T_q - T_q D) u`. Its retained coefficient `k` is

```
-(2*k + 1) * sum(a[n] for n > q if n-k is odd)
```

Only two discarded parity sums are needed. These are equivalent to the two
endpoint traces of the discarded potential. See the proofs for the exact sharp
gain and the field penalty of the classical compatible H1 projector.

## Example: an affine physical element

```python
import sys
sys.path.insert(0, "src")
import numpy as np
from facetail import admissible, derivative, pad

a = np.zeros(17)
a[0], a[1], a[7] = 1.0, 0.1, 0.001
h = 0.25
phi = pad(derivative(a), len(a)) * 2 / h
ok = admissible(a, phi, q=6, field_tolerance=1e-3,
                constraint_tolerance=1e-3, h=h)
print(ok)
```

This checks **both** the field error and the post-transfer reduction constraint,
including inherited source constraint error. It does not change the state.

## Contents

* `src/facetail.py`: diagnostics, sharp gain, classical compatible projection,
  nodal transfer and tensor-product identities.
* `tests/`: rational integration, inherited-error and decision checks, all
  Gauss/Lobatto nodal combinations, sharpness and tensor checks.
* `experiments/run.py`: random algebra, SVD checks, gauge-wave component
  transfers, nonaffine mapping counterexample and a periodic DG wave solver.
* `experiments/figures.py`: figures and numerical claim summary.
* `results/`: raw CSVs and environment/summary records.
* `figures/`: vector PDF figures and PNG previews.
* `paper/`: manuscript PDF, LaTeX and verified bibliography.
* `docs/`: literature positioning, source alignment and scope audit.

## Conditions and limitations

* Standard, unnormalised Legendre coefficients are in ascending degree order.
* Norms use continuous unweighted L2 integration, not lumped LGL quadrature.
* The physical guard is for an affine 1D element; a nonaffine map introduces
  another commutator. A supplied counterexample demonstrates its importance.
* Two scalars in 1D become two face **functions** per direction in tensor grids.
* The compatible projector is classical and changes the last two potential
  coefficients. It is not a general relativity constraint solver.
* Eliminating the immediate constraint jump does not guarantee lower later
  field error. Negative comparisons are included in the manuscript.
* SpECTRE source was inspected at the pinned revision documented in
  `docs/source-alignment.md`. Our NumPy implementation is independent; no
  linked SpECTRE execution or upstream patch is claimed.

## Availability and citation

The manuscript is in `paper/main.pdf`. The repository is released under the MIT
license. See `CITATION.cff`. Raw numerical results are reproducible outputs, not
observational data. The research does not establish patentability or commercial
performance.

## Journal manuscript preparation

On 17 October 2026, the author reviewed and approved a journal preparation
copy and the required disclosure of substantive Codex assistance. The manuscript
now includes computational provenance and that disclosure. Journal registration
and submission are in progress; this is not a claim of submission or acceptance.
