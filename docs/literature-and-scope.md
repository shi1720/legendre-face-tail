# Literature and scope audit

Search and primary-source inspection date: 17 October 2026.

Search families included Legendre projection/differentiation commutators;
endpoint-preserving H1 projectors; commuting projection-based interpolation;
optimal scalar-wave/generalized-harmonic constraint projection; spectral
coarsening and SpEC/SpECTRE constraint-based adaptivity. Search results were used
for discovery and checked against the papers, publisher metadata or project
source. This is a targeted review, not an exhaustive patent or priority search.

| Prior work | Established result and implication |
| --- | --- |
| Holst et al., PRD 70, 084017 (2004), DOI 10.1103/PhysRevD.70.084017 | Optimal constraint projection for symmetric hyperbolic evolution, including scalar fields on black holes. Constraint projection is not new. |
| Lindblom et al., CQG 23, S447 (2006), DOI 10.1088/0264-9381/23/16/S09 | First-order generalized harmonic formulation and damping. Reduction constraints are established. |
| Rinne, CQG 23, 6275 (2006), DOI 10.1088/0264-9381/23/22/013 | Constraint-preserving boundary analysis. Local transfer does not replace it. |
| Demkowicz and Buffa, CMAME 194, 267 (2005), DOI 10.1016/j.cma.2004.07.007 | Commuting projection-based interpolation. The H1 projector is classical. |
| Arnold, Falk and Winther, Acta Numerica 15, 1 (2006), DOI 10.1017/S0962492906210018 | Compatible spaces and commuting differential projections. No new framework is claimed. |
| Rapisarda, van Waarde and Camlibel, IEEE TAC 69, 4307 (2024), DOI 10.1109/TAC.2023.3321214 | Explicit Legendre derivative parity structure and projection/differentiation commutator. Noncommutation is not new. |
| Szilagyi, IJMPD 23, 1430014 (2014), DOI 10.1142/S0218271814300146 | Spectral AMR, topology-direction constraint projections, and removal of constraint-violating pile-up modes. Coarsening can reduce pre-existing constraints. |
| SpECTRE pinned source, cited in source-alignment.md | Pure-p modal transfer and explicit suggestion to evaluate a prospective lower-order constraint. The general guard idea is already present. |
| Alcubierre et al., CQG 21, 589 (2004), DOI 10.1088/0264-9381/21/2/019 | Gauge-wave testbed. Our experiment only transfers one metric component. |
| Kidder et al., JCP 335, 84 (2017), DOI 10.1016/j.jcp.2016.12.059; Lovelace et al., arXiv:2410.00265 | Production context and strong-field capabilities; our independent reference solver does not reproduce those simulations. |

## Defensible contribution

An explicit endpoint-trace expression for the Legendre reduction-constraint
transfer defect, its sharp finite-tail L2 amplification, an exact inherited-error
decomposition, the compatible projection's field penalty, a potential-only
repair obstruction, and a reproducible study of candidate screening and its
limits. The underlying results are elementary specializations of classical
polynomial identities and compatible approximation theory. The targeted review
did not identify this exact operational package, but cannot establish global
priority or guarantee acceptance. The manuscript deliberately makes a narrow
contribution claim.

## Negative evidence retained

* Equal-energy tails can inject either zero or the maximal constraint.
* Compatible transfer has a larger immediate field error than L2 truncation.
* Compatible transfer does not consistently improve later evolution error.
* Derivative resetting can add wave energy in the measured example.
* A varying inverse Jacobian invalidates a face-tail-only physical prediction.

## Publication readiness

This is suitable as a transparent working paper and a technical project for
discussion at a numerical-relativity school/workshop. It is not evidence for
an industry-defining advance, a major-conference acceptance probability, a
patent claim, or strong-field waveform improvement. Those claims would require
substantially more work and external specialist assessment.
