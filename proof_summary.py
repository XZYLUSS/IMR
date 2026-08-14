"""
imr/proof_summary.py -- the proof-summary transport layer of Section 6.3.

This implements exactly what the manuscript defines, and nothing beyond it:

  * the nine-coordinate obligation vector with four-valued entries
    {0, 1, NA, ?} and an applicability mask;
  * the partition of a mask into EQUIVALENCE-INVARIANT and PRESENTATION-SPECIFIC
    coordinates;
  * the obligation transport map lambda_T as a BIJECTION on the invariant part
    only -- retirement and introduction are impossible under strictly
    invertible groupoid arrows, so they are handled by the partition and not by
    the map;
  * the three functor laws for lambda, checked at admission;
  * the proof-summary discrepancy delta^P_T, in which a transported-claim
    mismatch is DECISIVE on its own and only the OBLIGATION component can be
    UNAVAILABLE (when no coordinate is comparable), with '?' routed to
    adjudication rather than compared literally.

The released calibration bank contains final-answer families only, so this
layer is exercised by the test suite rather than by the bank. That is stated
in the paper and in README.md; nothing here is claimed to have been run on
model outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# The nine obligation coordinates, in the order used by the manuscript.
COORDINATES: Tuple[str, ...] = (
    "logic", "domain", "local", "global", "quotient",
    "analytic", "boundary", "transport", "explanation",
)

NA = "NA"          # not applicable under the item's declared mask
UNKNOWN = "?"      # uncertain; routed to adjudication, never compared literally
UNAVAILABLE = "unavailable"   # declared third outcome of delta^P, not zero


VALID_ENTRIES = (0, 1, 0.0, 1.0, NA, UNKNOWN)


@dataclass(frozen=True)
class ObligationVector:
    """One judged artifact's obligation record.

    values : coordinate -> 0 | 1 | NA | '?'
    invariant : the subset of applicable coordinates the certificate proves are
                equivalence-invariant. The rest are presentation-specific and
                are reported per arm, never transported.

    Round-6 M08: the value domain, coordinate uniqueness and mask consistency
    are validated at construction; malformed records are refused, not carried."""
    values: Dict[str, object]
    invariant: Tuple[str, ...]

    def __post_init__(self):
        bad = set(self.values) - set(COORDINATES)
        if bad:
            raise ValueError(f"unknown coordinates: {sorted(bad)}")
        if len(self.values) != len(COORDINATES):
            raise ValueError(
                f"the obligation vector has exactly {len(COORDINATES)} "
                f"coordinates; got {len(self.values)}. Coordinates that do not "
                f"apply are recorded as NA, not omitted.")
        for c, v in self.values.items():
            if not any(v is e or v == e for e in VALID_ENTRIES):
                raise ValueError(f"coordinate {c}: value {v!r} outside "
                                 f"{{0, 1, NA, ?}}")
        if len(set(self.invariant)) != len(self.invariant):
            raise ValueError("duplicate coordinates in the invariant declaration")
        for c in self.invariant:
            if c not in self.values:
                raise ValueError(f"invariant coordinate {c} not in the record")
            if self.values[c] == NA:
                raise ValueError(f"invariant coordinate {c} cannot be NA")

    @property
    def mask(self) -> Tuple[str, ...]:
        """Applicable coordinates: everything not marked NA."""
        return tuple(c for c in COORDINATES
                     if c in self.values and self.values[c] != NA)

    @property
    def presentation_specific(self) -> Tuple[str, ...]:
        return tuple(c for c in self.mask if c not in self.invariant)


class LambdaT:
    """The obligation transport map: a bijection on invariant coordinates.

    A partial map with retirement (i -> dagger) and introduction (a target
    coordinate with no source) cannot satisfy lambda_{T^-1} = lambda_T^{-1}
    under strictly invertible arrows, and several retired coordinates cannot
    map injectively to one symbol. The constructor therefore REFUSES anything
    that is not a bijection between the two invariant sets."""

    def __init__(self, mapping: Dict[str, str], source: Sequence[str],
                 target: Sequence[str]):
        src, tgt = tuple(source), tuple(target)
        if len(set(src)) != len(src):
            raise ValueError(f"duplicate source coordinates: {list(src)}")
        if len(set(tgt)) != len(tgt):
            raise ValueError(f"duplicate target coordinates: {list(tgt)}")
        if len(src) != len(tgt):
            raise ValueError("lambda must be a bijection: |source| != |target|")
        if set(mapping) != set(src):
            raise ValueError("lambda must be total on the source invariant set")
        img = list(mapping.values())
        if len(set(img)) != len(img):
            raise ValueError("lambda must be injective")
        if set(img) != set(tgt):
            raise ValueError("lambda must be surjective onto the target invariant set")
        self.mapping = dict(mapping)
        self.source = src
        self.target = tgt

    def __call__(self, coord: str) -> str:
        return self.mapping[coord]

    def inverse(self) -> "LambdaT":
        return LambdaT({v: k for k, v in self.mapping.items()},
                       self.target, self.source)

    def compose(self, other: "LambdaT") -> "LambdaT":
        """self o other, i.e. lambda_U o lambda_T with other = lambda_T."""
        if other.target != self.source:
            raise ValueError("composition is not defined: target/source mismatch")
        return LambdaT({k: self.mapping[other.mapping[k]] for k in other.source},
                       other.source, self.target)

    @staticmethod
    def identity(coords: Sequence[str]) -> "LambdaT":
        return LambdaT({c: c for c in coords}, coords, coords)

    def __eq__(self, other) -> bool:
        return (isinstance(other, LambdaT) and self.mapping == other.mapping
                and self.source == other.source and self.target == other.target)

    def __repr__(self) -> str:
        return f"LambdaT({self.mapping})"


def check_lambda_laws(lam_T: LambdaT, *, lam_identity: LambdaT = None,
                      lam_inverse: LambdaT = None, lam_U: LambdaT = None,
                      lam_composite: LambdaT = None) -> Dict[str, Optional[bool]]:
    """Verify the three functor laws against INDEPENDENTLY SUPPLIED maps.

    Round-6 M07: checking a map against constructions derived from itself is
    algebraic self-consistency, not verification of certificate data. The
    certificate of each arrow declares its own lambda; the identity, inverse
    and composite arrows declare theirs; and this routine compares them
    EXTENSIONALLY. Maps that are not supplied are reported as unchecked rather
    than silently passed."""
    out: Dict[str, bool] = {}
    idT = LambdaT.identity(lam_T.source)
    out["identity_declared"] = (lam_identity == idT) if lam_identity is not None \
        else None
    out["inverse_declared"] = (lam_inverse == lam_T.inverse()) \
        if lam_inverse is not None else None
    if lam_U is not None and lam_composite is not None:
        out["composition_declared"] = (lam_composite == lam_U.compose(lam_T))
    else:
        out["composition_declared"] = None
    # algebraic self-consistency, reported separately and never as verification
    out["self_inverse_cancels"] = (lam_T.inverse().compose(lam_T) == idT)
    out["self_identity_neutral"] = (lam_T.compose(idT) == lam_T)
    return out


def comparable_coordinates(src: ObligationVector, tgt: ObligationVector,
                           lam: LambdaT) -> List[str]:
    """K_T: invariant coordinates resolved to {0,1} on BOTH sides. '?' on
    either side is routed to adjudication and excluded."""
    out = []
    for c in lam.source:
        a, b = src.values.get(c), tgt.values.get(lam(c))
        if a in (0, 1, 0.0, 1.0) and b in (0, 1, 0.0, 1.0):
            out.append(c)
    return out


def adjudication_load(src: ObligationVector, tgt: ObligationVector,
                      lam: LambdaT) -> List[str]:
    """Invariant coordinates carrying '?' on either side."""
    return [c for c in lam.source
            if src.values.get(c) == UNKNOWN or tgt.values.get(lam(c)) == UNKNOWN]


def delta_P(claim_tgt, transported_claim,
            src: ObligationVector, tgt: ObligationVector,
            lam: LambdaT) -> Dict[str, object]:
    """The proof-summary discrepancy, with CLAIM and OBLIGATION separated.

    Round-6 B04: branching first on an empty comparable set returned
    `unavailable` even when the transported claim definitely mismatched, which
    contradicts the target construct, where claim equality is an independent
    requirement. The claim component is therefore decisive on its own, and only
    the OBLIGATION component can be unavailable.

    Returns a typed record: (delta_claim, delta_obligation, delta_P,
    adjudication load). `delta_P` is the binary any-mismatch indicator of the
    manuscript: 1 if the claim mismatches or any comparable coordinate differs.
    The per-coordinate Hamming fraction is reported alongside as
    `hamming_obligation` for interpretation, never as the decision."""
    K = comparable_coordinates(src, tgt, lam)
    adj = adjudication_load(src, tgt, lam)
    delta_claim = 1.0 if claim_tgt != transported_claim else 0.0
    if not K:
        delta_obl: object = UNAVAILABLE
        hamming: object = UNAVAILABLE
        mismatched: List[str] = []
    else:
        mismatched = [c for c in K if src.values[c] != tgt.values[lam(c)]]
        delta_obl = 1.0 if mismatched else 0.0
        hamming = len(mismatched) / len(K)
    if delta_claim == 1.0:
        overall: object = 1.0                     # claim mismatch is decisive
    elif delta_obl is UNAVAILABLE:
        overall = UNAVAILABLE
    else:
        overall = delta_obl
    return {"delta_claim": delta_claim, "delta_obligation": delta_obl,
            "delta_P": overall, "hamming_obligation": hamming,
            "claim_mismatch": bool(delta_claim), "n_comparable": len(K),
            "mismatched": mismatched, "adjudication": adj,
            "presentation_specific_src": list(src.presentation_specific),
            "presentation_specific_tgt": list(tgt.presentation_specific)}
