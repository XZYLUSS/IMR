"""
imr/core.py  --  Reference implementation of the IMR measurement core.
Discharges the I2 reference-implementation contract (I2 sec L) at corpus scale.

Frozen objects (IMR v1.0 / I1-I2):
  - typed answer spaces with equality kernel and per-type distance d in [0,1]
  - the certified-transformation groupoid G_IMR via answer-transport maps tau_T
  - functorial laws: tau_id = id ; tau_{U o T} = tau_U o tau_T ; tau_{T^-1} = tau_T^-1
  - a non-coercing, family-specific MODEL-OUTPUT parser (not a gold-string parser):
        Pi_x : free text -> (typed_answer, obligation_flags) | PARSE_FAIL
  - the eight family unit-test invariants (I2 Def M.1)
  - the two-stage public failure router (I2 sec I)

No model is run by this module. All checks act on frozen golds.

Post-review hardening (I3 TGZ audit):
  FB-2  invariant 2 is now a genuine composition law tau_{T2 o T1} = tau_T2 o tau_T1
        checked over two DISTINCT transforms against a frozen composite gold
        (compose_transforms / check_composition_law), not a round-trip.
  FB-4  the parser is a family-specific accepted-form parser with typed,
        non-coercing value extraction and coordinate aliases (make_parser).
  M-1   the Stage-1 router checks `type` before `canonical` BY DESIGN, because
        canonical equality is undefined on a wrong-typed output (see erratum below).
  M-2   invariant 7 replays a per-family ANTI-CERTIFICATE (the specific obstruction),
        not a bare answer-difference (anti_* checkers).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import sympy as sp

PARSE_FAIL = None  # the unique typed parse-failure sentinel

# --------------------------------------------------------------------------
# Answer spaces: equality kernel + normalized distance d in [0,1]
# --------------------------------------------------------------------------
def _key(x: Any) -> Any:
    """Hashable canonical key for golds (rationals as Fraction, polys as tuples)."""
    if isinstance(x, (list, tuple)):
        return tuple(_key(z) for z in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _key(v)) for k, v in x.items()))
    if isinstance(x, str) and re.fullmatch(r"-?\d+/\d+", x):
        try:
            return Fraction(x)        # only pure rational literals, never expressions
        except ZeroDivisionError:
            return x                  # never coerce a zero-denominator token
    return x

def eq_tuple(a: Dict, b: Dict, coords: List[str]) -> bool:
    return all(_key(a.get(c)) == _key(b.get(c)) for c in coords)

def dist_tuple(a: Dict, b: Dict, coords: List[str]) -> float:
    """Normalized count of disagreeing invariant coordinates (QuotientClass metric)."""
    diff = sum(0 if _key(a.get(c)) == _key(b.get(c)) else 1 for c in coords)
    return diff / len(coords)

def dist_compound(a: Dict, b: Dict, weights: Dict[str, float]) -> float:
    """Weighted compound metric (e.g. BooleanWithWitness x Scalar). Weights sum to 1."""
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    return sum(w * (0.0 if _key(a.get(c)) == _key(b.get(c)) else 1.0)
               for c, w in weights.items())

# --------------------------------------------------------------------------
# A non-coercing, family-specific MODEL-OUTPUT parser  (FB-4)
# --------------------------------------------------------------------------
# Per-coordinate type tags drive a *typed* extraction that refuses to coerce.
#   "int"      : accept only -?\d+              (a bare 1/2 or x+1 -> PARSE_FAIL)
#   "rational" : accept only -?\d+(/\d+)?       (an expression      -> PARSE_FAIL)
#   "label"    : a symbolic token / set / poly  (stored as a normalized string)
_VALUE_PATTERNS = {
    "int":      r"[+-]?\d+",
    "rational": r"[+-]?\d+(?:/\d+)?",
    "label":    r"[^;,\n]+",
}

def _norm_label(s: str) -> str:
    s = s.strip().strip(".").strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()      # strip stray math delimiters
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s).strip()
    s = re.sub(r"\s+", "", s)                      # canonicalise spacing in tokens
    return s

# explanatory connectives: a label value must stop before these (FB-2)
_CONNECTIVES = re.compile(
    r"\s+(?:because|since|from|by\s|due\s+to|owing\s+to|therefore|hence|so\s+that|as\s|which\b|where\b)",
    re.IGNORECASE)

def _strip_explanation(raw: str) -> str:
    m = _CONNECTIVES.search(raw)
    return raw[:m.start()] if m else raw

def make_parser(coords: List[str],
                type_spec: Dict[str, str],
                required: List[str],
                aliases: Optional[Dict[str, List[str]]] = None,
                mechanism_cues: Optional[List[str]] = None,
                witness_cues: Optional[List[str]] = None) -> Callable[[str], Optional[Tuple[Dict, Dict]]]:
    """
    Build Pi_x for one family. Parses realistic model phrasings such as
    'the Wilson loop is W = 1', 'c_1 = 1', '\\boxed{7}', 'trace = 0; det = 1',
    while enforcing the per-coordinate type (a non-coercing extractor).
    Returns (found, obligation_flags) or PARSE_FAIL.
    """
    aliases = aliases or {}
    mechanism_cues = mechanism_cues or ["because", "since", "invariant under", "by ", "mechanism"]
    witness_cues   = witness_cues   or ["witness", "explicit", "period", "jacobian", "stokes", "exact form"]

    def _value_from(raw: str, tag: str, vpat: str):
        """Turn a captured value string into ('ok',val)|('coerce',None) by type."""
        if tag in ("int", "rational"):
            tok = re.match(r"\s*(\S+)", raw)
            if not tok:
                return ("coerce", None)
            t = tok.group(1).strip("().,;:[]{}$")          # peel wrapping punctuation
            if re.fullmatch(vpat, t):
                return ("ok", t)
            return ("coerce", None)                        # whole token wrong type -> non-coercion
        return ("ok", _norm_label(_strip_explanation(raw)))  # label: stop before connectives

    def _extract_one(text: str, coord: str) -> Tuple[str, Optional[str]]:
        """Return (status, value): status in {'ok','coerce','missing'}."""
        keys = [coord] + list(aliases.get(coord, []))
        tag = type_spec.get(coord, "label")
        vpat = _VALUE_PATTERNS[tag]
        saw_coerce = False
        for k in keys:
            kpat = re.escape(k)
            last = k[-1]
            bnd = r"(?![A-Za-z0-9_])" if (last.isalnum() or last == "_") else ""  # no \b after punctuation aliases
            # Pass A: explicit '=' or ':' separator (a gap may skip filler like 'is W ')
            mA = re.search(rf"{kpat}{bnd}[^=:\n]{{0,40}}?[:=]\s*([^;\n]+)", text, re.IGNORECASE)
            if mA:
                st, val = _value_from(mA.group(1), tag, vpat)
                if st == "ok":
                    return ("ok", val)
                saw_coerce = True
            # Pass B: ' is ' separator (only when no '=' / ':' lies between key and value)
            mB = re.search(rf"{kpat}{bnd}\s+is\s+([^=;:\n]+)", text, re.IGNORECASE)
            if mB:
                st, val = _value_from(mB.group(1), tag, vpat)
                if st == "ok":
                    return ("ok", val)
                saw_coerce = True
        # boxed single-answer form, only meaningful for a 1-coordinate family
        if len(coords) == 1:
            m = re.search(r"\\boxed\{\s*([^}]+)\s*\}", text)
            if m:
                st, val = _value_from(m.group(1).strip(), tag, vpat)
                if st == "ok":
                    return ("ok", val)
                saw_coerce = True
        return ("coerce", None) if saw_coerce else ("missing", None)

    def Pi(text: str) -> Optional[Tuple[Dict, Dict]]:
        if not isinstance(text, str) or not text.strip():
            return PARSE_FAIL
        found: Dict[str, Any] = {}
        for c in coords:
            status, val = _extract_one(text, c)
            if status == "coerce":
                return PARSE_FAIL                 # non-coercion: typed mismatch -> parse/type
            if status == "ok":
                tag = type_spec.get(c, "label")
                if tag == "int":
                    found[c] = int(val)
                elif tag == "rational":
                    found[c] = int(val) if re.fullmatch(r"[+-]?\d+", val) else val
                else:
                    found[c] = val
        if any(c not in found for c in coords):
            return PARSE_FAIL                     # missing coordinate -> type
        low = text.lower()
        flags = {"mechanism": any(cue in low for cue in mechanism_cues),
                 "witness":   any(cue in low for cue in witness_cues)}
        return (found, flags)
    return Pi

# --------------------------------------------------------------------------
# Transport functor + functorial laws (invariants 1-4 of Def M.1)
# --------------------------------------------------------------------------
@dataclass
class Transform:
    tid: str
    name: str
    tau: Callable[[Dict], Dict]          # answer-transport map
    tau_inv: Callable[[Dict], Dict]      # its inverse on the certified scope
    transported_gold: Dict               # frozen pre-run

def tau_identity(ans: Dict) -> Dict:
    return dict(ans)

def compose_transforms(T1: Transform, T2: Transform, composite_gold: Dict) -> Transform:
    """The composite arrow T2 o T1 of the groupoid, with composed transport."""
    return Transform(f"{T2.tid}o{T1.tid}", f"{T2.name} o {T1.name}",
                     tau=lambda a: T2.tau(T1.tau(a)),
                     tau_inv=lambda a: T1.tau_inv(T2.tau_inv(a)),
                     transported_gold=composite_gold)

def check_functorial_laws(canonical_gold: Dict, T: Transform,
                          eqf: Callable[[Dict, Dict], bool]) -> Dict[str, bool]:
    """Per-transform laws: identity, inverse, gold-transport (composition is family-level)."""
    res = {}
    res["identity"]       = eqf(tau_identity(canonical_gold), canonical_gold)
    res["inverse"]        = eqf(T.tau_inv(T.tau(canonical_gold)), canonical_gold)
    res["gold_transport"] = eqf(T.tau(canonical_gold), T.transported_gold)
    return res

def check_composition_law(canonical_gold: Dict, T1: Transform, T2: Transform,
                          composite_gold: Dict,
                          eqf: Callable[[Dict, Dict], bool]) -> bool:
    """
    Genuine composition law (FB-2): for the two DISTINCT transforms T1, T2,
        tau_{T2 o T1}(a_x)  ==  (tau_T2 o tau_T1)(a_x)  ==  composite_gold,
    where composite_gold is FROZEN independently in the registry. This exercises
    the composite arrow rather than a self-inverse round-trip.
    """
    composite = compose_transforms(T1, T2, composite_gold)
    direct_chain = T2.tau(T1.tau(canonical_gold))         # (tau_T2 o tau_T1)(a_x)
    via_composite = composite.tau(canonical_gold)         # tau_{T2 o T1}(a_x)
    law = eqf(direct_chain, via_composite)                # functoriality of composition
    anchored = eqf(direct_chain, composite_gold) and eqf(via_composite, composite_gold)
    return bool(law and anchored)

# --------------------------------------------------------------------------
# Anti-certificate replay for the negative control (M-2)
# --------------------------------------------------------------------------
# Each checker REPLAYS the specific obstruction that places n(x) outside the
# orbit -- it does not merely observe that two answers differ.
def anti_orbit_invariant(invariant_coords: List[str]) -> Callable[[Dict, Dict], bool]:
    """The certified orbit invariant (trace/character/spectrum/Chern/...) must
    DIFFER on n(x): conjugation/basis-change/orientation preserves it, so a
    differing value witnesses that no orbit morphism maps gold -> neg."""
    def chk(neg: Dict, gold: Dict) -> bool:
        return any(_key(neg.get(c)) != _key(gold.get(c)) for c in invariant_coords)
    return chk

def anti_not_integer(coord: str) -> Callable[[Dict, Dict], bool]:
    """Topological-charge control: the neg value is NOT an integer (a chart flux,
    not a Chern number) -- Chern-Weil/Dirac quantisation is violated."""
    def chk(neg: Dict, gold: Dict) -> bool:
        v = neg.get(coord)
        try:
            f = Fraction(v) if isinstance(v, str) else Fraction(v)
        except Exception:
            return False
        return f.denominator != 1
    return chk

def anti_classical_law(origin_coord: str, density_coord: str) -> Callable[[Dict, Dict], bool]:
    """Anomaly control: n(x) asserts the CLASSICAL conservation law
    (origin = classical_action AND density = 0), i.e. it omits the measure
    Jacobian -- the action-vs-measure confusion, not a convention rewrite."""
    def chk(neg: Dict, gold: Dict) -> bool:
        return (str(neg.get(origin_coord)) == "classical_action"
                and _key(neg.get(density_coord)) == _key(0)
                and str(gold.get(origin_coord)) == "measure_jacobian")
    return chk

def anti_unconditional(coord: str) -> Callable[[Dict, Dict], bool]:
    """Boundary control: n(x) claims the boundary term vanishes '{always}'
    (unconditionally), contradicting the certified conditional set."""
    def chk(neg: Dict, gold: Dict) -> bool:
        return "{always}" in str(neg.get(coord)) and "{always}" not in str(gold.get(coord))
    return chk

def anti_nonnatural(markers: Tuple[str, ...] = ("entry", "eigvec", "basis-dependent",
                                                "component")) -> Callable[[Dict, Dict], bool]:
    """Naturality control: n(x) reports a coordinate-dependent (non-natural)
    readout -- an entry / eigenvector component -- that changes under change of
    basis, hence is not transportable as the identity."""
    def chk(neg: Dict, gold: Dict) -> bool:
        blob = " ".join(str(v) for v in neg.values()).lower()
        return any(m in blob for m in markers)
    return chk

# --------------------------------------------------------------------------
# Family object
# --------------------------------------------------------------------------
@dataclass
class Family:
    family_id: str
    module: str
    rung: str
    coords: List[str]                                   # answer-space coordinates
    canonical_gold: Dict
    transforms: List[Transform]
    eqf: Callable[[Dict, Dict], bool]                   # equality kernel
    distf: Callable[[Dict, Dict], float]                # metric d
    normalizer: Callable[[str], Optional[Tuple[Dict, Dict]]]  # Pi_x (text -> (ans,flags)|PARSE_FAIL)
    neg_control_answer: Dict                            # the (different) correct answer for n(x)
    neg_equivalence_claim: Dict                         # what a faulty checker WOULD pass for n(x)
    obligations_required: List[str]                     # required obligation coordinates
    composite_gold: Dict                                # frozen gold for tau_{T2 o T1} (FB-2)
    anti_certificate: Callable[[Dict, Dict], bool]      # replays the obstruction (M-2)
    # --- richer registry schema (M-3) ---
    answer_type: Dict = field(default_factory=dict)
    source_stratum: str = "synthetic_algorithmic"
    type_spec: Dict[str, str] = field(default_factory=dict)
    aliases: Dict[str, List[str]] = field(default_factory=dict)
    allowed_forms: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    transform_records: List[Dict] = field(default_factory=list)
    presentation: str = ""

    # --- the eight unit-test invariants (Def M.1) ---
    def audit(self) -> Dict[str, Any]:
        assert len(self.transforms) >= 2, f"{self.family_id}: audit requires >= 2 certified transforms"
        report: Dict[str, Any] = {"family_id": self.family_id, "checks": {}}
        # invariants 1,3,4 over every certified transform
        for T in self.transforms:
            report["checks"][f"functorial[{T.tid}]"] = check_functorial_laws(
                self.canonical_gold, T, self.eqf)
        # invariant 2: genuine composition law over the (T1,T2) pair (FB-2)
        T1, T2 = self.transforms[0], self.transforms[1]
        report["checks"]["composition"] = check_composition_law(
            self.canonical_gold, T1, T2, self.composite_gold, self.eqf)
        # invariant 5: parser soundness on the gold form
        gold_text = render_gold(self.canonical_gold, self.coords)
        parsed = self.normalizer(gold_text)
        report["checks"]["parser_soundness"] = (parsed is not PARSE_FAIL
                                                 and self.eqf(parsed[0], self.canonical_gold))
        # invariant 6: parser non-coercion on prose with no typed answer
        report["checks"]["parser_non_coercion"] = (
            self.normalizer("there is no typed answer in this prose") is PARSE_FAIL)
        # invariant 7: negative-control rejection -- replay the anti-certificate (M-2)
        rejected = not self.eqf(self.neg_control_answer, self.canonical_gold)
        anti_ok = bool(self.anti_certificate(self.neg_control_answer, self.canonical_gold))
        report["checks"]["neg_control_rejected"] = rejected and anti_ok
        # invariant 8: evaluator invariance D_eval = 0 on frozen gold-equivalence pairs
        report["checks"]["kernel_gold_self_consistency"] = self.kernel_gold_self_consistency() == 0.0
        report["passed"] = _all_true(report["checks"])
        return report

    def kernel_gold_self_consistency(self) -> float:
        """KERNEL/GOLD SELF-CONSISTENCY, not a live evaluator audit.

        This compares frozen gold objects using the same equality kernel that
        defined them, so a zero here says the apparatus does not contradict
        itself. It says NOTHING about the presentation sensitivity of a
        renderer, parser, human grader or LLM judge; auditing those is a
        separate procedure that is not implemented in this bundle. The name was
        `evaluator_audit`, which overstated it."""
        pairs = [(self.canonical_gold, T.transported_gold) for T in self.transforms]
        pairs.append((self.canonical_gold, self.composite_gold))
        disagree = sum(1 for ga, gb in pairs if not self.eqf(ga, gb))
        d_eval = disagree / max(1, len(pairs))
        return d_eval

def _all_true(d) -> bool:
    if isinstance(d, dict):
        return all(_all_true(v) for v in d.values())
    return bool(d)

def render_gold(gold: Dict, coords: List[str]) -> str:
    return "; ".join(f"{c} = {gold[c]}" for c in coords)

# --------------------------------------------------------------------------
# Two-stage public failure router (I2 sec I)
# --------------------------------------------------------------------------
# ERRATUM (M-1, relative to I2): I3 refines the Stage-1 order by checking `type`
# BEFORE `canonical`. Canonical equality is undefined on a wrong-typed output
# (one cannot ask whether a returned matrix equals a scalar gold), so typedness
# is logically prior to correctness. The Stage-0 {evaluator, generator} gates and
# the remaining Stage-1 categories are unchanged from the frozen I2 taxonomy.
STAGE0 = ["evaluator", "generator"]   # harness gates, charged before any model label
STAGE1 = ["refusal", "parse", "type", "canonical", "transport", "mechanism", "witness"]

def route_failure(*, eval_gate_ok: bool, harness_ok: bool, refused: bool,
                  parsed, typed_ok: bool, canonical_ok: bool, transport_ok: bool,
                  mechanism_ok: bool, witness_ok: bool, required: List[str]) -> str:
    # Stage 0 -- harness gates
    if not eval_gate_ok:  return "evaluator"
    if not harness_ok:    return "generator"
    # Stage 1 -- per-response routing, deterministic first match (type before canonical)
    if refused:                       return "refusal"
    if parsed is PARSE_FAIL:          return "parse"
    if not typed_ok:                  return "type"
    if not canonical_ok:              return "canonical"
    if not transport_ok:              return "transport"
    if "mechanism" in required and not mechanism_ok: return "mechanism"
    if "witness" in required and not witness_ok:     return "witness"
    return "correct"
