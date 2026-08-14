"""
imr/families.py  --  the 24-family pilot bank (I3 base ten + I4 expansion fourteen),
encoded against core.py, balanced at four families per obstruction module (A-F).
Six worked families (from I2 sec K) + four expansion families instantiated in I3.
Golds are the values verified by direct computation in verify_all.py.

Post-review hardening (I3 TGZ audit):
  M-3  registry now carries the I2 schema per family: answer_type, source_stratum,
       type_spec, aliases, allowed_forms, exclusions, transform_records, composite_gold.
  FB-2 each family ships two DISTINCT transforms T1,T2 and a frozen composite_gold.
  FB-4 each family ships a typed, alias-aware MODEL-OUTPUT parser (make_parser).
  M-2  each family ships an anti_certificate that replays its specific obstruction.

I4 expansion (24-family pilot bank, balanced 4 per module A-F):
  +14 certified families; AN-004 uses a concrete scale a=2 in its anti-certificate;
  AM-003 states SU(2)-fundamental trace normalisation; BV-002/003/004 record the
  Euler-Lagrange operator as an equation class up to a nonzero scalar (mostly-minus eta).
"""
from typing import Dict, List
from fractions import Fraction
from core import (Family, Transform, tau_identity, eq_tuple, dist_tuple, dist_compound,
                  make_parser, anti_orbit_invariant, anti_not_integer, anti_classical_law,
                  anti_unconditional, anti_nonnatural)

# --------------------------------------------------------------------------
# AM-002 convention transport: an EXECUTABLE normalizer that equates the
# epsilon-FF and F.Ftilde presentations via the algebraic identity
#   eps^{mnrs} F_mn F_rs = 2 F_mn Ftilde^{mn}.
# The answer encodes (coeff_num, structure, origin), where coeff_num is the
# rational multiple of q^2/pi^2 and structure in {epsFF, FFtilde, none}. The
# physical density normalises to coeff_num * structure_factor (epsFF |-> 2).
# --------------------------------------------------------------------------
def _am_normal(coeff_num, structure):
    factor = {"epsFF": 2, "FFtilde": 1, "none": 0}.get(str(structure))
    if factor is None:
        return None
    f = coeff_num if isinstance(coeff_num, Fraction) else Fraction(str(coeff_num))
    return f * factor

def am002_family():
    coords = ["coeff_num", "structure", "origin"]
    canonical   = {"coeff_num": Fraction(1, 16), "structure": "epsFF",   "origin": "measure_jacobian"}
    transported = {"coeff_num": Fraction(1, 8),  "structure": "FFtilde", "origin": "measure_jacobian"}
    neg         = {"coeff_num": Fraction(0),     "structure": "none",    "origin": "classical_action"}

    def eqf(a, b):
        na, nb = _am_normal(a.get("coeff_num"), a.get("structure")), _am_normal(b.get("coeff_num"), b.get("structure"))
        return na is not None and na == nb and str(a.get("origin")) == str(b.get("origin"))
    def dist(a, b):
        return 0.0 if eqf(a, b) else 1.0

    # T1: a GENUINE non-identity convention transport  epsFF -> FFtilde
    def tau_T1(a):
        f = a.get("coeff_num"); f = f if isinstance(f, Fraction) else Fraction(str(f))
        if str(a.get("structure")) == "epsFF":
            return {"coeff_num": f * 2, "structure": "FFtilde", "origin": a.get("origin")}
        return dict(a)
    def tau_T1_inv(a):
        f = a.get("coeff_num"); f = f if isinstance(f, Fraction) else Fraction(str(f))
        if str(a.get("structure")) == "FFtilde":
            return {"coeff_num": f / 2, "structure": "epsFF", "origin": a.get("origin")}
        return dict(a)
    T1 = Transform("T1", "convention epsFF->FFtilde (non-identity)", tau_T1, tau_T1_inv, dict(transported))
    # T2: a metric/orientation convention that is the identity on the physical normal
    T2 = Transform("T2", "metric-orientation-convention", tau_identity, tau_identity, dict(canonical))
    composite_gold = dict(transported)   # tau_T2(tau_T1(canonical)) = FFtilde presentation

    parser = make_parser(coords, {"coeff_num": "rational", "structure": "label", "origin": "label"},
                         ["mechanism"],
                         aliases={"coeff_num": ["coefficient", "coeff"], "structure": ["form"], "origin": ["source"]})
    meta = _records("convention_ledger",
                    {"kind": "ConventionCoefficient", "encoding": ["coeff_num", "structure", "origin"]},
                    ["coefficient_in_stated_convention"], ["raw_coefficient_without_convention"],
                    "4D ABJ anomaly (under the convention ledger): rational coefficient of q^2/pi^2, "
                    "structure in {epsFF,FFtilde}, and origin.",
                    [{"id": "T1", "kind": "convention-epsFF->FFtilde", "transport": "non-identity (coeff x2, structure relabel)"},
                     {"id": "T2", "kind": "metric-orientation-convention", "transport": "identity-on-normal"}])
    return Family("IMR-AM-002", "D-anomaly-measure", "L1", coords, dict(canonical), [T1, T2],
                  eqf, dist, parser, dict(neg), {}, ["mechanism"], composite_gold,
                  anti_classical_law("origin", "coeff_num"),
                  type_spec={"coeff_num": "rational", "structure": "label", "origin": "label"},
                  aliases={"coeff_num": ["coefficient", "coeff"], "structure": ["form"], "origin": ["source"]},
                  **_filter(meta))

def _two_identity_transforms(gold: Dict):
    """Two DISTINCT certified arrows with identity transport (the invariant is
    preserved by both gauge/basis/orientation moves); composite gold is frozen."""
    T1 = Transform("T1", "transform-1", tau_identity, tau_identity, dict(gold))
    T2 = Transform("T2", "transform-2", tau_identity, tau_identity, dict(gold))
    return [T1, T2], dict(gold)        # transforms, composite_gold

def _records(stratum, answer_type, allowed_forms, exclusions, presentation, transforms_meta):
    return dict(source_stratum=stratum, answer_type=answer_type, allowed_forms=allowed_forms,
                exclusions=exclusions, presentation=presentation, transform_records=transforms_meta)

def quotient_family(fid, module, coords, gold, neg_gold, *, type_spec, aliases,
                    anti_cert, meta, rung="L2", required=("mechanism",)):
    eqf  = lambda a, b: eq_tuple(a, b, coords)
    dist = lambda a, b: dist_tuple(a, b, coords)
    transforms, comp = _two_identity_transforms(gold)
    parser = make_parser(coords, type_spec, list(required), aliases)
    return Family(fid, module, rung, coords, dict(gold), transforms, eqf, dist, parser,
                  dict(neg_gold), {}, list(required), comp, anti_cert,
                  type_spec=type_spec, aliases=aliases, **_filter(meta))

def compound_family(fid, module, coords, gold, neg_gold, *, weights, required, type_spec,
                    aliases, anti_cert, meta, rung="L2"):
    eqf  = lambda a, b: eq_tuple(a, b, coords)
    dist = lambda a, b: dist_compound(a, b, weights)
    transforms, comp = _two_identity_transforms(gold)
    parser = make_parser(coords, type_spec, list(required), aliases)
    return Family(fid, module, rung, coords, dict(gold), transforms, eqf, dist, parser,
                  dict(neg_gold), {}, list(required), comp, anti_cert,
                  type_spec=type_spec, aliases=aliases, **_filter(meta))

def _filter(meta):
    keep = ("answer_type", "source_stratum", "allowed_forms", "exclusions",
            "transform_records", "presentation")
    return {k: v for k, v in meta.items() if k in keep}

# tolerate an accidental kw in quotient_family

# ===================== the base ten families (I3) ====================
def build_corpus() -> Dict[str, Family]:
    fams: Dict[str, Family] = {}

    # ---- Module A : quotient / descent ----
    fams["IMR-QD-001"] = compound_family(
        "IMR-QD-001", "A-quotient-descent", ["trace", "det", "char_poly"],
        {"trace": 0, "det": 1, "char_poly": "lam^2+1"},
        {"trace": 2, "det": 1, "char_poly": "lam^2-2lam+2"},               # H+I (trace 2)
        weights={"trace": 1/3, "det": 1/3, "char_poly": 1/3}, required=["mechanism"],
        type_spec={"trace": "int", "det": "int", "char_poly": "label"},
        aliases={"trace": ["tr"], "det": ["determinant"], "char_poly": ["chi", "characteristic polynomial"]},
        anti_cert=anti_orbit_invariant(["trace"]),
        meta=_records("synthetic_algorithmic", {"kind": "QuotientClass", "encoding": ["trace", "det", "char_poly"]},
                      ["conjugacy_class_invariant"], ["raw_matrix_entry"],
                      "Report the conjugation invariants (trace, det, char. poly.) of the quarter-turn.",
                      [{"id": "T1", "kind": "conjugation-by-shear", "transport": "identity"},
                       {"id": "T2", "kind": "conjugation-by-diagonal", "transport": "identity"}]))

    fams["IMR-QD-002"] = quotient_family(
        "IMR-QD-002", "A-quotient-descent", ["wilson_loop"],
        {"wilson_loop": 1},                                                # 2cos(pi/3)=1
        {"wilson_loop": 0},                                                # 2cos(pi/2)=0
        type_spec={"wilson_loop": "int"},
        aliases={"wilson_loop": ["W", "wilson loop", "trU", "tr U", "character"]},
        anti_cert=anti_orbit_invariant(["wilson_loop"]),
        meta=_records("synthetic_algorithmic", {"kind": "QuotientClass", "encoding": ["wilson_loop"]},
                      ["conjugacy_class_invariant"], ["raw_matrix_entry", "gauge_dependent_component"],
                      "Report the gauge-invariant Wilson loop W = tr U of an SU(2) holonomy.",
                      [{"id": "T1", "kind": "conjugation-generic", "transport": "identity-on-class"},
                       {"id": "T2", "kind": "conjugation-su2-explicit", "transport": "identity-on-class"}]))

    # ---- Module B : local / global ----
    fams["IMR-LG-001"] = compound_family(
        "IMR-LG-001", "B-local-global", ["globally_exact", "period"],
        {"globally_exact": "no", "period": "2pi"},
        {"globally_exact": "yes", "period": 0},                           # contractible arc
        weights={"globally_exact": 0.5, "period": 0.5}, required=["witness"],
        type_spec={"globally_exact": "label", "period": "label"},
        aliases={"period": ["integral", "holonomy"], "globally_exact": ["exact"]},
        anti_cert=anti_orbit_invariant(["period", "globally_exact"]),
        meta=_records("synthetic_algorithmic", {"kind": "BooleanWithWitness", "encoding": ["globally_exact", "period"]},
                      ["closed_loop_period"], ["contractible_arc_period"],
                      "Is d(theta) globally exact on S^1, and what is its period?",
                      [{"id": "T1", "kind": "shift-rep-by-exact", "transport": "identity"},
                       {"id": "T2", "kind": "orientation-preserving-reparam", "transport": "identity"}]))

    fams["IMR-LG-002"] = compound_family(
        "IMR-LG-002", "B-local-global", ["chern_number"],
        {"chern_number": 1},                                              # (1/2pi)int = 1
        {"chern_number": "1/2"},                                          # hemisphere = 1/2
        weights={"chern_number": 1.0}, required=["witness"],
        type_spec={"chern_number": "int"},
        aliases={"chern_number": ["c_1", "c1", "chern number", "first chern number"]},
        anti_cert=anti_not_integer("chern_number"),
        meta=_records("synthetic_algorithmic", {"kind": "Integer", "encoding": ["chern_number"]},
                      ["closed_manifold_integral"], ["single_chart_flux"],
                      "Compute the first Chern number c_1 = (1/2pi) int_{S^2} F.",
                      [{"id": "T1", "kind": "add-exact-form", "transport": "identity"},
                       {"id": "T2", "kind": "orientation-preserving-reparam", "transport": "identity"}]))

    # ---- Module C : analytic / regularization ----
    fams["IMR-AN-001"] = compound_family(
        "IMR-AN-001", "C-analytic-regularization", ["lim_int", "int_lim", "interchange_valid"],
        {"lim_int": 1, "int_lim": 0, "interchange_valid": "no"},
        {"lim_int": 1, "int_lim": 1, "interchange_valid": "yes"},         # a DCT-valid sequence
        weights={"lim_int": 1/3, "int_lim": 1/3, "interchange_valid": 1/3}, required=["mechanism"],
        type_spec={"lim_int": "int", "int_lim": "int", "interchange_valid": "label"},
        aliases={"interchange_valid": ["interchange", "may interchange"]},
        anti_cert=anti_orbit_invariant(["interchange_valid", "int_lim"]),
        meta=_records("synthetic_algorithmic", {"kind": "Compound", "encoding": ["lim_int", "int_lim", "interchange_valid"]},
                      ["limit_integral_values"], ["dct_valid_sequence"],
                      "For f_n = n*1_{[0,1/n]}: give lim-int, int-lim, and whether interchange is valid.",
                      [{"id": "T1", "kind": "rescale-domain", "transport": "identity"},
                       {"id": "T2", "kind": "shift-index", "transport": "identity"}]))

    # ---- Module D : anomaly / measure ----
    fams["IMR-AM-001"] = compound_family(
        "IMR-AM-001", "D-anomaly-measure", ["physical_density", "origin"],
        {"physical_density": "(q/pi)F01", "origin": "measure_jacobian"},
        {"physical_density": 0, "origin": "classical_action"},            # false conservation
        weights={"physical_density": 0.5, "origin": 0.5}, required=["mechanism"], rung="L1",
        type_spec={"physical_density": "label", "origin": "label"},
        aliases={"physical_density": ["density", "divergence"], "origin": ["source"]},
        anti_cert=anti_classical_law("origin", "physical_density"),
        meta=_records("convention_ledger", {"kind": "StructuralLabel", "encoding": ["physical_density", "origin"]},
                      ["anomaly_density_in_convention"], ["classical_conservation"],
                      "2D axial anomaly: give the anomalous divergence density and its origin.",
                      [{"id": "T1", "kind": "ledger-rewrite", "transport": "identity"},
                       {"id": "T2", "kind": "convention-change", "transport": "identity"}]))

    fams["IMR-AM-002"] = am002_family()

    # ---- Module E : boundary / variation ----
    fams["IMR-BV-001"] = compound_family(
        "IMR-BV-001", "E-boundary-variation", ["bulk", "boundary_vanishes_iff"],
        {"bulk": "-lap(phi)", "boundary_vanishes_iff": "{Dirichlet,Neumann,closed,compact}"},
        {"bulk": "-lap(phi)", "boundary_vanishes_iff": "{always}"},       # free-endpoint: live bdy term
        weights={"bulk": 0.5, "boundary_vanishes_iff": 0.5}, required=["witness"],
        type_spec={"bulk": "label", "boundary_vanishes_iff": "label"},
        aliases={"bulk": ["euler_lagrange", "bulk_term"], "boundary_vanishes_iff": ["boundary", "bdy"]},
        anti_cert=anti_unconditional("boundary_vanishes_iff"),
        meta=_records("synthetic_algorithmic", {"kind": "Compound", "encoding": ["bulk", "boundary_vanishes_iff"]},
                      ["bulk_plus_conditional_boundary"], ["unconditional_vanishing"],
                      "Vary S[phi]=1/2 int |grad phi|^2: give the bulk term and the boundary-vanishing condition.",
                      [{"id": "T1", "kind": "integrate-by-parts-variant", "transport": "identity"},
                       {"id": "T2", "kind": "relabel-coordinates", "transport": "identity"}]))

    # ---- Module F : transport / naturality ----
    fams["IMR-TN-001"] = compound_family(
        "IMR-TN-001", "F-transport-naturality", ["trace", "det", "char_poly", "spec"],
        {"trace": 5, "det": 6, "char_poly": "lam^2-5lam+6", "spec": "{2,3}"},
        {"trace": 5, "det": 6, "char_poly": "lam^2-5lam+6", "spec": "entry(1,2)=1"},  # non-natural
        weights={"trace": 0.25, "det": 0.25, "char_poly": 0.25, "spec": 0.25}, required=["mechanism"],
        type_spec={"trace": "int", "det": "int", "char_poly": "label", "spec": "label"},
        aliases={"trace": ["tr"], "det": ["determinant"], "spec": ["spectrum", "eigenvalues"]},
        anti_cert=anti_nonnatural(),
        meta=_records("synthetic_algorithmic", {"kind": "QuotientClass", "encoding": ["trace", "det", "char_poly", "spec"]},
                      ["similarity_invariant"], ["matrix_entry"],
                      "Report the similarity invariants (trace, det, char. poly., spectrum).",
                      [{"id": "T1", "kind": "conjugation-A", "transport": "identity"},
                       {"id": "T2", "kind": "conjugation-B", "transport": "identity"}]))

    fams["IMR-TN-002"] = quotient_family(
        "IMR-TN-002", "F-transport-naturality", ["tr_AB", "det_AB"],
        {"tr_AB": 7, "det_AB": -42},
        {"tr_AB": "eigvec[0]", "det_AB": "basis-dependent"},              # non-natural eigvec comp
        type_spec={"tr_AB": "int", "det_AB": "int"},
        aliases={"tr_AB": ["tr(AB)", "trace of AB", "trAB", "trace"], "det_AB": ["det(AB)", "determinant of AB", "detAB", "determinant"]},
        anti_cert=anti_nonnatural(), required=["mechanism"],
        meta=_records("synthetic_algorithmic", {"kind": "Compound", "encoding": ["tr_AB", "det_AB"]},
                      ["natural_invariant"], ["entry", "eigenvector_component"],
                      "Report tr(AB) and det(AB) for the given A,B.",
                      [{"id": "T1", "kind": "simultaneous-basis-change", "transport": "identity"},
                       {"id": "T2", "kind": "order-swap-AB-BA", "transport": "identity"}]))

    # ===================== I4 EXPANSION (toward the 24-30 pilot bank) =====================
    # Every gold below is independently re-derived in verify_all.py (SymPy). Transports are
    # identity where the answer is an invariant (the certified move preserves it); AM-002 in
    # the base ten remains the worked non-identity (convention) transport exemplar.

    # ---- Module A : quotient / descent ----
    fams["IMR-QD-003"] = quotient_family(
        "IMR-QD-003", "A-quotient-descent", ["casimir"],
        {"casimir": 2}, {"casimir": "(J_z)_{11} generator component"},     # spin-1 Casimir; neg = a basis-dependent GENERATOR component, not an entry of C2=2I (NB-FB-1)
        type_spec={"casimir": "int"},
        aliases={"casimir": ["C2", "C_2", "quadratic casimir", "casimir"]},
        anti_cert=anti_nonnatural(),
        meta=_records("synthetic_algorithmic", {"kind": "Integer", "encoding": ["casimir"]},
                      ["rep_invariant"], ["matrix_entry", "basis_component"],
                      "Quadratic Casimir C_2 = j(j+1) of the SU(2) spin-1 representation.",
                      [{"id": "T1", "kind": "change-of-basis-in-rep", "transport": "identity"},
                       {"id": "T2", "kind": "unitary-equivalence-of-rep", "transport": "identity"}]))

    fams["IMR-QD-004"] = compound_family(
        "IMR-QD-004", "A-quotient-descent", ["trace", "det"],
        {"trace": 1, "det": 1}, {"trace": 2, "det": 1},                   # SO(3) by pi/2; neg different class
        weights={"trace": 0.5, "det": 0.5}, required=["mechanism"],
        type_spec={"trace": "int", "det": "int"},
        aliases={"trace": ["tr"], "det": ["determinant"]},
        anti_cert=anti_orbit_invariant(["trace"]),
        meta=_records("synthetic_algorithmic", {"kind": "QuotientClass", "encoding": ["trace", "det"]},
                      ["conjugacy_invariant"], ["matrix_entry"],
                      "Trace and determinant of the SO(3) rotation by pi/2 (1+2cos t, det 1).",
                      [{"id": "T1", "kind": "conjugation-A", "transport": "identity"},
                       {"id": "T2", "kind": "conjugation-B", "transport": "identity"}]))

    # ---- Module B : local / global ----
    fams["IMR-LG-004"] = quotient_family(
        "IMR-LG-004", "B-local-global", ["euler_char"],
        {"euler_char": 2}, {"euler_char": 1},                            # chi(S^2)=2 via Gauss-Bonnet; neg disk
        type_spec={"euler_char": "int"},
        aliases={"euler_char": ["chi", "euler characteristic", "euler number"]},
        anti_cert=anti_orbit_invariant(["euler_char"]),
        meta=_records("synthetic_algorithmic", {"kind": "Integer", "encoding": ["euler_char"]},
                      ["closed_surface_gauss_bonnet"], ["bounded_domain"],
                      "Euler characteristic of S^2 via Gauss-Bonnet: (1/2pi) int K dA = 2.",
                      [{"id": "T1", "kind": "metric-rescaling", "transport": "identity"},
                       {"id": "T2", "kind": "orientation-preserving-diffeo", "transport": "identity"}]))

    fams["IMR-LG-005"] = quotient_family(
        "IMR-LG-005", "B-local-global", ["euler_char"],
        {"euler_char": 0}, {"euler_char": 2},                            # chi(T^2)=0; neg claims sphere
        type_spec={"euler_char": "int"},
        aliases={"euler_char": ["chi", "euler characteristic", "euler number"]},
        anti_cert=anti_orbit_invariant(["euler_char"]),
        meta=_records("synthetic_algorithmic", {"kind": "Integer", "encoding": ["euler_char"]},
                      ["closed_surface_gauss_bonnet"], ["wrong_genus"],
                      "Euler characteristic of the torus T^2: chi = 2 - 2g = 0 (g=1).",
                      [{"id": "T1", "kind": "flat-metric-rescaling", "transport": "identity"},
                       {"id": "T2", "kind": "conformal-change", "transport": "identity"}]))

    # ---- Module C : analytic / regularization ----
    fams["IMR-AN-002"] = quotient_family(
        "IMR-AN-002", "C-analytic-regularization", ["residue"],
        {"residue": 1}, {"residue": 0},                                  # res(1/z,0)=1; neg res(1/z^2,0)=0
        type_spec={"residue": "int"},
        aliases={"residue": ["res"]},
        anti_cert=anti_orbit_invariant(["residue"]),
        meta=_records("synthetic_algorithmic", {"kind": "Integer", "encoding": ["residue"]},
                      ["contour_residue"], ["regular_point_residue"],
                      "Residue of 1/z at 0 (so oint_{|z|=1} dz/z = 2 pi i): residue = 1.",
                      [{"id": "T1", "kind": "contour-deformation-in-punctured-plane", "transport": "identity"},
                       {"id": "T2", "kind": "biholomorphic-reparam", "transport": "identity"}]))

    fams["IMR-AN-003"] = quotient_family(
        "IMR-AN-003", "C-analytic-regularization", ["value"],
        {"value": "pi^2/6"}, {"value": "rearrangement_dependent"},       # zeta(2) abs. conv.; neg cond. conv.
        type_spec={"value": "label"},
        aliases={"value": ["sum", "total", "series value"]},
        anti_cert=anti_orbit_invariant(["value"]),
        meta=_records("synthetic_algorithmic", {"kind": "StructuralLabel", "encoding": ["value"]},
                      ["absolutely_convergent_sum"], ["conditionally_convergent_rearrangement"],
                      "Sum_{n>=1} 1/n^2 = zeta(2) = pi^2/6 (absolutely convergent: rearrangement-invariant).",
                      [{"id": "T1", "kind": "term-rearrangement", "transport": "identity"},
                       {"id": "T2", "kind": "regrouping", "transport": "identity"}]))

    fams["IMR-AN-004"] = quotient_family(
        "IMR-AN-004", "C-analytic-regularization", ["integral"],
        {"integral": "sqrt(pi)"}, {"integral": "2*sqrt(pi)"},            # Gaussian; neg drops Jacobian at a=2
        type_spec={"integral": "label"},
        aliases={"integral": ["value", "result"]},
        anti_cert=anti_orbit_invariant(["integral"]),
        meta=_records("synthetic_algorithmic", {"kind": "StructuralLabel", "encoding": ["integral"]},
                      ["substitution_with_jacobian"], ["substitution_without_jacobian"],
                      "Gaussian int_{-inf}^{inf} e^{-x^2} dx = sqrt(pi); transform x=u/a keeps it sqrt(pi) only WITH the 1/a Jacobian. Concrete scale a=2.",
                      [{"id": "T1", "kind": "substitution-x=u/a-with-Jacobian", "transport": "identity"},
                       {"id": "T2", "kind": "even-symmetry-x->-x", "transport": "identity"}]))

    # ---- Module D : anomaly / measure (L1 convention-ledger) ----
    fams["IMR-AM-003"] = compound_family(
        "IMR-AM-003", "D-anomaly-measure", ["index", "origin"],
        {"index": 1, "origin": "measure_jacobian"},
        {"index": 0, "origin": "classical_action"},                      # zero modes from anomaly; neg classical
        weights={"index": 0.5, "origin": 0.5}, required=["mechanism"], rung="L1",
        type_spec={"index": "int", "origin": "label"},
        aliases={"index": ["zero modes", "net zero modes", "analytic index"], "origin": ["source"]},
        anti_cert=anti_classical_law("origin", "index"),
        meta=_records("convention_ledger", {"kind": "Compound", "encoding": ["index", "origin"]},
                      ["index_from_anomaly"], ["classical_no_zero_modes"],
                      "Atiyah-Singer index ind D_R = 2 T(R) k; SU(2) fundamental T(R)=1/2, charge k=1 => net Dirac zero modes = 1.",
                      [{"id": "T1", "kind": "ledger-rewrite", "transport": "identity"},
                       {"id": "T2", "kind": "convention-change", "transport": "identity"}]))

    fams["IMR-AM-004"] = compound_family(
        "IMR-AM-004", "D-anomaly-measure", ["central_charge", "origin"],
        {"central_charge": 1, "origin": "conformal_anomaly"},
        {"central_charge": 0, "origin": "classically_traceless"},        # c=1 free scalar; neg classical
        weights={"central_charge": 0.5, "origin": 0.5}, required=["mechanism"], rung="L1",
        type_spec={"central_charge": "int", "origin": "label"},
        aliases={"central_charge": ["c", "central charge"], "origin": ["source"]},
        anti_cert=anti_orbit_invariant(["central_charge", "origin"]),
        meta=_records("convention_ledger", {"kind": "Compound", "encoding": ["central_charge", "origin"]},
                      ["central_charge_in_convention"], ["classical_trace_zero"],
                      "2D conformal anomaly: central charge of a free real scalar c = 1 (trace anomaly).",
                      [{"id": "T1", "kind": "ledger-rewrite", "transport": "identity"},
                       {"id": "T2", "kind": "scheme-convention-change", "transport": "identity"}]))

    # ---- Module E : boundary / variation ----
    fams["IMR-BV-002"] = compound_family(
        "IMR-BV-002", "E-boundary-variation", ["bulk", "boundary_vanishes_iff"],
        {"bulk": "-(box+m^2)phi", "boundary_vanishes_iff": "{Dirichlet,Neumann,closed,compact}"},
        {"bulk": "-(box+m^2)phi", "boundary_vanishes_iff": "{always}"},  # massive scalar; neg unconditional
        weights={"bulk": 0.5, "boundary_vanishes_iff": 0.5}, required=["witness"],
        type_spec={"bulk": "label", "boundary_vanishes_iff": "label"},
        aliases={"bulk": ["euler_lagrange", "bulk_term", "EL"], "boundary_vanishes_iff": ["boundary", "bdy"]},
        anti_cert=anti_unconditional("boundary_vanishes_iff"),
        meta=_records("synthetic_algorithmic", {"kind": "Compound", "encoding": ["bulk", "boundary_vanishes_iff"]},
                      ["bulk_plus_conditional_boundary"], ["unconditional_vanishing"],
                      "Vary S=int(1/2 (d phi)^2 - 1/2 m^2 phi^2) (mostly-minus eta): EL class (box+m^2)phi=0 up to nonzero scalar, plus the boundary-vanishing condition.",
                      [{"id": "T1", "kind": "integrate-by-parts-variant", "transport": "identity"},
                       {"id": "T2", "kind": "relabel-coordinates", "transport": "identity"}]))

    fams["IMR-BV-003"] = compound_family(
        "IMR-BV-003", "E-boundary-variation", ["bulk", "boundary_vanishes_iff"],
        {"bulk": "-(xddot+w^2x)", "boundary_vanishes_iff": "{fixed_endpoints,periodic}"},
        {"bulk": "-(xddot+w^2x)", "boundary_vanishes_iff": "{always}"}, # oscillator; neg unconditional
        weights={"bulk": 0.5, "boundary_vanishes_iff": 0.5}, required=["witness"],
        type_spec={"bulk": "label", "boundary_vanishes_iff": "label"},
        aliases={"bulk": ["euler_lagrange", "equation_of_motion", "EOM"], "boundary_vanishes_iff": ["boundary", "bdy"]},
        anti_cert=anti_unconditional("boundary_vanishes_iff"),
        meta=_records("synthetic_algorithmic", {"kind": "Compound", "encoding": ["bulk", "boundary_vanishes_iff"]},
                      ["eom_plus_conditional_boundary"], ["unconditional_vanishing"],
                      "Vary S=int(1/2 xdot^2 - 1/2 w^2 x^2) dt: EOM class xddot+w^2 x=0 up to nonzero scalar, plus the endpoint condition.",
                      [{"id": "T1", "kind": "integrate-by-parts-in-time", "transport": "identity"},
                       {"id": "T2", "kind": "time-relabel", "transport": "identity"}]))

    fams["IMR-BV-004"] = compound_family(
        "IMR-BV-004", "E-boundary-variation", ["bulk", "boundary_vanishes_iff"],
        {"bulk": "lap^2(phi)", "boundary_vanishes_iff": "{clamped,free,simply_supported,compact}"},
        {"bulk": "lap^2(phi)", "boundary_vanishes_iff": "{always}"},   # bending action; neg unconditional
        weights={"bulk": 0.5, "boundary_vanishes_iff": 0.5}, required=["witness"],
        type_spec={"bulk": "label", "boundary_vanishes_iff": "label"},
        aliases={"bulk": ["euler_lagrange", "biharmonic", "EL"], "boundary_vanishes_iff": ["boundary", "bdy"]},
        anti_cert=anti_unconditional("boundary_vanishes_iff"),
        meta=_records("synthetic_algorithmic", {"kind": "Compound", "encoding": ["bulk", "boundary_vanishes_iff"]},
                      ["bulk_plus_two_conditional_boundary_terms"], ["unconditional_vanishing"],
                      "Vary S=int 1/2 (phi'')^2: EL class biharmonic lap^2(phi)=0 up to nonzero scalar, plus TWO boundary terms vanishing under clamped/free/simply-supported/compact data.",
                      [{"id": "T1", "kind": "double-integrate-by-parts-variant", "transport": "identity"},
                       {"id": "T2", "kind": "relabel-coordinates", "transport": "identity"}]))

    # ---- Module F : transport / naturality ----
    fams["IMR-TN-003"] = quotient_family(
        "IMR-TN-003", "F-transport-naturality", ["det_kron"],
        {"det_kron": 1764}, {"det_kron": "kron_entry"},                  # det(A x B)=det(A)^2 det(B)^2; neg entry
        type_spec={"det_kron": "int"},
        aliases={"det_kron": ["det(A kron B)", "det kron", "tensor determinant", "det(A tensor B)"]},
        anti_cert=anti_nonnatural(),
        meta=_records("synthetic_algorithmic", {"kind": "Integer", "encoding": ["det_kron"]},
                      ["natural_invariant"], ["matrix_entry"],
                      "Determinant of A (x) B for 2x2 A,B: det(A)^2 det(B)^2 = 1764.",
                      [{"id": "T1", "kind": "simultaneous-basis-change", "transport": "identity"},
                       {"id": "T2", "kind": "factorwise-conjugation", "transport": "identity"}]))

    fams["IMR-TN-004"] = quotient_family(
        "IMR-TN-004", "F-transport-naturality", ["tr_comm"],
        {"tr_comm": 0}, {"tr_comm": "offdiag_entry"},                    # tr[A,B]=0; neg basis-dependent entry
        type_spec={"tr_comm": "int"},
        aliases={"tr_comm": ["tr[A,B]", "trace of commutator", "commutator trace", "tr(AB-BA)"]},
        anti_cert=anti_nonnatural(),
        meta=_records("synthetic_algorithmic", {"kind": "Integer", "encoding": ["tr_comm"]},
                      ["natural_invariant"], ["matrix_entry"],
                      "Trace of the commutator tr[A,B] = 0 (a natural, basis-independent invariant).",
                      [{"id": "T1", "kind": "conjugation", "transport": "identity"},
                       {"id": "T2", "kind": "factorwise-conjugation", "transport": "identity"}]))

    return fams
