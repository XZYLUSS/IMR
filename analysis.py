"""
imr/analysis.py -- the IMR confirmatory analysis pipeline.

This module implements the protocol specified in Section 9 of

    "What LLM Mathematics Benchmarks Measure: Construct Validity, Artifact
     Projection, and Certified Semantic Transport"

This module is the PRIMARY PAIRED-ENDPOINT AND INFERENCE-ENGINE PROTOTYPE.
It is not an implementation of Section 9 as a whole, and the scope table below
says exactly which components are released. Where paper and code could
disagree, the paper is the specification.

  Section 9 component                          status / location
  -------------------------------------------  --------------------------------
  paired observation, JP(T), delta_T, taxonomy  RELEASED  observation.py
  D^u, D^t, A_G, A_orb, mechanism/tie counts    RELEASED  observation.metric_suite
  partial identification + decision + gates     RELEASED  run_confirmatory_analysis
  frozen admission, module universe, audit      RELEASED  admission.json + this file
  obligation transport lambda_T, delta^P        PROTOTYPE proof_summary.py, not
                                                          wired into the primary path
  Dawid-Skene annotator aggregation             NOT RELEASED (analysis plan)
  live evaluator audit of renderer/parser/judge NOT RELEASED (analysis plan)
  H2 / H3                                       NOT RELEASED (descriptive / target)
  generation harness                            NOT RELEASED (no model is run)

Confirmatory vs exploratory. `run_confirmatory_analysis` loads the pinned
manifest internally and admits NO caller override of protocol identity.
`run_exploratory_analysis` is the permissive API: subsets, custom manifests and
precomputed scalar defects live there, its results carry `mode: exploratory`,
and it never returns the bare label `admissible`.

The remaining mapping is:

  paper                                          this module
  --------------------------------------------   ---------------------------
  composite per-arrow defect                     composite_defect
  family difference D_F = d_str - d_surf         family_difference
  stratum-weighted point estimate                point_estimate
  family-level variance                          variance
  stratified percentile-t lower bound            percentile_t_lcb
  null-imposed wild cluster bootstrap fallback   wild_cluster_pvalue
  evaluator gate (family-cluster upper bound)    family_cluster_upper, evaluator_gate
  yield gate                                     yield_gate
  Manski sharp bounds for unobserved cells       sharp_bounds
  threshold curve                                threshold_curve
  studentized-maximum multiplicity statistic     max_t_global (joint, FWER)
  differential evaluator sensitivity             differential_evaluator_sensitivity
  evaluator census and family-cluster gate       census_defect, evaluator_gate
  paired two-sided simulation                    simulate_observed_families
  per-cell endpoint calibration                  solve_beta_for_endpoint, run_grid
  decision rule with its gates                   decide
  the legacy scalar route entry point               _run_legacy_scalar_analysis

NO MODEL IS RUN ANYWHERE IN THIS FILE. Running `python analysis.py` executes
the pipeline on explicitly synthetic data, to demonstrate that the verdict is
a deterministic function of arm-level outcomes and to print a reduced
calibration grid. It reports no empirical result about any language model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- constants --
DELTA_SEM = 0.05           # smallest effect size of interest (margin)
EVAL_GATE = 0.05           # evaluator-defect gate, on a one-sided UPPER bound
RESPONSE_FLOOR = 0.90      # yield gate: administered-arm response rate
PARSE_FLOOR = 0.80         # yield gate: post-repair typed-parse rate
BOOT_B = 20_000            # bootstrap replicates
BOOT_SEED = 20240517       # master seed
MIN_CLUSTERS_PER_STRATUM = 8   # below this the wild-cluster fallback fires
MAX_DEGENERACY = 0.05      # above this share of zero-variance resamples, fallback

UNOBSERVED = None          # apparatus failure / grader unavailable -> bounds
N_MODULES_DECLARED = 6     # the declared module set; a gate cannot pass without all of them
BONFERRONI_MODULES = 6     # simultaneity correction for the max-over-modules gate
B_PRIM, B_WILD, B_CURVE, B_SENS = 20_000, 4_000, 4_000, 2_000
R_REPLICATES = 4           # generations per arm (paper Section 9.2)
APPARATUS_RETRIES = 3      # identical re-administrations after apparatus failure


def derive_seed(master: int, model: str, family: str, arm: str, replicate: int) -> int:
    """SHA-256 seed derivation, truncated to 64 bits. Python's built-in hash()
    is process-randomized and must not be used for anything reproducible."""
    import hashlib, json
    # Canonical JSON serialization: unescaped delimiter concatenation is not
    # injective (a model name containing '|' could collide with another tuple).
    key = json.dumps([int(master), str(model), str(family), str(arm),
                      int(replicate)], separators=(",", ":"),
                     ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def majority_defect(replicates: Sequence[float]) -> float:
    """Conservative majority over r replicate defects: ties resolve to a defect."""
    v = [x for x in replicates if x is not UNOBSERVED]
    if not v:
        return UNOBSERVED
    return 1.0 if (2 * sum(v) >= len(v)) else 0.0


ADMISSION_FILE = "admission.json"
ADMISSION_SCHEMA = "imr-admission/2"
ARMS_REQUIRED = {"structural": 2, "surface": 1}
ALLOWED_ARM_KINDS = frozenset(("structural", "surface"))


class ProtocolError(ValueError):
    """Raised when input does not instantiate the frozen protocol. The
    confirmatory path fails closed on every one of these."""


def validate_admission_document(doc: Dict) -> None:
    """Strict schema validation of the frozen manifest (round-6 M16)."""
    if doc.get("schema") != ADMISSION_SCHEMA:
        raise ProtocolError(f"admission schema must be {ADMISSION_SCHEMA}, "
                            f"got {doc.get('schema')!r}")
    if not isinstance(doc.get("version"), str) or not doc["version"]:
        raise ProtocolError("admission manifest needs a string version")
    pol = doc.get("policy")
    if not isinstance(pol, dict):
        raise ProtocolError("admission manifest needs a policy object")
    uni = pol.get("module_universe")
    if not (isinstance(uni, list) and len(set(uni)) == len(uni)):
        raise ProtocolError("policy.module_universe must be a duplicate-free list")
    if len(uni) != N_MODULES_DECLARED:
        raise ProtocolError(
            f"policy.module_universe must name exactly {N_MODULES_DECLARED} "
            f"modules (the frozen design), got {len(uni)}: {uni}")
    if "module_D_transition" not in pol or not isinstance(
            pol["module_D_transition"], str) or not pol["module_D_transition"]:
        raise ProtocolError("policy.module_D_transition must state the exact "
                            "eligibility transition")
    if pol.get("arms_per_family") != ARMS_REQUIRED:
        raise ProtocolError(f"policy.arms_per_family must be {ARMS_REQUIRED}")
    if pol.get("replicates_per_arm") != R_REPLICATES:
        raise ProtocolError(f"policy.replicates_per_arm must be {R_REPLICATES}")
    fams = doc.get("families")
    if not isinstance(fams, dict) or not fams:
        raise ProtocolError("admission manifest needs a non-empty families map")
    for fid, rec in fams.items():
        for field, typ in (("module", str), ("certificate_rung", str),
                           ("ni_control", bool), ("surface_complexity_matched", bool),
                           ("non_identity_transport", bool),
                           ("confirmatory_eligible", bool),
                           ("arms", dict), ("replicates_per_arm", int)):
            if field not in rec or not isinstance(rec[field], typ):
                raise ProtocolError(f"family {fid}: field {field} missing or mistyped")
        if rec["arms"] != ARMS_REQUIRED:
            raise ProtocolError(f"family {fid}: arms {rec['arms']} != {ARMS_REQUIRED}")
        if rec["replicates_per_arm"] != R_REPLICATES:
            raise ProtocolError(
                f"family {fid}: replicates_per_arm {rec['replicates_per_arm']} "
                f"!= {R_REPLICATES}")
        if not rec["confirmatory_eligible"] and not rec.get("reasons_ineligible"):
            raise ProtocolError(f"family {fid}: ineligible records must state "
                                "reasons_ineligible")
        if rec["module"] not in uni:
            raise ProtocolError(f"family {fid}: module {rec['module']} outside the "
                                "frozen module universe")


def load_admission(path: str = None) -> "Admission":
    """Load and VALIDATE the frozen admission manifest.

    `_run_legacy_scalar_analysis` requires it and fails closed when it is absent,
    malformed, or of the wrong schema version: confirmatory eligibility must be
    tied to an immutable released record, never to an object a caller builds at
    the call site."""
    import json, os, hashlib
    p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             ADMISSION_FILE)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"admission manifest not found at {p}; _run_legacy_scalar_analysis fails "
            "closed without it")
    raw = open(p, "rb").read()
    doc = json.loads(raw.decode("utf-8"))
    validate_admission_document(doc)
    return Admission(records={fid: dict(rec) for fid, rec in doc["families"].items()},
                     module_universe=tuple(doc["policy"]["module_universe"]),
                     version=str(doc["version"]),
                     schema=str(doc["schema"]),
                     digest=hashlib.sha256(raw).hexdigest(),
                     source=p)


@dataclass(frozen=True)
class Admission:
    """The frozen eligibility manifest, as a STRICT MAP from family ID to its
    recorded properties.

    Round-6 B01: the previous exclusion-list form was fail-OPEN -- a family
    whose ID was simply absent from the manifest passed, because it appeared on
    no exclusion list. Admission is now a positive test: an input family is
    admitted only if its ID occurs in the manifest, its module matches the
    recorded module, and its record carries `confirmatory_eligible: true`."""
    records: Dict[str, Dict] = None
    module_universe: Tuple[str, ...] = ()
    version: str = "ad-hoc"
    schema: str = ADMISSION_SCHEMA
    digest: str = ""
    source: str = "constructed at call site"
    permissive: bool = False      # smoke/simulation only; never confirmatory

    def __post_init__(self):
        object.__setattr__(self, "records", dict(self.records or {}))

    @property
    def excluded_modules(self) -> Tuple[str, ...]:
        eligible = {r["module"] for r in self.records.values()
                    if r.get("confirmatory_eligible")}
        return tuple(m for m in self.module_universe if m not in eligible)

    @property
    def eligible_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(fid for fid, r in self.records.items()
                            if r.get("confirmatory_eligible")))

    def admit(self, fams: Sequence["Family"]) -> List["Family"]:
        """Fail-closed admission. Raises ProtocolError on anything the frozen
        record does not vouch for."""
        if self.permissive:
            return list(fams)
        seen = set()
        out = []
        for f in fams:
            if f.fid in seen:
                raise ProtocolError(f"duplicate family id {f.fid}")
            seen.add(f.fid)
            rec = self.records.get(f.fid)
            if rec is None:
                raise ProtocolError(
                    f"family {f.fid} is absent from the frozen admission "
                    f"manifest ({self.source}); admission is a positive test")
            if rec["module"] != f.module:
                raise ProtocolError(
                    f"family {f.fid}: module {f.module!r} does not match the "
                    f"recorded module {rec['module']!r}")
            if not rec.get("confirmatory_eligible"):
                continue
            out.append(f)
        return out

    # retained name for readability at call sites; same fail-closed semantics
    def filter(self, fams: Sequence["Family"]) -> List["Family"]:
        return self.admit(fams)


def validate_confirmatory_design(fams: Sequence["Family"]) -> None:
    """Round-6 B03: the confirmatory entry point must refuse anything that is
    not the protocol's experimental unit -- exactly two structural arms, one
    surface arm, r complete replicate records per arm, allowed arm kinds,
    finite binary defects, unique ids."""
    if not fams:
        raise ProtocolError("no families supplied")
    seen = set()
    for f in fams:
        if f.fid in seen:
            raise ProtocolError(f"duplicate family id {f.fid}")
        seen.add(f.fid)
        counts: Dict[str, int] = {}
        for a in f.arms:
            if a.kind not in ALLOWED_ARM_KINDS:
                raise ProtocolError(f"{f.fid}: arm kind {a.kind!r} is not in "
                                    f"{sorted(ALLOWED_ARM_KINDS)}")
            counts[a.kind] = counts.get(a.kind, 0) + 1
            if not a.replicates:
                raise ProtocolError(
                    f"{f.fid}/{a.kind}: confirmatory input requires explicit "
                    f"replicate records; the legacy scalar arm is a smoke-test "
                    f"adapter only")
            if len(a.replicates) != R_REPLICATES:
                raise ProtocolError(
                    f"{f.fid}/{a.kind}: {len(a.replicates)} replicates, "
                    f"protocol requires {R_REPLICATES}")
            for r in a.replicates:
                if r.defect is UNOBSERVED:
                    continue
                try:
                    v = float(r.defect)
                except (TypeError, ValueError):
                    raise ProtocolError(f"{f.fid}/{a.kind}: non-numeric defect")
                if not math.isfinite(v) or v not in (0.0, 1.0):
                    raise ProtocolError(
                        f"{f.fid}/{a.kind}: defect {r.defect!r} is not binary/finite")
        if counts != ARMS_REQUIRED:
            raise ProtocolError(
                f"{f.fid}: arm composition {counts} does not match the frozen "
                f"design {ARMS_REQUIRED}")


# ------------------------------------------------------------ arm and family --
@dataclass(frozen=True)
class Replicate:
    """ONE complete response record: what a single generation produces.

    The protocol administers r = 4 of these per arm, and each carries its own
    provider finish reason, so apparatus failure, refusal, parse failure and
    judge output are all replicate-level events, not arm-level ones."""
    responded: bool = True        # non-empty attempt
    parsed: bool = True           # typed answer after the single repair pass
    apparatus_ok: bool = True     # provider-reported transport error if False
    defect: object = 0.0          # 0/1 once graded; ignored if not gradable

    def composite(self) -> object:
        """Replicate-level composite outcome: UNOBSERVED for apparatus failure,
        1.0 for refusal or parse failure (model behaviour), else the defect."""
        if not self.apparatus_ok:
            return UNOBSERVED
        if not self.responded or not self.parsed:
            return 1.0
        return float(self.defect)


@dataclass(frozen=True)
class Arm:
    """One administered arrow of a family.

    Either supply `replicates` (the protocol form: r complete response records)
    or the legacy scalar fields, which are treated as a single replicate."""
    kind: str
    defect: object = 0.0
    responded: bool = True
    parsed: bool = True
    replicates: Tuple[Replicate, ...] = ()

    def records(self) -> Tuple[Replicate, ...]:
        if self.replicates:
            return self.replicates
        return (Replicate(responded=self.responded, parsed=self.parsed,
                          apparatus_ok=self.defect is not UNOBSERVED,
                          defect=0.0 if self.defect is UNOBSERVED else self.defect),)


@dataclass(frozen=True)
class Family:
    fid: str
    module: str
    arms: Tuple[Arm, ...]

    def by_kind(self, kind: str) -> List[Arm]:
        return [a for a in self.arms if a.kind == kind]


def composite_defect(arm: Arm) -> object:
    """Arm outcome = conservative majority over the arm's replicate records.

    Refusal and parse failure are model behaviour and count as replicate
    defects; apparatus failure makes a replicate UNOBSERVED. Aggregation rule
    when only SOME replicates are unobserved, which the protocol must fix and
    now does: the majority is taken over the observed replicates, and the arm
    is UNOBSERVED only if every replicate is. Ties resolve to a defect."""
    vals = [r.composite() for r in arm.records()]
    obs = [v for v in vals if v is not UNOBSERVED]
    if not obs:
        return UNOBSERVED
    return 1.0 if (2 * sum(obs) >= len(obs)) else 0.0


def observed_yield(fams: Sequence[Family]) -> Dict[str, Tuple[float, float]]:
    """Yield rates DERIVED FROM THE IMMUTABLE REPLICATE RECORDS, per arm class.

    The gate must not accept externally supplied summary rates: a caller could
    otherwise pass perfect yield while the records contain refusals and parse
    failures. Rates are computed over replicates whose apparatus succeeded,
    since an apparatus failure is not a model non-response."""
    tally: Dict[str, List[int]] = {}
    for f in fams:
        for a in f.arms:
            t = tally.setdefault(a.kind, [0, 0, 0])   # n_apparatus_ok, responded, parsed
            for r in a.records():
                if not r.apparatus_ok:
                    continue
                t[0] += 1
                t[1] += int(r.responded)
                t[2] += int(r.responded and r.parsed)
    return {k: ((v[1] / v[0]) if v[0] else 0.0, (v[2] / v[0]) if v[0] else 0.0)
            for k, v in tally.items()}


def _mean_observed(vals: Sequence[object]) -> object:
    obs = [v for v in vals if v is not UNOBSERVED]
    return float(np.mean(obs)) if obs else UNOBSERVED


def family_difference(fam: Family, impute_str: object = None,
                      impute_surf: object = None) -> object:
    """D_F = d_str - d_surf.

    Imputation is DIRECTIONAL because the surface term enters with a minus
    sign: the worst case for D_F sets unobserved structural cells to 0 and
    unobserved surface cells to 1, and the best case does the reverse. Passing
    None for either leaves that arm class complete-case."""
    def val(a: Arm, imp: object) -> object:
        v = composite_defect(a)
        return imp if (v is UNOBSERVED and imp is not None) else v

    d_str = _mean_observed([val(a, impute_str) for a in fam.by_kind("structural")])
    d_surf = _mean_observed([val(a, impute_surf) for a in fam.by_kind("surface")])
    if d_str is UNOBSERVED or d_surf is UNOBSERVED:
        return UNOBSERVED
    return d_str - d_surf


def declared_modules(fams: Sequence[Family]) -> List[str]:
    """The module set as ADMINISTERED, fixed before outcomes are seen."""
    return sorted({f.module for f in fams})


def family_differences(fams: Sequence[Family], impute_str: object = None,
                       impute_surf: object = None,
                       modules: Sequence[str] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Family differences with module codes indexed against the DECLARED module
    list, never against the set that happens to yield observations. A module
    that yields nothing keeps its code and shows up as an empty stratum, which
    `check_strata` then turns into an inadmissible run rather than a silent
    recoding of the remaining modules."""
    decl = list(modules) if modules is not None else declared_modules(fams)
    diffs, mods = [], []
    for f in fams:
        d = family_difference(f, impute_str, impute_surf)
        if d is not UNOBSERVED:
            diffs.append(d)
            mods.append(f.module)
    codes = np.array([decl.index(m) for m in mods], dtype=int)
    return np.asarray(diffs, dtype=float), codes


def check_strata(codes: np.ndarray, n_declared: int) -> Dict[str, object]:
    """Every declared module must contribute at least one observed family."""
    present = {int(k) for k in codes.tolist()}
    empty = sorted(set(range(n_declared)) - present)
    return {"n_declared": n_declared, "empty_strata": empty,
            "admissible": not empty}


# ---------------------------------------------- point estimate and variance --
def _strata(codes: np.ndarray, n_declared: int = None) -> List[np.ndarray]:
    if codes.size == 0:
        return []
    k = n_declared if n_declared is not None else int(codes.max()) + 1
    return [np.flatnonzero(codes == j) for j in range(k)]


def fixed_weights(fams: Sequence[Family], modules: Sequence[str] = None
                  ) -> np.ndarray:
    """Pre-run weights from ADMITTED family counts. These are fixed before any
    outcome is seen and are never renormalized over the modules that happen to
    yield observations; renormalizing would make them post-outcome quantities
    and retarget the estimand toward modules with better apparatus yield."""
    decl = list(modules) if modules is not None else declared_modules(fams)
    counts = np.array([sum(1 for f in fams if f.module == m) for m in decl],
                      dtype=float)
    return counts / counts.sum()


def point_estimate(diffs: np.ndarray, codes: np.ndarray,
                   weights: np.ndarray = None) -> float:
    """Stratum-weighted mean. `weights` are the fixed pre-run weights; when
    omitted they default to observed stratum shares (used only inside the
    bootstrap, where the resample preserves stratum sizes by construction)."""
    if diffs.size == 0:
        return float("nan")
    k = len(weights) if weights is not None else None
    strata = _strata(codes, k)
    if weights is None:
        total = diffs.size
        w = np.array([idx.size / total for idx in strata], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    out, wsum = 0.0, 0.0
    for j, idx in enumerate(strata):
        if idx.size:
            out += w[j] * float(diffs[idx].mean())
            wsum += w[j]
    return float(out / wsum) if wsum > 0 else float("nan")


def variance(diffs: np.ndarray, codes: np.ndarray,
             weights: np.ndarray = None) -> float:
    """Family is the influence unit; stratified variance of the weighted mean."""
    if diffs.size == 0:
        return float("nan")
    k = len(weights) if weights is not None else None
    strata = _strata(codes, k)
    if weights is None:
        total = diffs.size
        w = np.array([idx.size / total for idx in strata], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    wsum = sum(w[j] for j, idx in enumerate(strata) if idx.size)
    if wsum <= 0:
        return float("nan")
    v = 0.0
    for j, idx in enumerate(strata):
        n = idx.size
        if n >= 2:
            v += (w[j] / wsum) ** 2 * float(diffs[idx].var(ddof=1)) / n
    return float(v)


def min_cluster_count(codes: np.ndarray) -> int:
    strata = [i.size for i in _strata(codes) if i.size]
    return min(strata) if strata else 0


# ------------------------------------------------------ percentile-t bounds --
def _resample(codes: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """(B, n) index matrix; families drawn with replacement WITHIN strata."""
    out = np.empty((B, codes.size), dtype=int)
    for idx in _strata(codes, int(codes.max()) + 1 if codes.size else 0):
        if idx.size:
            out[:, idx] = rng.choice(idx, size=(B, idx.size), replace=True)
    return out


def percentile_t_lcb(diffs: np.ndarray, codes: np.ndarray, *, B: int = BOOT_B,
                     alpha: float = 0.05, seed: int = BOOT_SEED,
                     weights: np.ndarray = None) -> Tuple[float, float]:
    """One-sided (1-alpha) studentized lower bound. Returns (lcb, degeneracy).

    A degenerate resample (zero variance) contributes t = 0 when the numerator
    also vanishes and +/- inf otherwise; the degeneracy share is returned so
    the caller can trigger the fallback."""
    theta = point_estimate(diffs, codes, weights)
    var = variance(diffs, codes, weights)
    if not np.isfinite(var) or var <= 0:
        return float("nan"), 1.0
    rng = np.random.default_rng(seed)
    idx = _resample(codes, B, rng)
    ts = np.empty(B, dtype=float)
    degenerate = 0
    for b in range(B):
        d_b, c_b = diffs[idx[b]], codes[idx[b]]
        var_b = variance(d_b, c_b, weights)
        num = point_estimate(d_b, c_b, weights) - theta
        if not np.isfinite(var_b) or var_b <= 0:
            degenerate += 1
            ts[b] = 0.0 if num == 0 else math.copysign(np.inf, num)
        else:
            ts[b] = num / math.sqrt(var_b)
    # Degenerate replicates are RETAINED in the quantile, with +inf ordering
    # above every finite value and -inf below: dropping them would silently
    # narrow the interval. np.quantile cannot consume infinities, so the
    # quantile is taken by order statistic on the sorted array.
    order = np.sort(ts)
    pos = min(int(math.ceil((1 - alpha) * B)) - 1, B - 1)
    q = float(order[max(pos, 0)])
    if not np.isfinite(q):
        return float("-inf") if q > 0 else float("nan"), degenerate / B
    return float(theta - q * math.sqrt(var)), degenerate / B


def wild_cluster_pvalue(diffs: np.ndarray, codes: np.ndarray, *,
                        margin: float = DELTA_SEM, B: int = BOOT_B,
                        seed: int = BOOT_SEED,
                        weights: np.ndarray = None) -> float:
    """Null-imposed wild cluster bootstrap, Rademacher weights, one-sided
    p-value for H0: Delta <= margin (Cameron-Gelbach-Miller 2008)."""
    theta = point_estimate(diffs, codes, weights)
    var = variance(diffs, codes, weights)
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    t_obs = (theta - margin) / math.sqrt(var)
    # Null imposed under a CONSTRAINED stratified fit: module means are
    # preserved and only the global weighted mean is shifted to the margin, so
    # heterogeneous module means are not flattened by the null.
    strata = _strata(codes, len(weights) if weights is not None else None)
    shift = margin - theta
    fitted = np.empty_like(diffs)
    for idx in strata:
        if idx.size:
            fitted[idx] = diffs[idx].mean() + shift
    resid = diffs - fitted
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(B, diffs.size))
    count = valid = 0
    for b in range(B):
        d_b = fitted + resid * signs[b]
        var_b = variance(d_b, codes, weights)
        if not np.isfinite(var_b) or var_b <= 0:
            continue                       # degenerate: excluded from BOTH
        valid += 1
        if (point_estimate(d_b, codes, weights) - margin) / math.sqrt(var_b) >= t_obs:
            count += 1
    if valid == 0:
        return float("nan")
    return (count + 1) / (valid + 1)


def sharp_bounds(fams: Sequence[Family], modules: Sequence[str] = None,
                 weights: np.ndarray = None) -> Tuple[float, float]:
    """Manski worst-case interval for D_F, imputed DIRECTIONALLY and ARMWISE
    before family averaging: the lower bound sets unobserved structural arms to
    0 and unobserved surface arms to 1; the upper bound reverses both. Because
    d_surf enters with a minus sign, a single constant applied to every missing
    arm does not bound the difference."""
    mods = list(modules) if modules is not None else declared_modules(fams)
    lo_d, lo_c = family_differences(fams, impute_str=0.0, impute_surf=1.0, modules=mods)
    hi_d, hi_c = family_differences(fams, impute_str=1.0, impute_surf=0.0, modules=mods)
    return (point_estimate(lo_d, lo_c, weights), point_estimate(hi_d, hi_c, weights))


def target_population_mismatch(fams: Sequence[Family], modules: Sequence[str] = None,
                               weights: np.ndarray = None) -> Dict[str, object]:
    """The complete-case point estimate and the sharp bounds target DIFFERENT
    family sets: the point estimate drops families with a wholly unobserved arm
    class, while the bounds cover every administered family. They therefore
    need not nest. A complete-case value outside the interval is a
    TARGET-POPULATION MISMATCH: the selected sub-population differs from the
    admitted one. It is consistent with informative missingness but does not
    identify the missingness mechanism, and it is reported as a mismatch, not
    as a diagnosis of why the data are missing."""
    mods = list(modules) if modules is not None else declared_modules(fams)
    d, c = family_differences(fams, modules=mods)
    pt = point_estimate(d, c, weights)
    lo, hi = sharp_bounds(fams, modules=mods, weights=weights)
    n_admin = len(fams)
    n_cc = int(d.size)
    return {"complete_case_estimate": pt, "sharp_bounds": (lo, hi),
            "n_administered": n_admin, "n_complete_case": n_cc,
            "n_dropped": n_admin - n_cc,
            "estimate_outside_bounds": bool(np.isfinite(pt) and (pt < lo or pt > hi)),
            "note": ("complete-case statistic lies outside the full-population "
                     "interval: target-population mismatch, consistent with but "
                     "not identifying informative missingness"
                     if np.isfinite(pt) and (pt < lo or pt > hi) else
                     "complete-case statistic lies within the full-population "
                     "interval")}


def threshold_curve(diffs: np.ndarray, codes: np.ndarray,
                    deltas: Sequence[float] = (0.0, 0.02, 0.05, 0.10),
                    **kw) -> Dict[float, float]:
    """delta -> LCB - delta, so the decision can be read off at any margin."""
    lcb, _ = percentile_t_lcb(diffs, codes, **kw)
    return {float(d): float(lcb - d) for d in deltas}


def common_support(per_model_fams: Dict[str, Sequence[Family]],
                   modules: Sequence[str] = None
                   ) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Fix the common administered-family panel across models, then form each
    model's lower-endpoint vector on it. Returns (vectors, codes, fids)."""
    names = list(per_model_fams)
    for nm in names:                       # one record per id, ids unique
        seen = [f.fid for f in per_model_fams[nm]]
        if len(seen) != len(set(seen)):
            raise ValueError(f"duplicate family ids for model {nm}")
    mod_of: Dict[str, str] = {}
    for nm in names:                       # identical module labels across models
        for f in per_model_fams[nm]:
            if mod_of.setdefault(f.fid, f.module) != f.module:
                raise ValueError(f"module label for {f.fid} differs across models")
    ids = [ {f.fid for f in per_model_fams[nm]} for nm in names ]
    keep = sorted(set.intersection(*ids)) if ids else []
    mods = list(modules) if modules is not None else sorted(
        {f.module for f in per_model_fams[names[0]] if f.fid in keep})
    out, codes = {}, None
    for nm in names:
        sel = [f for f in per_model_fams[nm] if f.fid in keep]
        sel.sort(key=lambda f: f.fid)
        d, c = family_differences(sel, impute_str=0.0, impute_surf=1.0, modules=mods)
        out[nm] = d
        codes = c if codes is None else codes
    return out, (codes if codes is not None else np.array([], dtype=int)), keep


def max_t_global(per_model: Dict[str, np.ndarray], codes: np.ndarray, *,
                 margin: float = DELTA_SEM, weights: np.ndarray = None,
                 B: int = 2000, alpha: float = 0.05, seed: int = BOOT_SEED
                 ) -> Dict[str, object]:
    """Global test across models by the studentized maximum, with its null
    distribution obtained by resampling FAMILIES ONCE per replicate and
    recomputing every model on that same resample, so the within-family
    correlation of model outcomes is preserved.

    `per_model` maps model name -> family-difference vector, all aligned to the
    same `codes` on a COMMON SUPPORT: the families administered to every model
    in the declared set. A per-model complete-case support would make T_max a
    maximum over quantities defined on different populations, which is not a
    test statistic; `common_support` builds the aligned panel. Returns the observed statistics, the FWER critical value, the
    global p-value, and the per-model rejection decisions at that critical
    value (which is what controls familywise error, not the marginal bounds)."""
    names = list(per_model)
    obs: Dict[str, float] = {}
    for nm in names:
        d = per_model[nm]
        var = variance(d, codes, weights)
        obs[nm] = ((point_estimate(d, codes, weights) - margin) / math.sqrt(var)
                   if np.isfinite(var) and var > 0 else float("nan"))
    finite = [v for v in obs.values() if np.isfinite(v)]
    t_max = max(finite) if finite else float("nan")

    rng = np.random.default_rng(seed)
    idx = _resample(codes, B, rng)
    null = np.empty(B, dtype=float)
    for b in range(B):
        sel = idx[b]
        c_b = codes[sel]
        row = []
        for nm in names:
            d_b = per_model[nm][sel]
            var_b = variance(d_b, c_b, weights)
            if np.isfinite(var_b) and var_b > 0:
                centre = point_estimate(per_model[nm], codes, weights)
                row.append((point_estimate(d_b, c_b, weights) - centre)
                           / math.sqrt(var_b))
        null[b] = max(row) if row else np.nan
    good = null[np.isfinite(null)]
    crit = float(np.quantile(good, 1 - alpha)) if good.size else float("nan")
    pval = (float((good >= t_max).sum()) + 1) / (good.size + 1) if good.size else float("nan")
    return {"observed": obs, "T_max": t_max, "critical_value": crit,
            "p_value": pval, "reject_global": bool(np.isfinite(crit) and t_max > crit),
            "reject_per_model": {nm: bool(np.isfinite(obs[nm]) and np.isfinite(crit)
                                          and obs[nm] > crit) for nm in names}}


def differential_evaluator_sensitivity(diffs: np.ndarray, codes: np.ndarray,
                                       delta_eval: float, *,
                                       weights: np.ndarray = None,
                                       **kw) -> Dict[str, float]:
    """Section 9.4: subtract the DIFFERENTIAL evaluator quantity (structural
    minus surface evaluator defect) from the endpoint and re-decide. The
    marginal gate does not bound this; it can be larger, smaller, or of the
    opposite sign."""
    base_lcb, _ = percentile_t_lcb(diffs, codes, weights=weights, **kw)
    adj_lcb, _ = percentile_t_lcb(diffs - delta_eval, codes, weights=weights, **kw)
    return {"delta_eval": float(delta_eval),
            "lcb": float(base_lcb), "lcb_adjusted": float(adj_lcb),
            "decision_changes": bool((base_lcb > DELTA_SEM) != (adj_lcb > DELTA_SEM))}


# ---------------------------------------------------------------- the gates --
def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided upper Clopper-Pearson limit, SciPy-free: bisect on p in
    P(X <= k | n, p) = alpha."""
    if n == 0 or k >= n:
        return 1.0

    def tail(p: float) -> float:
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        total, term = 0.0, (1.0 - p) ** n
        total += term
        for i in range(1, k + 1):
            term *= (n - i + 1) / i * p / (1.0 - p)
            total += term
        return total

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def census_defect(pair_defects: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """(a.i) Finite-bank census: the exact observed evaluator defect over every
    gold pair in the bank. No interval, because this is not a sample."""
    out = {m: float(np.mean(v)) for m, v in pair_defects.items() if len(v)}
    out["__all__"] = float(np.mean([x for v in pair_defects.values() for x in v])) \
        if any(len(v) for v in pair_defects.values()) else float("nan")
    return out


def family_cluster_upper(family_defects: Sequence[float], *, alpha: float = 0.05,
                         **_ignored) -> float:
    """One-sided upper bound on the evaluator presentation-consistency defect,
    with the FAMILY as the independent sampling unit.

    The bound is the EXACT one-sided Clopper-Pearson limit for the binomial
    count of failing families, at every count and not only at zero. A
    nonparametric percentile bootstrap on the same data is anticonservative
    here: at n = 94, k = 1, alpha = 0.05/6 it returns about 0.043 where the
    exact limit is about 0.071, so it would pass a gate the specified
    procedure fails. Inputs are coerced to the binary family-level failure
    indicator, which is the quantity the exact interval applies to."""
    z = binary_family_failures(family_defects)
    n = len(z)
    if n == 0:
        return 1.0
    k = int(sum(z))
    if k == 0:
        return float(1.0 - alpha ** (1.0 / n))   # exact all-zero CP limit
    return clopper_pearson_upper(k, n, alpha)


def evaluator_gate(family_defects_by_module: Dict[str, Sequence[float]],
                   gate: float = EVAL_GATE, *, declared_modules_list: Sequence[str] = None,
                   alpha: float = 0.05, bonferroni: int = None, **kw
                   ) -> Dict[str, object]:
    """Section 9.5(a.ii). The gate is a MAXIMUM over modules, so the per-module
    bounds must be SIMULTANEOUS: each is taken at level alpha/|M| (Bonferroni).
    The sampling unit is the FAMILY, and the audited quantity is the BINARY
    family-level presentation-consistency failure indicator, so that the exact
    all-zero binomial limit applies.

    A gate cannot pass unless every DECLARED module is audited: an earlier
    version checked only the modules it was handed, so supplying one module
    passed the gate for six."""
    decl = list(declared_modules_list) if declared_modules_list is not None \
        else sorted(family_defects_by_module)
    k = bonferroni if bonferroni is not None else max(1, len(decl))
    missing = [m for m in decl if m not in family_defects_by_module]
    ub = {m: family_cluster_upper(binary_family_failures(family_defects_by_module[m]),
                                  alpha=alpha / k, **kw)
          for m in decl if m in family_defects_by_module}
    worst = max(ub.values()) if ub else 1.0
    passes = bool(not missing and ub and worst < gate)
    return {"per_module_upper": ub, "max_upper": worst, "passes": passes,
            "sampling_unit": "family", "declared_modules": decl,
            "unaudited_modules": missing, "alpha_per_module": alpha / k,
            "note": ("gate fails: modules not audited: " + ", ".join(missing))
                    if missing else ""}


@dataclass(frozen=True)
class EvaluatorAuditRecord:
    """One audited certified arrow: keyed, provenanced, Boolean.

    Round-6 B06/B07 replace anonymous module vectors. The gate now consumes
    records that name the family and the arrow, carry the certificate digest
    and judge version, and report a BOOLEAN outcome -- so NaNs, negatives and
    duplicated zeros cannot be coerced into successes."""
    family_id: str
    arrow_id: str
    module: str
    discrepant: bool
    certificate_digest: str = ""
    judge_version: str = ""
    parser_version: str = ""
    renderer_version: str = ""


REQUIRED_ARROWS = ("T1", "T2", "s")


def validate_audit_records(records: Sequence[EvaluatorAuditRecord],
                           admission: "Admission",
                           modules: Sequence[str],
                           administered_ids: Sequence[str] = None,
                           strict: bool = True) -> Dict[str, List[bool]]:
    """Validate provenance, uniqueness, manifest coverage and outcome type,
    then DERIVE the binary family-failure indicators internally."""
    if not records:
        raise ProtocolError("no evaluator audit records supplied")
    seen = set()
    by_family: Dict[str, List[bool]] = {}
    fam_module: Dict[str, str] = {}
    for r in records:
        if not isinstance(r, EvaluatorAuditRecord):
            raise ProtocolError("audit records must be EvaluatorAuditRecord")
        if not isinstance(r.discrepant, bool):
            raise ProtocolError(
                f"{r.family_id}/{r.arrow_id}: outcome must be Boolean, got "
                f"{r.discrepant!r}")
        key = (r.family_id, r.arrow_id)
        if key in seen:
            raise ProtocolError(f"duplicate audit record for {key}")
        seen.add(key)
        rec = admission.records.get(r.family_id) if admission.records else None
        if not admission.permissive:
            if rec is None:
                raise ProtocolError(
                    f"audit record for {r.family_id} is absent from the frozen "
                    "manifest")
            if rec["module"] != r.module:
                raise ProtocolError(
                    f"{r.family_id}: audited module {r.module!r} does not match "
                    f"the recorded module {rec['module']!r}")
        if r.module not in modules:
            raise ProtocolError(
                f"{r.family_id}: audited module {r.module!r} outside the frozen "
                "module universe")
        if strict:
            if r.arrow_id not in REQUIRED_ARROWS:
                raise ProtocolError(
                    f"{r.family_id}: arrow {r.arrow_id!r} is not one of "
                    f"{REQUIRED_ARROWS}; the audit must cover the exact "
                    "certified arrows")
            if not r.certificate_digest:
                raise ProtocolError(f"{r.family_id}/{r.arrow_id}: empty "
                                    "certificate digest")
            if not r.judge_version:
                raise ProtocolError(f"{r.family_id}/{r.arrow_id}: empty judge "
                                    "version")
        by_family.setdefault(r.family_id, []).append(r.discrepant)
        fam_module[r.family_id] = r.module
    if strict:
        seen_arrows: Dict[str, set] = {}
        for r in records:
            seen_arrows.setdefault(r.family_id, set()).add(r.arrow_id)
        for fid, arrows in seen_arrows.items():
            if arrows != set(REQUIRED_ARROWS):
                raise ProtocolError(
                    f"{fid}: audited arrows {sorted(arrows)} != "
                    f"{list(REQUIRED_ARROWS)}; complete arrow coverage is required")
        if administered_ids is not None:
            audited, admin = set(seen_arrows), set(administered_ids)
            if audited != admin:
                raise ProtocolError(
                    f"audited family set differs from the administered set: "
                    f"missing {sorted(admin - audited)[:4]}, "
                    f"extra {sorted(audited - admin)[:4]}")
    out: Dict[str, List[bool]] = {m: [] for m in modules}
    for fid, flags in by_family.items():
        out[fam_module[fid]].append(any(flags))     # family fails if ANY arrow does
    missing = [m for m in modules if not out[m]]
    if missing:
        raise ProtocolError(f"modules with no evaluator audit coverage: {missing}")
    return out


def binary_family_failures(family_defects: Sequence[float]) -> List[float]:
    """The gated quantity is the BINARY indicator 'this family showed any
    evaluator discrepancy'. Values must be Boolean or finite in [0, 1]: an
    earlier version mapped anything not > 0 to success, so NaNs and negative
    numbers silently became ninety-four clean families and passed the gate."""
    out = []
    for x in family_defects:
        if isinstance(x, (bool, np.bool_)):
            out.append(1.0 if x else 0.0); continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            raise ProtocolError(f"evaluator audit value {x!r} is not numeric")
        if not math.isfinite(v) or v < 0.0 or v > 1.0:
            raise ProtocolError(
                f"evaluator audit value {x!r} outside the declared domain [0,1]")
        out.append(1.0 if v > 0 else 0.0)
    return out


def required_cluster_count(gate: float = EVAL_GATE, alpha: float = 0.05,
                           bonferroni: int = BONFERRONI_MODULES) -> int:
    """Smallest number of independent families PER MODULE that can pass the
    simultaneous gate at zero observed failures: least N with
    1 - (alpha/|M|)**(1/N) < gate."""
    a = alpha / max(1, bonferroni)
    n = 1
    while (1.0 - a ** (1.0 / n)) >= gate and n < 1_000_000:
        n += 1
    return n


def yield_gate(rates_by_arm_class: Dict[str, Tuple[float, float]]) -> Dict[str, object]:
    """Section 9.5(b): response and parse rates are checked SEPARATELY for each
    arm class. A pooled rate can pass while the structural class sits below the
    floor, which is exactly the failure the gate exists to catch.

    Input: arm class -> (response_rate, parse_rate)."""
    required = ("structural", "surface")
    missing = [c for c in required if c not in rates_by_arm_class]
    per_class = {c: {"response_rate": r, "parse_rate": p,
                     "passes": bool(r >= RESPONSE_FLOOR and p >= PARSE_FLOOR)}
                 for c, (r, p) in rates_by_arm_class.items()}
    passes = bool(not missing and per_class and all(v["passes"] for v in per_class.values()))
    return {"per_arm_class": per_class, "response_floor": RESPONSE_FLOOR,
            "parse_floor": PARSE_FLOOR, "missing_arm_classes": missing,
            "passes": passes, "timing": "post-collection admissibility, "
                                        "evaluated before any endpoint is computed"}


def decide(*, lcb: float, eval_gate_result: Dict[str, object],
           yield_gate_result: Dict[str, object],
           margin: float = DELTA_SEM) -> Dict[str, object]:
    """The decision rule with its gates. A failed gate makes the test
    INADMISSIBLE; that is never reported as a negative result."""
    admissible = bool(eval_gate_result["passes"] and yield_gate_result["passes"])
    rejected = bool(admissible and np.isfinite(lcb) and lcb > margin)
    return {"admissible": admissible, "reject_H0": rejected, "margin": margin,
            "lcb": lcb,
            "verdict": ("reject H0: structural-arm excess exceeds the margin" if rejected
                        else "fail to reject H0" if admissible
                        else "INADMISSIBLE: a pre-specified gate failed")}


# ------------------------------------------------------ simulation machinery --
def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def simulate_families(n_families: int, n_modules: int, *, beta: float,
                      mu: Sequence[float], sigma: Sequence[float],
                      rng: np.random.Generator,
                      p_apparatus: float = 0.0, apparatus_offset: float = 0.0,
                      p_refusal: float = 0.0, p_parsefail: float = 0.0,
                      p_judge: float = 0.0, r: int = R_REPLICATES) -> List[Family]:
    """The generating process of Section 9.6, at REPLICATE level.

    Each arm is administered r times and each of the r generations is a
    complete, independent response record with its own apparatus outcome,
    refusal, parse failure, judge error and defect -- which is what the
    protocol actually produces. An arm-level draw of the response process
    would model one event per arm instead of r, and would not exercise the
    aggregation rule for partially unobserved arms.

    Family latent brittleness u ~ N(0, sigma_m^2) is shared across the
    family's arms, inducing within-family correlation."""
    fams: List[Family] = []
    per_mod = max(1, n_families // n_modules)
    for i in range(n_families):
        m = min(i // per_mod, n_modules - 1)
        u = float(rng.normal(0.0, sigma[m]))
        arms: List[Arm] = []
        for kind, extra in (("structural", beta), ("structural", beta),
                            ("surface", 0.0)):
            pa = p_apparatus + (apparatus_offset if kind == "structural" else 0.0)
            pr = _sigmoid(mu[m] + u + extra)
            reps = []
            for _ in range(r):
                # Protocol: an apparatus failure is re-administered up to three
                # times under the identical protocol; only a cell still failing
                # after those retries is unobserved (Section 9.4, mechanism i).
                if pa > 0:
                    attempts = 0
                    while attempts < 1 + APPARATUS_RETRIES and rng.random() < pa:
                        attempts += 1
                    if attempts >= 1 + APPARATUS_RETRIES:
                        reps.append(Replicate(apparatus_ok=False)); continue
                if p_refusal and rng.random() < p_refusal:
                    reps.append(Replicate(responded=False)); continue
                if p_parsefail and rng.random() < p_parsefail:
                    reps.append(Replicate(parsed=False)); continue
                d = 1.0 if rng.random() < pr else 0.0
                if p_judge and rng.random() < p_judge:
                    d = 1.0 - d
                reps.append(Replicate(defect=d))
            arms.append(Arm(kind, replicates=tuple(reps)))
        fams.append(Family(f"SIM-{i:03d}", f"M{m}", tuple(arms)))
    return fams


def _majority_prob(p: float, r: int = R_REPLICATES) -> float:
    """P(conservative majority of r Bernoulli(p) draws = 1), ties -> 1, i.e.
    P(sum >= ceil(r/2)) with the tie included."""
    from math import comb
    need = (r + 1) // 2 if r % 2 else r // 2      # ties count as a defect
    return float(sum(comb(r, k) * p ** k * (1 - p) ** (r - k)
                     for k in range(need, r + 1)))


def marginal_excess(beta: float, mu: Sequence[float], sigma: Sequence[float],
                    n_draw: int = 20000, seed: int = 11,
                    r: int = R_REPLICATES) -> float:
    """The marginal Delta on the MAJORITY-REDUCED scale, which is the scale the
    endpoint is computed on. Vectorized: inverting this is on the hot path of
    every calibration cell, so it must not be a Python loop."""
    rng = np.random.default_rng(seed)
    need = (r + 1) // 2 if r % 2 else r // 2
    from math import comb
    ks = np.arange(need, r + 1)
    cf = np.array([comb(r, int(k)) for k in ks], dtype=float)

    def maj(p: np.ndarray) -> np.ndarray:
        p = p[:, None]
        return (cf * p ** ks * (1 - p) ** (r - ks)).sum(axis=1)

    vals = []
    for m in range(len(mu)):
        u = rng.normal(0.0, sigma[m], size=n_draw)
        s = maj(1.0 / (1.0 + np.exp(-(mu[m] + u + beta))))
        t = maj(1.0 / (1.0 + np.exp(-(mu[m] + u))))
        vals.append(float((s - t).mean()))
    return float(np.mean(vals))


def solve_beta(target: float, mu: Sequence[float], sigma: Sequence[float],
               lo: float = 0.0, hi: float = 6.0, tol: float = 1e-4) -> float:
    """Invert marginal_excess to hit a TARGET Delta on the probability scale."""
    if target <= 0:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if marginal_excess(mid, mu, sigma) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _run_legacy_scalar_analysis(fams: Sequence[Family], *,
                         admission: "Admission" = None,
                         require_manifest: bool = True,
                         audit_records: Sequence[EvaluatorAuditRecord] = None,
                         family_defects_by_module: Dict[str, Sequence[float]] = None,
                         delta_eval: float = 0.0,
                         margin: float = DELTA_SEM, seed: int = BOOT_SEED,
                         B_prim: int = B_PRIM, B_wild: int = B_WILD,
                         B_curve: int = B_CURVE, B_sens: int = B_SENS,
                         declared_modules_list: Sequence[str] = None
                         ) -> Dict[str, object]:
    """LEGACY SCALAR ROUTE -- PRIVATE. Not a confirmatory entry point.

    Round-9 Patch Set 6. This accepts precomputed scalar defects and a
    permissive admission, so it can never yield a confirmatory verdict. It is
    reachable only through `run_exploratory_analysis`, which stamps the result
    `mode: exploratory` and removes the bare `admissible` label. The
    confirmatory schema is produced by `run_confirmatory_analysis` alone.

    FAIL-CLOSED ORDER. Every step below can only reduce what follows it, and a
    failure at any step means no endpoint is computed at all:

      (-1) confirmatory design validation: 2 structural + 1 surface arms,
           r complete replicates per arm, binary finite defects, unique ids
      (0)  frozen admission manifest: positive ID lookup, module match,
           explicit confirmatory_eligible; unknown IDs are REJECTED
      (1)  module universe: the frozen six, exactly; the Bonferroni denominator
           comes from the manifest, never from the observed input
      (2)  declared strata non-empty on OBSERVED families
      (3)  apparatus gate on validated, family/arrow-keyed audit records
      (4)  yield admissibility derived from the immutable replicate records
      (5)  estimation, only if everything above passed
    """
    # ---- (-1)/(0) admission -------------------------------------------------
    if admission is None:
        admission = load_admission() if require_manifest else Admission(permissive=True)
    if not admission.permissive:
        validate_confirmatory_design(fams)
    fams = admission.admit(fams)

    # ---- (1) frozen module universe ----------------------------------------
    if declared_modules_list is not None:
        modules = list(declared_modules_list)
    elif admission.module_universe:
        modules = list(admission.module_universe)
    else:
        modules = declared_modules(fams)
    n_bonf = len(modules)

    def _inadmissible(reasons, extra=None):
        out = {"admission": {"version": admission.version, "schema": admission.schema,
                             "digest": admission.digest, "source": admission.source,
                             "module_universe": list(admission.module_universe),
                             "eligible_families": len(fams)},
               "modules": modules, "n_bonferroni": n_bonf,
               "n_families_admitted": len(fams),
               "estimation_performed": False,
               "admissible": False, "reject_H0": False,
               "verdict": "INADMISSIBLE: " + "; ".join(reasons)}
        if extra:
            out.update(extra)
        return out

    if not fams:
        return _inadmissible(["no confirmatory-eligible families under the frozen "
                              "manifest"])
    supplied = set(declared_modules(fams))
    if not admission.permissive and not supplied <= set(modules):
        return _inadmissible([f"modules {sorted(supplied - set(modules))} outside "
                              "the frozen universe"])

    # ---- (2) declared strata, judged on OBSERVED families ------------------
    weights = fixed_weights(fams, modules)
    _obs_d, _obs_c = family_differences(fams, modules=modules)
    strata = check_strata(_obs_c, len(modules))

    # ---- (3) apparatus gate on validated records ---------------------------
    try:
        if audit_records is not None:
            derived_audit = validate_audit_records(audit_records, admission, modules)
            eg = evaluator_gate(derived_audit, declared_modules_list=modules,
                                bonferroni=n_bonf)
            eg["provenance"] = "family/arrow-keyed audit records"
        elif family_defects_by_module is not None:
            if not admission.permissive:
                raise ProtocolError(
                    "confirmatory runs require family/arrow-keyed audit records; "
                    "anonymous per-module vectors are a smoke-test input only")
            eg = evaluator_gate(family_defects_by_module,
                                declared_modules_list=modules, bonferroni=n_bonf)
            eg["provenance"] = "anonymous module vectors (smoke test only)"
        else:
            eg = {"per_module_upper": {}, "max_upper": 1.0, "passes": False,
                  "sampling_unit": "family", "unaudited_modules": list(modules),
                  "provenance": "none",
                  "note": "no evaluator audit supplied: gate cannot pass"}
    except ProtocolError as exc:
        return _inadmissible([f"evaluator audit invalid: {exc}"])

    # ---- (4) yield, derived from the immutable replicate records -----------
    derived = observed_yield(fams)
    yg = yield_gate(derived)

    reasons = []
    if not strata["admissible"]:
        reasons.append(f"empty declared strata {strata['empty_strata']}")
    if not eg["passes"]:
        reasons.append("evaluator gate")
    if not yg["passes"]:
        reasons.append("yield admissibility")
    if reasons:
        return _inadmissible(reasons, {"strata": strata, "evaluator_gate": eg,
                                       "yield_gate": yg, "derived_yield": derived})

    # ---- (5) estimation, only now -----------------------------------------
    lo_d, lo_c = family_differences(fams, impute_str=0.0, impute_surf=1.0,
                                    modules=modules)
    hi_d, hi_c = family_differences(fams, impute_str=1.0, impute_surf=0.0,
                                    modules=modules)
    lo, hi = point_estimate(lo_d, lo_c, weights), point_estimate(hi_d, hi_c, weights)
    cc_d, cc_c = family_differences(fams, modules=modules)
    lcb_lo, degen = percentile_t_lcb(lo_d, lo_c, B=B_prim, seed=seed, weights=weights)
    min_clusters = min_cluster_count(lo_c)

    use_fallback = bool(min_clusters < MIN_CLUSTERS_PER_STRATUM
                        or degen > MAX_DEGENERACY or not np.isfinite(lcb_lo))
    if use_fallback:
        wild_p = wild_cluster_pvalue(lo_d, lo_c, margin=margin, B=B_wild,
                                     seed=seed, weights=weights)
        curve = {float(d): wild_cluster_pvalue(lo_d, lo_c, margin=d, B=B_curve,
                                               seed=seed, weights=weights)
                 for d in (0.0, 0.02, 0.05, 0.10)}
        curve_kind = "wild-cluster p-value at each margin"
        adj = wild_cluster_pvalue(lo_d - delta_eval, lo_c, margin=margin,
                                  B=B_sens, seed=seed, weights=weights)
        sens = {"delta_eval": float(delta_eval), "engine": "wild-cluster",
                "p_value": wild_p, "p_value_adjusted": adj,
                "decision_changes": bool((np.isfinite(wild_p) and wild_p < 0.05)
                                         != (np.isfinite(adj) and adj < 0.05))}
        reject, basis = bool(np.isfinite(wild_p) and wild_p < 0.05), \
            "wild-cluster fallback p-value on the lower endpoint"
    else:
        wild_p = float("nan")
        curve = threshold_curve(lo_d, lo_c, B=B_curve, seed=seed, weights=weights)
        curve_kind = "LCB - delta"
        sens = differential_evaluator_sensitivity(lo_d, lo_c, delta_eval,
                                                  weights=weights, B=B_sens, seed=seed)
        sens["engine"] = "percentile-t"
        reject, basis = bool(np.isfinite(lcb_lo) and lcb_lo > margin), \
            "percentile-t lower bound on the lower endpoint"

    cc = point_estimate(cc_d, cc_c, weights)
    return {
        "admission": {"version": admission.version, "schema": admission.schema,
                      "digest": admission.digest, "source": admission.source,
                      "module_universe": list(admission.module_universe),
                      "eligible_families": len(fams)},
        "modules": modules, "n_bonferroni": n_bonf, "weights": weights.tolist(),
        "n_families_admitted": len(fams),
        "n_families_complete_case": int(cc_d.size),
        "strata": strata, "estimation_performed": True,
        "partial_identification": {"lower": lo, "upper": hi},
        "lcb_lower_endpoint": lcb_lo, "degeneracy": degen,
        "min_cluster_count": min_clusters,
        "secondary_complete_case": cc,
        "target_population_mismatch": bool(np.isfinite(cc) and not (lo <= cc <= hi)),
        "threshold_curve": curve, "threshold_curve_kind": curve_kind,
        "used_fallback": use_fallback, "wild_p_value": wild_p,
        "decision_basis": basis,
        "evaluator_gate": eg, "yield_gate": yg, "derived_yield": derived,
        "differential_evaluator": sens,
        "bootstrap_counts": {"primary": B_prim, "wild": B_wild,
                             "curve": B_curve, "sensitivity": B_sens},
        "admissible": True, "reject_H0": reject,
        "verdict": ("reject H0: the lower partial-identification endpoint exceeds "
                    "the margin" if reject else "fail to reject H0"),
    }


def _cell_lower_endpoint(beta: float, *, n: int, n_modules: int, mu, sigma,
                         p_apparatus, apparatus_offset, p_refusal, p_parsefail,
                         p_judge, reps: int, rng: np.random.Generator) -> float:
    """Monte-Carlo mean of the ACTUAL directional lower endpoint produced by the
    complete replicate-level DGP, after refusal, parse failure, apparatus loss,
    retries, judge error, majority aggregation and directional bounding."""
    vals = []
    for _ in range(reps):
        fams = simulate_families(n, n_modules, beta=beta, mu=mu, sigma=sigma,
                                 rng=rng, p_apparatus=p_apparatus,
                                 apparatus_offset=apparatus_offset,
                                 p_refusal=p_refusal, p_parsefail=p_parsefail,
                                 p_judge=p_judge)
        mods = declared_modules(fams)
        w = fixed_weights(fams, mods)
        lo, _hi = sharp_bounds(fams, modules=mods, weights=w)
        if np.isfinite(lo):
            vals.append(lo)
    return float(np.mean(vals)) if vals else float("nan")


def solve_beta_for_endpoint(target: float, *, n: int, n_modules: int, mu, sigma,
                            p_apparatus, apparatus_offset, p_refusal, p_parsefail,
                            p_judge, reps: int = 120, seed: int = 5,
                            lo: float = 0.0, hi: float = 6.0,
                            tol: float = 0.004) -> Tuple[float, float]:
    """Round-6 B08. Calibrate beta against the quantity each cell ACTUALLY
    analyses -- the directional lower endpoint under the full response process
    -- not against a latent no-missingness majority excess.

    Returns (beta, realized_lower_endpoint). The realized value is stored in the
    grid output so no cell can be mislabelled by its nominal target."""
    # Round-7 blocker 6: a zero target is SOLVED, not assumed to be beta = 0.
    # Under composite failures and directional bounding, E[lower endpoint] at
    # beta = 0 is not generally zero, so beta may be negative.
    if target <= 0:
        blo, bhi = -3.0, 3.0
        for _ in range(22):
            mid = 0.5 * (blo + bhi)
            got = _cell_lower_endpoint(
                mid, n=n, n_modules=n_modules, mu=mu, sigma=sigma,
                p_apparatus=p_apparatus, apparatus_offset=apparatus_offset,
                p_refusal=p_refusal, p_parsefail=p_parsefail, p_judge=p_judge,
                reps=reps, rng=np.random.default_rng(seed))
            if not np.isfinite(got) or abs(got - target) < tol:
                break
            if got < target:
                blo = mid
            else:
                bhi = mid
        b = 0.5 * (blo + bhi)
        return b, _cell_lower_endpoint(
            b, n=n, n_modules=n_modules, mu=mu, sigma=sigma,
            p_apparatus=p_apparatus, apparatus_offset=apparatus_offset,
            p_refusal=p_refusal, p_parsefail=p_parsefail, p_judge=p_judge,
            reps=reps, rng=np.random.default_rng(seed))
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        got = _cell_lower_endpoint(
            mid, n=n, n_modules=n_modules, mu=mu, sigma=sigma,
            p_apparatus=p_apparatus, apparatus_offset=apparatus_offset,
            p_refusal=p_refusal, p_parsefail=p_parsefail, p_judge=p_judge,
            reps=reps, rng=np.random.default_rng(seed))
        if not np.isfinite(got):
            break
        if abs(got - target) < tol:
            return mid, got
        if got < target:
            lo = mid
        else:
            hi = mid
    b = 0.5 * (lo + hi)
    return b, _cell_lower_endpoint(
        b, n=n, n_modules=n_modules, mu=mu, sigma=sigma,
        p_apparatus=p_apparatus, apparatus_offset=apparatus_offset,
        p_refusal=p_refusal, p_parsefail=p_parsefail, p_judge=p_judge,
        reps=reps, rng=np.random.default_rng(seed))


def simulate_observed_families(n_families: int, n_modules: int, *, beta: float,
                               mu: Sequence[float], sigma: Sequence[float],
                               rng: np.random.Generator,
                               p_apparatus: float = 0.0,
                               apparatus_offset: float = 0.0,
                               p_refusal: float = 0.0, p_parsefail: float = 0.0,
                               p_answertype: float = 0.0, p_judge: float = 0.0,
                               r: int = R_REPLICATES,
                               modules: Sequence[str] = None,
                               family_ids: Sequence[str] = None,
                               transport_factor: int = 1
                               ) -> List["ObservedFamily"]:
    """Simulate BOTH SIDES of every administered arrow independently, through
    the same record type, transport and router the real protocol uses.

    `transport_factor` k gives a genuine non-identity transport tau(v) = k*v, so
    the simulator exercises the total map rather than an identity special case."""
    from observation import (ArrowResponse, NegativeControl, ObservedFamily,
                             SideResponse, Status, Transport, REQUIRED_ARROWS)
    mods = list(modules) if modules is not None else \
        [f"M{k}" for k in range(n_modules)]
    fams: List[ObservedFamily] = []
    per_mod = max(1, n_families // len(mods))
    GOLD, OTHER = 1, 2

    def _mk_transport(fid, aid):
        """Resolved from the PACKAGE-PINNED registry, so simulated records use
        the same globally certified transports a confirmatory run requires
        (round-11 section 1.2). The per-arrow certificate digest is carried on
        the returned object."""
        from certificate_registry import (PRODUCTION_REGISTRY,
                                          TRANSPORT_DIGESTS, BuiltinTransportId)
        tid = (BuiltinTransportId.INT_IDENTITY_V1.value if transport_factor == 1
               else BuiltinTransportId.INT_NEGATION_V1.value)
        return PRODUCTION_REGISTRY.resolve_transport(
            transport_id=tid, version="1",
            expected_transport_certificate_digest=TRANSPORT_DIGESTS[tid],
            arrow_certificate_digest=f"cert-{fid}-{aid}")

    def _observed(pid, status, normalized=None, reason="stop"):
        import hashlib
        dig = hashlib.sha256(f"{pid}|{status}|{normalized}|{reason}".encode()).hexdigest()
        return SideResponse(pid, status, normalized=normalized,
                            provider_finish_reason=reason, raw_digest=dig,
                            record_kind="observed")

    def _side(pid, p_wrong, pa, tau):
        if pa > 0:
            att = 0
            while att < 1 + APPARATUS_RETRIES and rng.random() < pa:
                att += 1
            if att >= 1 + APPARATUS_RETRIES:
                return _observed(pid, Status.APPARATUS, reason="transport_error")
        if p_refusal and rng.random() < p_refusal:
            return _observed(pid, Status.REFUSAL, reason="stop")
        if p_parsefail and rng.random() < p_parsefail:
            return _observed(pid, Status.PARSE_FAIL, reason="length")
        if p_answertype and rng.random() < p_answertype:
            return _observed(pid, Status.ANSWER_TYPE, reason="stop")
        # Round-9 Patch Set 7: two explicit layers. The MODEL produces an
        # answer; the JUDGE then reports it, and p_judge flips the JUDGE's
        # reading, not the model's answer. The row records which layer each
        # parameter acts on so the DGP in the paper and the code agree.
        model_wrong = rng.random() < p_wrong
        judged_wrong = (not model_wrong) if (p_judge and rng.random() < p_judge) \
            else model_wrong
        val = (OTHER if judged_wrong else GOLD)
        return _observed(pid, Status.OK, normalized=tau(val), reason="stop")

    for i in range(n_families):
        m = min(i // per_mod, len(mods) - 1)
        mod = mods[m]
        fid = family_ids[i] if family_ids is not None else f"SIM-{i:03d}"
        u = float(rng.normal(0.0, sigma[m]))
        arrows = []
        for aid in REQUIRED_ARROWS:
            kind = "surface" if aid == "s" else "structural"
            extra = 0.0 if kind == "surface" else beta
            tr = _mk_transport(fid, aid)
            p_wrong_t = _sigmoid(mu[m] + u + extra)
            p_wrong_c = _sigmoid(mu[m] + u)
            pa = p_apparatus + (apparatus_offset if kind == "structural" else 0.0)
            for rep in range(r):
                arrows.append(ArrowResponse(
                    family_id=fid, module=mod, arrow_id=aid, replicate=rep,
                    certificate_digest=f"cert-{fid}-{aid}",
                    canonical=_side(f"{fid}:x", p_wrong_c, pa, lambda v: v),
                    transformed=_side(f"{fid}:{aid}", p_wrong_t, pa, tr.tau),
                    canonical_gold=GOLD, transport=tr,
                    judge_version="sim-judge-1", parser_version="sim-parser-1",
                    renderer_version="sim-renderer-1"))
        ctrl = NegativeControl(
            family_id=fid, presentation_id=f"{fid}:n", canonical_gold=GOLD,
            neighboring_gold=OTHER, anti_certificate_id=f"anti-{fid}",
            anti_certificate_digest=f"antidig-{fid}",
            normalizer_status=Status.OK, normalizer_output=OTHER,
            parser_version="sim-parser-1", renderer_version="sim-renderer-1",
            provenance_digest=f"prov-{fid}")
        fams.append(ObservedFamily(fid, mod, tuple(arrows), negative_control=ctrl))
    return fams


def simulation_registry(fams):
    """TEST-ONLY registry for simulated families (round-11 section 2.6).

    Reachable only from the private confirmatory test helper. Results obtained
    through it carry mode='test-helper' and admissible_test, never the
    confirmatory verdict schema."""
    from certificate_registry import (AntiCertificateEntry, CertificateRegistry)
    reg = CertificateRegistry()
    for f in fams:
        nc = f.negative_control
        reg.register_anti(AntiCertificateEntry(
            anti_certificate_id=nc.anti_certificate_id,
            anti_certificate_digest=nc.anti_certificate_digest,
            permitted_families=(f.family_id,), control_type="N-i",
            verifier=lambda c: (c.neighboring_gold != c.canonical_gold
                                and c.normalizer_output == c.neighboring_gold),
            verifier_version="test-1"))
    return reg.seal()


PINNED_MANIFEST = "admission.json"
# Round-11 section 4. Regenerated by tools/pin_admission.py at freeze time.
PINNED_ADMISSION_SHA256 = "72b365a4d43306f5da2a65d5dd70325bab9ec7ca5ce1c04551c1df3604613730"


def _verify_manifest_entry(anc_dir: str, filename: str, digest: str) -> None:
    """Locate `filename` in anc/MANIFEST.sha256, require exactly one entry, and
    require it to equal `digest` (round-11 section 4.2)."""
    import os
    path = os.path.join(anc_dir, "MANIFEST.sha256")
    if not os.path.exists(path):
        raise ProtocolError("anc/MANIFEST.sha256 is missing: the release "
                            "integrity record is not present")
    hits = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d, rel = line.split("  ", 1)
        except ValueError:
            raise ProtocolError(f"malformed MANIFEST entry: {line!r}")
        if rel == filename:
            hits.append(d)
    if not hits:
        raise ProtocolError(f"{filename} has no entry in MANIFEST.sha256")
    if len(hits) > 1:
        raise ProtocolError(f"{filename} has {len(hits)} entries in "
                            "MANIFEST.sha256; exactly one is required")
    if hits[0] != digest:
        raise ProtocolError(
            f"{filename} does not match its MANIFEST entry: on disk "
            f"{digest[:16]}..., manifest {hits[0][:16]}...")


def _pinned_manifest_path() -> str:
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        PINNED_MANIFEST)


def _load_pinned(expected_digest: str = None) -> "Admission":
    """Authenticate admission.json automatically (round-11 section 4).

    The embedded PINNED_ADMISSION_SHA256 is checked ALWAYS; the caller's
    optional `expected_digest` can only add a stricter assertion and can never
    replace it. The file is additionally verified against its MANIFEST entry,
    so a locally edited manifest is caught at confirmatory startup rather than
    only by a separate integrity run."""
    import os
    adm = load_admission(_pinned_manifest_path())
    anc = os.path.dirname(os.path.abspath(__file__))
    if PINNED_ADMISSION_SHA256.startswith("@@"):
        raise ProtocolError("PINNED_ADMISSION_SHA256 was never generated; the "
                            "release was not frozen")
    if adm.digest != PINNED_ADMISSION_SHA256:
        raise ProtocolError(
            f"admission.json does not match the digest embedded in this "
            f"release: on disk {adm.digest[:16]}..., embedded "
            f"{PINNED_ADMISSION_SHA256[:16]}...")
    _verify_manifest_entry(anc, PINNED_MANIFEST, adm.digest)
    if expected_digest and adm.digest != expected_digest:
        raise ProtocolError(
            f"caller-supplied expected digest {expected_digest[:16]}... does "
            f"not match the released manifest")
    return adm


def _validate_audit_join(observed, audit_records, adm, modules):
    """Round-8 B03/M14: join observation and audit records on
    (family, module, arrow) and require apparatus identity, not just outer keys."""
    from observation import REQUIRED_ARROWS as ARROWS
    admin = {}
    for o in observed:
        for a in o.arrows:
            admin[(a.family_id, a.arrow_id)] = a
    seen = set()
    for r in audit_records:
        key = (r.family_id, r.arrow_id)
        if key in seen:
            raise ProtocolError(f"duplicate audit record for {key}")
        seen.add(key)
        adminrec = admin.get(key)
        if adminrec is None:
            raise ProtocolError(f"audit record {key} was never administered")
        if r.module != adminrec.module:
            raise ProtocolError(f"{key}: audit module {r.module} != administered "
                                f"{adminrec.module}")
        if r.certificate_digest != adminrec.certificate_digest:
            raise ProtocolError(
                f"{key}: audit certificate {r.certificate_digest!r} != "
                f"administered certificate {adminrec.certificate_digest!r}")
        if r.judge_version != adminrec.judge_version:
            raise ProtocolError(
                f"{key}: audit judge {r.judge_version!r} != administered "
                f"{adminrec.judge_version!r}")
        if getattr(r, "parser_version", "") != adminrec.parser_version:
            raise ProtocolError(f"{key}: audit parser version mismatch")
        if getattr(r, "renderer_version", "") != adminrec.renderer_version:
            raise ProtocolError(f"{key}: audit renderer version mismatch")
    missing = set(admin) - seen
    if missing:
        raise ProtocolError(f"audit does not cover administered arrows: "
                            f"{sorted(missing)[:4]}")
    out = {m: [] for m in modules}
    fam_flags: Dict[str, List[bool]] = {}
    fam_module: Dict[str, str] = {}
    for r in audit_records:
        if not isinstance(r.discrepant, bool):
            raise ProtocolError(f"{r.family_id}/{r.arrow_id}: non-Boolean outcome")
        fam_flags.setdefault(r.family_id, []).append(r.discrepant)
        fam_module[r.family_id] = r.module
    for fid, flags in fam_flags.items():
        out[fam_module[fid]].append(any(flags))
    empty = [m for m in modules if not out[m]]
    if empty:
        raise ProtocolError(f"modules with no evaluator audit coverage: {empty}")
    return out


def run_confirmatory_analysis(observed: Sequence["ObservedFamily"], *,
                              audit_records: Sequence[EvaluatorAuditRecord],
                              expected_manifest_digest: str = None,
                              delta_eval: float = 0.0,
                              margin: float = DELTA_SEM, seed: int = BOOT_SEED
                              ) -> Dict[str, object]:
    """THE confirmatory entry point.

    Round-8 B01: there is NO `manifest_path` parameter. The manifest is the one
    pinned inside the package, and an optional `expected_manifest_digest`
    lets a caller pin the exact document it believes it is running against --
    it can only make the check stricter, never substitute a document. Synthetic
    manifests are reachable only through `_run_confirmatory_with_manifest`,
    which is a private test helper and is named as one.

    Enforced before anything is computed:
        M_input == M_manifest, F_administered == F_eligible,
        F_audited == F_administered, plus per-record join and N-i control."""
    from certificate_registry import PRODUCTION_REGISTRY
    return _confirmatory_core(observed, audit_records,
                              _load_pinned(expected_manifest_digest),
                              delta_eval, margin, seed,
                              PRODUCTION_REGISTRY)


def _run_confirmatory_with_manifest(observed, audit_records, manifest_path, **kw):
    """PRIVATE test helper. Never call this from a confirmatory run: its result
    is produced against a caller-supplied manifest and is not a confirmatory
    verdict about the released protocol."""
    res = dict(_confirmatory_core(observed, audit_records,
                                  load_admission(manifest_path),
                                  kw.get("delta_eval", 0.0),
                                  kw.get("margin", DELTA_SEM),
                                  kw.get("seed", BOOT_SEED),
                                  kw.get("registry")))
    # Round-9 Patch Set 6: a caller-supplied manifest must NEVER yield the
    # confirmatory schema. The result is re-labelled, not merely annotated.
    res["mode"] = "test-helper"
    res["manifest_provenance"] = "caller-supplied (test helper)"
    if "admissible" in res:
        res["admissible_test"] = res.pop("admissible")
    return res


def _confirmatory_core(observed, audit_records, adm, delta_eval, margin, seed,
                       registry=None):
    from observation import metric_suite, yield_rates, REQUIRED_ARROWS as ARROWS

    def _bad(reasons, extra=None):
        out = {"mode": "confirmatory",
               "manifest_provenance": "package-pinned",
               "admission": {"version": adm.version, "schema": adm.schema,
                             "digest": adm.digest, "source": adm.source,
                             "module_universe": list(adm.module_universe)},
               "modules": list(adm.module_universe),
               "n_bonferroni": len(adm.module_universe),
               "estimation_performed": False, "admissible": False,
               "reject_H0": False,
               "verdict": "INADMISSIBLE: " + "; ".join(reasons)}
        if extra:
            out.update(extra)
        return out

    modules = list(adm.module_universe)
    eligible = set(adm.eligible_ids)
    administered = [o.family_id for o in observed]
    if len(set(administered)) != len(administered):
        return _bad(["duplicate administered family ids"])
    if set(administered) != eligible:
        return _bad([f"administered set != eligible set "
                     f"(missing {sorted(eligible - set(administered))[:4]}, "
                     f"extra {sorted(set(administered) - eligible)[:4]})"])
    if not eligible:
        return _bad(["no confirmatory-eligible families in the frozen manifest"])
    if sorted({o.module for o in observed}) != sorted(modules):
        return _bad(["module set != frozen universe"])

    # ---- N-i negative control is REQUIRED and must have succeeded (B04) ----
    missing_ctrl = [o.family_id for o in observed if o.negative_control is None]
    if missing_ctrl:
        return _bad([f"families without an N-i control record: "
                     f"{missing_ctrl[:4]}"])
    # Round-10 blocker 2: the gate RESOLVES and REPLAYS every anti-certificate
    # through the frozen registry, and binds the control to its family and
    # apparatus. Replay is never an accepted Boolean.
    if registry is None:
        return _bad(["no anti-certificate registry supplied: N-i replay cannot "
                     "be performed, and replay is never assumed"])
    failed_ctrl = []
    for o in observed:
        pv = {a.parser_version for a in o.arrows}
        rv = {a.renderer_version for a in o.arrows}
        why = o.negative_control.reasons(
            registry, family_id=o.family_id,
            parser_version=next(iter(pv)) if len(pv) == 1 else None,
            renderer_version=next(iter(rv)) if len(rv) == 1 else None)
        if why:
            failed_ctrl.append((o.family_id, why[0]))
    if failed_ctrl:
        fid, why = failed_ctrl[0]
        return _bad([f"N-i negative control not established for {len(failed_ctrl)} "
                     f"famil{'y' if len(failed_ctrl) == 1 else 'ies'} "
                     f"(e.g. {fid}: {why})"])

    from certificate_registry import PRODUCTION_REGISTRY as _PR
    for o in observed:
        for a in o.arrows:
            if a.transport.domain_kind == "typed" and \
                    not _PR.is_certified_transport(a.transport):
                return _bad([f"{o.family_id}/{a.arrow_id}: confirmatory typed "
                             "transport was not issued by the package-pinned "
                             "production registry"])
            for side in (a.canonical, a.transformed):
                if side.record_kind != "observed":
                    return _bad([f"{o.family_id}/{a.arrow_id}: response "
                                 f"{side.presentation_id} is marked "
                                 f"{side.record_kind!r}; synthetic records "
                                 "cannot enter a confirmatory run"])
    for o in observed:
        rec = adm.records.get(o.family_id)
        if rec is None or rec["module"] != o.module:
            return _bad([f"{o.family_id}: module does not match the manifest"])
        if set(o.arrow_ids()) != set(ARROWS):
            return _bad([f"{o.family_id}: arrows {o.arrow_ids()} != {ARROWS}"])

    try:
        derived_audit = _validate_audit_join(observed, audit_records, adm, modules)
    except ProtocolError as exc:
        return _bad([f"evaluator audit invalid: {exc}"])

    eg = evaluator_gate(derived_audit, declared_modules_list=modules,
                        bonferroni=len(modules))
    from observation import yield_detail
    yg = yield_gate(yield_rates(observed))
    yg["detail"] = yield_detail(observed)

    diffs, mods_of = [], []
    for o in observed:
        d = o.difference()
        if d is not None:
            diffs.append(d); mods_of.append(o.module)
    obs_codes = np.array([modules.index(m) for m in mods_of], dtype=int)
    strata = check_strata(obs_codes, len(modules))

    reasons = []
    if not strata["admissible"]:
        reasons.append(f"empty declared strata {strata['empty_strata']}")
    if not eg["passes"]:
        reasons.append("evaluator gate")
    if not yg["passes"]:
        reasons.append("yield admissibility")
    if reasons:
        return _bad(reasons, {"strata": strata, "evaluator_gate": eg,
                              "yield_gate": yg, "metrics": metric_suite(observed)})

    counts = np.array([sum(1 for o in observed if o.module == m) for m in modules],
                      dtype=float)
    weights = counts / counts.sum()
    lo_v, hi_v, codes = [], [], []
    for o in observed:
        lo_f, hi_f = o.difference_bounds()   # round-9 Patch Set 5
        lo_v.append(lo_f); hi_v.append(hi_f)
        codes.append(modules.index(o.module))
    lo_d = np.asarray(lo_v); hi_d = np.asarray(hi_v)
    lc = np.asarray(codes, dtype=int)
    lo = point_estimate(lo_d, lc, weights); hi = point_estimate(hi_d, lc, weights)
    lcb_lo, degen = percentile_t_lcb(lo_d, lc, B=B_PRIM, seed=seed, weights=weights)
    use_fallback = bool(min_cluster_count(lc) < MIN_CLUSTERS_PER_STRATUM
                        or degen > MAX_DEGENERACY or not np.isfinite(lcb_lo))

    # ---- threshold curve and differential sensitivity, deciding engine (B06)
    if use_fallback:
        wild_p = wild_cluster_pvalue(lo_d, lc, margin=margin, B=B_WILD,
                                     seed=seed, weights=weights)
        curve = {float(d): wild_cluster_pvalue(lo_d, lc, margin=d, B=B_CURVE,
                                               seed=seed, weights=weights)
                 for d in (0.0, 0.02, 0.05, 0.10)}
        curve_kind = "wild-cluster p-value at each margin"
        adj = wild_cluster_pvalue(lo_d - delta_eval, lc, margin=margin,
                                  B=B_SENS, seed=seed, weights=weights)
        sens = {"delta_eval": float(delta_eval), "engine": "wild-cluster",
                "p_value": wild_p, "p_value_adjusted": adj,
                "decision_changes": bool((np.isfinite(wild_p) and wild_p < 0.05)
                                         != (np.isfinite(adj) and adj < 0.05))}
        reject = bool(np.isfinite(wild_p) and wild_p < 0.05)
        basis = "wild-cluster fallback p-value on the lower endpoint"
    else:
        wild_p = float("nan")
        curve = threshold_curve(lo_d, lc, B=B_CURVE, seed=seed, weights=weights)
        curve_kind = "LCB - delta"
        sens = differential_evaluator_sensitivity(lo_d, lc, delta_eval,
                                                  weights=weights, B=B_SENS,
                                                  seed=seed)
        sens["engine"] = "percentile-t"
        reject = bool(np.isfinite(lcb_lo) and lcb_lo > margin)
        basis = "percentile-t lower bound on the lower endpoint"

    cc = point_estimate(np.asarray(diffs, dtype=float), obs_codes, weights) \
        if diffs else float("nan")
    return {
        "mode": "confirmatory", "manifest_provenance": "package-pinned",
        "admission": {"version": adm.version, "schema": adm.schema,
                      "digest": adm.digest, "source": adm.source,
                      "module_universe": modules},
        "modules": modules, "n_bonferroni": len(modules),
        "n_families_admitted": len(observed), "weights": weights.tolist(),
        "strata": strata, "estimation_performed": True,
        "partial_identification": {"lower": lo, "upper": hi},
        "lcb_lower_endpoint": lcb_lo, "degeneracy": degen,
        "secondary_complete_case": cc,
        "target_population_mismatch": bool(np.isfinite(cc)
                                           and not (lo <= cc <= hi)),
        "threshold_curve": curve, "threshold_curve_kind": curve_kind,
        "differential_evaluator": sens,
        "used_fallback": use_fallback, "wild_p_value": wild_p,
        "decision_basis": basis,
        "evaluator_gate": eg, "yield_gate": yg,
        "negative_controls": {"all_present": True, "all_rejected": True},
        "metrics": metric_suite(observed),
        "bootstrap_counts": {"primary": B_PRIM, "wild": B_WILD,
                             "curve": B_CURVE, "sensitivity": B_SENS},
        "admissible": True, "reject_H0": reject,
        "verdict": ("reject H0: the lower partial-identification endpoint "
                    "exceeds the margin" if reject else "fail to reject H0"),
    }


def run_exploratory_analysis(*args, **kwargs) -> Dict[str, object]:
    """The permissive API. Round-7 blocker 1 and M18: subsets, custom
    manifests, caller-declared module sets and precomputed scalar defects live
    HERE and only here, and the result can never carry the bare label
    `admissible`. It reports `mode: exploratory` and `admissible_exploratory`,
    so no permissive run can be mistaken for a confirmatory verdict."""
    res = dict(_run_legacy_scalar_analysis(*args, **kwargs))
    res["mode"] = "exploratory"
    res["route"] = "legacy scalar"
    if "admissible" in res:
        res["admissible_exploratory"] = res.pop("admissible")
    res.pop("estimation_performed", None) if False else None
    res["verdict"] = "EXPLORATORY (not a confirmatory verdict): " + str(
        res.get("verdict", ""))
    return res


def run_grid(targets=(0.0, 0.05, 0.15), n_families=(24,), n_modules=6,
             reps=25, seed=7, p_apparatus=0.04, apparatus_offset=0.0,
             p_refusal=0.02, p_parsefail=0.03, p_answertype=0.0, p_judge=0.02,
             heterogeneity=(1.0,), B_prim=B_PRIM, B_wild=B_WILD,
             B_curve=B_CURVE, B_sens=B_SENS, calib_reps=120,
             transport_factor=1) -> List[Dict[str, float]]:
    """Calibration grid on the PAIRED observation model (round-8 B05).

    Every cell now simulates paired canonical/transformed responses through
    `simulate_observed_families`, derives the endpoint from those records by
    exactly the declared aggregation, and calibrates beta per cell against the
    realized directional LOWER ENDPOINT. Bootstrap counts are four distinct
    fixed values.

    SCOPE, stated exactly and matched by the manuscript: this calibrates the
    INFERENCE ENGINE on paired records. It does not exercise the fail-closed
    confirmatory gates -- manifest identity, audit provenance join and the N-i
    control -- which are exercised instead by the confirmatory regression
    tests against a fully valid synthetic registry."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for n in n_families:
        for het in heterogeneity:
            span = math.log(het) if het > 0 else 0.0
            mu = [-1.2 - 0.5 * span * (k / max(1, n_modules - 1))
                  for k in range(n_modules)]
            sigma = [0.6 * (het ** (k / max(1, n_modules - 1)))
                     for k in range(n_modules)]
            kw = dict(n_modules=n_modules, mu=mu, sigma=sigma,
                      p_apparatus=p_apparatus, apparatus_offset=apparatus_offset,
                      p_refusal=p_refusal, p_parsefail=p_parsefail,
                      p_answertype=p_answertype, p_judge=p_judge,
                      transport_factor=transport_factor)
            for target in targets:
                beta, realized = _solve_beta_paired(target, n=n, reps=calib_reps,
                                                    seed=seed, **kw)
                los, rejects, fallback = [], 0, 0
                for _ in range(reps):
                    obs = simulate_observed_families(
                        n, n_modules, beta=beta, rng=rng,
                        mu=mu, sigma=sigma, p_apparatus=p_apparatus,
                        apparatus_offset=apparatus_offset, p_refusal=p_refusal,
                        p_parsefail=p_parsefail, p_answertype=p_answertype,
                        p_judge=p_judge, transport_factor=transport_factor)
                    lo_d, codes, w = _paired_lower_vector(obs)
                    if lo_d.size == 0:
                        continue
                    los.append(point_estimate(lo_d, codes, w))
                    lcb, degen = percentile_t_lcb(lo_d, codes, B=B_prim,
                                                  seed=int(rng.integers(1 << 30)),
                                                  weights=w)
                    fb = bool(min_cluster_count(codes) < MIN_CLUSTERS_PER_STRATUM
                              or degen > MAX_DEGENERACY or not np.isfinite(lcb))
                    fallback += int(fb)
                    if fb:
                        pv = wild_cluster_pvalue(lo_d, codes, margin=DELTA_SEM,
                                                 B=B_wild,
                                                 seed=int(rng.integers(1 << 30)),
                                                 weights=w)
                        rejects += int(np.isfinite(pv) and pv < 0.05)
                    else:
                        rejects += int(np.isfinite(lcb) and lcb > DELTA_SEM)
                rows.append({
                    "n_families": float(n), "n_modules": float(n_modules),
                    "heterogeneity": float(het), "mu0": float(mu[0]),
                    "sigma0": float(sigma[0]),
                    "p_apparatus": float(p_apparatus),
                    "apparatus_offset": float(apparatus_offset),
                    "p_refusal": float(p_refusal),
                    "p_parsefail": float(p_parsefail),
                    "p_answertype": float(p_answertype),
                    "p_judge": float(p_judge),
                    "transport_factor": float(transport_factor),
                    "r_replicates": float(R_REPLICATES),
                    "B_prim": float(B_prim), "B_wild": float(B_wild),
                    "B_curve": float(B_curve), "B_sens": float(B_sens),
                    "calib_reps": float(calib_reps), "seed": float(seed),
                    "target_lower_endpoint": float(target), "beta": float(beta),
                    "calibrated_lower_endpoint": float(realized),
                    "mean_lower_endpoint":
                        float(np.mean(los)) if los else float("nan"),
                    "reps": float(reps),
                    "rejection_rate": rejects / max(1, reps),
                    "fallback_rate": fallback / max(1, reps),
                    "scope": "inference engine on paired records; gates not exercised",
                })
    return rows


def _paired_lower_vector(obs):
    """Directional LOWER endpoint vector, module codes and fixed weights, all
    derived from paired records by the declared aggregation."""
    mods = sorted({o.module for o in obs})
    counts = np.array([sum(1 for o in obs if o.module == m) for m in mods],
                      dtype=float)
    w = counts / counts.sum()
    v, c = [], []
    for o in obs:
        lo, _hi = o.difference_bounds()      # missing replicates voted through
        v.append(lo)
        c.append(mods.index(o.module))
    return np.asarray(v, dtype=float), np.asarray(c, dtype=int), w


def _paired_mean_lower(beta, *, n, reps, seed, **kw):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        obs = simulate_observed_families(n, kw["n_modules"], beta=beta, rng=rng,
                                         **{k: v for k, v in kw.items()
                                            if k != "n_modules"})
        lo_d, codes, w = _paired_lower_vector(obs)
        if lo_d.size:
            vals.append(point_estimate(lo_d, codes, w))
    return float(np.mean(vals)) if vals else float("nan")


def _solve_beta_paired(target, *, n, reps, seed, tol=0.006, **kw):
    """Round-8 B05: beta is solved per cell against the realized lower endpoint
    of the PAIRED model, including the zero target (beta may be negative)."""
    lo, hi = (-3.0, 3.0) if target <= 0 else (0.0, 6.0)
    best = 0.5 * (lo + hi)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        got = _paired_mean_lower(mid, n=n, reps=reps, seed=seed, **kw)
        best = mid
        if not np.isfinite(got) or abs(got - target) < tol:
            break
        if got < target:
            lo = mid
        else:
            hi = mid
    return best, _paired_mean_lower(best, n=n, reps=reps, seed=seed, **kw)


def benchmark_analysis(reps: int = 3, **kw) -> Dict[str, float]:
    """Round-7 M11. Returns BOTH quantities, unambiguously:

        seconds_per_analysis  = wall time / reps   (one primary analysis)
        seconds_per_cell_at   = seconds_per_analysis * reps_declared

    The full-grid estimate is  972 cells x reps_declared x seconds_per_analysis.
    The old `benchmark_analysis` name divided a one-cell run by reps and called the
    result seconds per cell, which was two different things at once."""
    import time
    t0 = time.time()
    run_grid(targets=(0.05,), reps=reps, calib_reps=8, **kw)
    total = time.time() - t0
    per = total / max(1, reps)
    return {"seconds_total": total, "reps_measured": float(reps),
            "seconds_per_analysis": per,
            "seconds_per_cell_at_2000_reps": per * 2000,
            "estimated_full_grid_seconds": per * 2000 * 972}


def run_full_grid(reps: int = 2000, seed: int = BOOT_SEED,
                  out_path: str = "calibration_grid.json", resume: bool = True,
                  workers: int = 1, calib_reps: int = 120) -> str:
    """The declared factorial calibration of Section 9.6.

        targets       {0, 0.025, 0.05, 0.075, 0.10, 0.15}  (LOWER-ENDPOINT scale)
        heterogeneity {1:1, 2:1, 4:1}
        apparatus     {0, 0.05, 0.15} x structural offset {0, 0.05}
        judge error   {0, 0.02, 0.05}
        N             {24, 30, 48}

    972 cells. Each cell calibrates beta against its own realized lower
    endpoint and runs the primary analysis at the DECLARED bootstrap counts
    (20,000 / 4,000 / 4,000 / 2,000) -- not one shared value.

    RUNTIME. `benchmark_analysis()` measures seconds per analysis in the current
    environment; multiply by 972 x reps. At the released defaults this is a
    multi-day sequential job, which is why `workers > 1` runs cells in parallel
    processes and why results are written incrementally and the run is
    resumable. The earlier 'hours' claim was not supported by measurement and
    has been withdrawn."""
    import json, os, itertools
    from concurrent.futures import ProcessPoolExecutor
    done: Dict[str, Dict] = {}
    if resume and os.path.exists(out_path):
        done = {r["cell"]: r for r in json.load(open(out_path, encoding="utf-8"))}
    targets = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15)
    hets = (1.0, 2.0, 4.0)
    apps = ((0.0, 0.0), (0.05, 0.0), (0.05, 0.05),
            (0.15, 0.0), (0.15, 0.05), (0.0, 0.05))
    judges = (0.0, 0.02, 0.05)
    Ns = (24, 30, 48)
    jobs = []
    for idx, (t, het, (pa, off), pj, n) in enumerate(
            itertools.product(targets, hets, apps, judges, Ns), start=1):
        cell = f"t{t}_h{het}_a{pa}_o{off}_j{pj}_n{n}"
        if cell in done:
            continue
        jobs.append((idx, cell, dict(targets=(t,), n_families=(n,), reps=reps,
                                     seed=seed + idx, p_apparatus=pa,
                                     apparatus_offset=off, p_judge=pj,
                                     heterogeneity=(het,), calib_reps=calib_reps)))
    def _emit(cell, row):
        row = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
               for k, v in row.items()}          # JSON null, never NaN (M12)
        row["cell"] = cell
        row["status"] = "ok"
        done[cell] = row
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(sorted(done.values(), key=lambda r: r["cell"]), fh, indent=1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_grid, **kw): cell for _i, cell, kw in jobs}
            for fut, cell in futs.items():
                _emit(cell, fut.result()[0])
    else:
        for _i, cell, kw in jobs:
            _emit(cell, run_grid(**kw)[0])
    return out_path


# ------------------------------------------------------------------- driver --
def _demo() -> int:
    """Fast smoke demo. Expected runtime: a few seconds. The declared factorial
    calibration is a separate batch job: `python analysis.py --full-grid`."""
    import time
    t0 = time.time()
    print("IMR analysis pipeline -- SYNTHETIC DATA ONLY, no model is run.\n")
    mu, sg = [-1.3] * 6, [0.5] * 6
    rng = np.random.default_rng(BOOT_SEED)
    fams = simulate_families(24, 6, beta=solve_beta(0.15, mu, sg), mu=mu, sigma=sg,
                             rng=rng, p_apparatus=0.04, p_refusal=0.02,
                             p_parsefail=0.03, p_judge=0.02)
    mods = declared_modules(fams)
    n_req = required_cluster_count()

    print("gate order: admission -> strata -> evaluator -> yield -> estimation\n")
    print("fail-closed identity: simulated families are NOT in the frozen manifest")
    try:
        _run_legacy_scalar_analysis(fams)   # private legacy route; fails closed here
    except ProtocolError as exc:
        print(f"  confirmatory path correctly refuses -> {str(exc)[:66]}...")
    print("  (the smoke path below is explicitly marked permissive)\n")

    r0 = _run_legacy_scalar_analysis(
        fams, admission=Admission(permissive=True), require_manifest=False,
        family_defects_by_module={m: [0.0] * 4 for m in mods},
        B_prim=600, B_wild=400, B_curve=400, B_sens=200)
    print(f"pilot audit (4 families/module) : {r0['verdict']}")
    print(f"  estimation performed          : {r0['estimation_performed']}  "
          f"<- no endpoint is computed when a gate fails")
    print(f"  evaluator max upper bound     : {r0['evaluator_gate']['max_upper']:.4f} "
          f"(exact Clopper-Pearson, Bonferroni alpha/6)")

    r = _run_legacy_scalar_analysis(
        fams, admission=Admission(permissive=True), require_manifest=False,
        family_defects_by_module={m: [0.0] * n_req for m in mods},
        delta_eval=0.01, B_prim=1500, B_wild=800, B_curve=800, B_sens=400)
    pi = r["partial_identification"]
    print(f"\ngate-size audit ({n_req} families/module):")
    print(f"  derived yield (from records)  : "
          + ", ".join(f"{k}: resp {v[0]:.3f}, parse {v[1]:.3f}"
                      for k, v in sorted(r['derived_yield'].items())))
    print(f"  partial identification        : [{pi['lower']:+.4f}, {pi['upper']:+.4f}]")
    print(f"  LCB on the LOWER endpoint     : {r['lcb_lower_endpoint']:+.4f}")
    print(f"  complete-case (secondary)     : {r['secondary_complete_case']:+.4f} "
          f"(target-population mismatch: {r['target_population_mismatch']})")
    print(f"  engine                        : {r['decision_basis']}")
    print(f"  VERDICT                       : {r['verdict']}")

    print(f"\nexact interval, the round-5 countercheck:")
    print(f"  n=94, k=1, alpha=0.05/6 -> CP {family_cluster_upper([1.0] + [0.0] * 93, alpha=0.05/6):.4f} "
          f"(fails the 0.05 gate, as it must)")
    print(f"  n=94, k=0, alpha=0.05/6 -> CP {family_cluster_upper([0.0] * 94, alpha=0.05/6):.4f}")

    print("\nfrozen admission manifest:")
    adm = load_admission()
    print(f"  schema {adm.schema}, version {adm.version}, digest {adm.digest[:12]}")
    print(f"  {len(adm.records)} families recorded, {len(adm.eligible_ids)} "
          f"confirmatory-eligible, module universe {list(adm.module_universe)}")
    print(f"  modules with no eligible family: {list(adm.excluded_modules)}  "
          f"<- a six-module maximum cannot be evaluated, so no confirmatory run")

    print("\nsmoke calibration grid (full factorial: --full-grid):")
    for row in run_grid(reps=12, calib_reps=30, B_prim=200, B_wild=150,
                        B_curve=150, B_sens=100):
        print(f"  target={row['target_lower_endpoint']:.3f} "
              f"beta={row['beta']:+.3f} "
              f"calibrated={row['calibrated_lower_endpoint']:+.3f} "
              f"observed={row['mean_lower_endpoint']:+.3f} "
              f"reject={row['rejection_rate']:.2f} "
              f"fallback={row['fallback_rate']:.2f}")

    print(f"\nelapsed: {time.time() - t0:.1f}s. "
          "No empirical claim about any language model is made by this file.")
    return 0


def _main(argv: Sequence[str]) -> int:
    """CLI. Default is the fast smoke demo; --full-grid runs the declared
    factorial calibration, which is a resumable batch job."""
    args = list(argv)
    if "--full-grid" in args:
        reps = 2000
        for a in args:
            if a.startswith("--reps="):
                reps = int(a.split("=", 1)[1])
        bm = benchmark_analysis(reps=2)
        est = bm["estimated_full_grid_seconds"] * (reps / 2000.0)
        print(f"declared factorial calibration grid: 972 cells x {reps} reps.")
        print(f"measured {bm['seconds_per_analysis']:.2f} s per primary analysis "
              f"here; estimated total {est/3600:.1f} h "
              f"({est/86400:.1f} days) sequential.")
        print("This is a multi-day chunked batch job, not a smoke test. "
              "Resumable; use workers>1 to parallelize. Writing "
              "calibration_grid.json")
        run_full_grid(reps=reps)
        return 0
    return _demo()


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(_main(_sys.argv[1:]))
