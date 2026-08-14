"""
imr/observation.py -- the executable IMR observation.

Every quantity the paper reports is DERIVED here from immutable paired records.
Round-8 closes the remaining gaps between the defined and the executable
instrument:

  B02  total certified transport tau_T, applied to EVERY normalized value,
       with a declared inverse, typed answer spaces and a certificate digest.
       Gold-only "transport" is gone.
  B03  construction-time validation and cryptographic joining of records on
       (family, module, arrow, replicate), with presentation ids and full
       apparatus provenance.
  B04  the N-i negative control is a required, validated record.
  B07  an explicit answer-TYPE state, so a wrong-typed answer routes to
       F_answer-type and not to F_canonical.
  B08  aggregation levels are declared and implemented identically:
       replicate -> arrow (conservative majority) -> family -> module -> model.
  M08  the parse-failure sentinel is a unique object, not a string compared by
       identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple


class _ParseFail:
    """Unique sentinel. An equal-but-separately-constructed string can never be
    mistaken for it (round-8 M08)."""
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "PARSE_FAIL"

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return hash("imr.parse-fail.sentinel")


PARSE_FAIL = _ParseFail()
R_REPLICATES = 4
REQUIRED_ARROWS = ("T1", "T2", "s")
ARM_KIND_OF = {"T1": "structural", "T2": "structural", "s": "surface"}


class FinishReason(str, Enum):
    """Typed provider finish-reason vocabulary (round-11 section 3.2). The
    provider's original string is preserved separately in `raw_finish_reason`
    when it does not map onto a known value."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_ERROR = "tool_error"
    TRANSPORT_ERROR = "transport_error"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN = "unknown_provider_value"
    SYNTHETIC = "synthetic"

    @property
    def is_apparatus(self) -> bool:
        return self in (FinishReason.TRANSPORT_ERROR, FinishReason.TIMEOUT,
                        FinishReason.PROVIDER_ERROR, FinishReason.TOOL_ERROR)

    @property
    def is_budget_truncation(self) -> bool:
        return self is FinishReason.LENGTH


SYNTHETIC_DIGEST = "synthetic:" + "0" * 54          # explicit, never SHA-256


def is_sha256(s: str) -> bool:
    return (isinstance(s, str) and len(s) == 64
            and all(c in "0123456789abcdef" for c in s))


class ObservationError(ValueError):
    """Any record that does not instantiate the protocol."""


class Status(str, Enum):
    """Typed response-process status; one value per generation."""
    OK = "ok"                        # a well-typed answer was obtained
    ANSWER_TYPE = "answer_type"      # extractable but WRONG TYPE (round-8 B07)
    REFUSAL = "refusal"              # declined; behaviour, counts as defect
    PARSE_FAIL = "parse_fail"        # attempted, nothing extractable
    TRUNCATED_BUDGET = "truncated_budget"   # budget exhaustion -> parse failure
    APPARATUS = "apparatus"          # provider transport error -> UNOBSERVED
    GRADER_MISSING = "grader_missing"       # no grader -> UNOBSERVED

    @property
    def observed(self) -> bool:
        return self not in (Status.APPARATUS, Status.GRADER_MISSING)

    @property
    def behavioural_defect(self) -> bool:
        return self in (Status.REFUSAL, Status.PARSE_FAIL,
                        Status.TRUNCATED_BUDGET, Status.ANSWER_TYPE)


# --------------------------------------------------------------- transport --
@dataclass(frozen=True)
class Transport:
    """A TOTAL certified answer-space bijection tau_T : A_x -> A_Tx.

    Round-10 blocker 1. Two defects are fixed here.

    (a) IMAGE/TARGET EQUALITY. Checking tau_inverse(tau(x)) == x on the source
        and tau(tau_inverse(y)) == y on the target does not force
        tau(source) == target: a declared target of (2, 6) for tau(x) = 2x on
        (1, 2) passed both laws while the image was {2, 4}. The equality is now
        enforced directly.

    (b) TOTALITY. A finite tuple is a PROBE SET, not an answer space. Declaring
        a finite `source_domain` for an infinite type such as int made
        `apply(3)` raise instead of returning a defect. A transport now
        declares its space in one of two honest ways:

          domain_kind="finite"    -- the answer space IS the enumerated set;
                                     values outside it are a protocol error
          domain_kind="typed"     -- the space is the declared type, described
                                     by `source_predicate`/`target_predicate`;
                                     the enumerated tuple is only a probe set
                                     used to verify the laws, and `apply` is
                                     total on every value satisfying the
                                     predicate

    A wrong-but-well-typed answer therefore transports normally and yields a
    defect; it never crashes the discrepancy calculation."""
    transport_id: str
    transport_version: str
    source_answer_type: str
    target_answer_type: str
    transport_certificate_digest: str
    tau: Callable[[object], object]
    tau_inverse: Callable[[object], object]
    source_domain: Tuple = ()
    target_domain: Tuple = ()
    source_predicate: Optional[Callable[[object], bool]] = None
    target_predicate: Optional[Callable[[object], bool]] = None
    domain_kind: str = "finite"          # "finite" | "typed"
    arrow_certificate_digest: str = ""
    # Round-12 Patch B. The token is attached ONLY by the registry resolver,
    # via object.__setattr__. The public constructor refuses a non-None value,
    # so a caller cannot mint certification.
    _registry_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._registry_token is not None:
            raise ObservationError(
                "the public Transport constructor does not accept a registry "
                "token; production transports are issued by the registry")
        for f in ("transport_id", "transport_version", "source_answer_type",
                  "target_answer_type", "transport_certificate_digest"):
            if not getattr(self, f):
                raise ObservationError(f"transport: {f} must be non-empty")
        if not (callable(self.tau) and callable(self.tau_inverse)):
            raise ObservationError("transport: tau and tau_inverse must be callable")
        if self.domain_kind not in ("finite", "typed"):
            raise ObservationError("transport: domain_kind must be 'finite' or 'typed'")
        if not self.source_domain:
            raise ObservationError(
                "transport: a source domain (finite space, or probe set for a "
                "typed space) is required; a bijection cannot be verified over "
                "an unspecified set")
        if self.domain_kind == "typed" and not (self.source_predicate
                                                and self.target_predicate):
            raise ObservationError(
                "transport: a typed space needs source_predicate and "
                "target_predicate, so that `apply` is total on the type")

        image = tuple(self.tau(x) for x in self.source_domain)
        declared = tuple(self.target_domain) if self.target_domain else image
        object.__setattr__(self, "target_domain", declared)

        # (a) image/target equality -- the check that was missing
        if self.domain_kind == "finite":
            if set(image) != set(declared):
                raise ObservationError(
                    f"transport {self.transport_id}: tau(source_domain) = "
                    f"{sorted(set(image), key=repr)} != declared target_domain "
                    f"{sorted(set(declared), key=repr)}")
        else:
            for y in image:
                if not self.target_predicate(y):
                    raise ObservationError(
                        f"transport {self.transport_id}: image {y!r} fails the "
                        "declared target predicate")

        for x in self.source_domain:
            if self.source_predicate and not self.source_predicate(x):
                raise ObservationError(f"transport: probe {x!r} fails the source "
                                       "predicate")
            y = self.tau(x)
            if self.tau_inverse(y) != x:
                raise ObservationError(
                    f"transport {self.transport_id}: tau_inverse(tau({x!r})) != "
                    f"{x!r} -- not injective / inverse is wrong")
        for y in declared:
            if self.target_predicate and not self.target_predicate(y):
                raise ObservationError(f"transport: target probe {y!r} fails the "
                                       "target predicate")
            x = self.tau_inverse(y)
            if self.tau(x) != y:
                raise ObservationError(
                    f"transport {self.transport_id}: tau(tau_inverse({y!r})) != "
                    f"{y!r} -- not surjective onto the declared target")
        if len(set(image)) != len(tuple(self.source_domain)):
            raise ObservationError(
                f"transport {self.transport_id}: not injective on its domain")
        if self.source_answer_type == self.target_answer_type:
            if self.domain_kind == "finite" and set(image) != set(self.source_domain):
                raise ObservationError(
                    f"transport {self.transport_id}: source and target answer "
                    f"types are both {self.source_answer_type!r}, so tau must be "
                    f"an automorphism; image {sorted(set(image), key=repr)} != "
                    f"domain {sorted(set(self.source_domain), key=repr)}")

    def accepts(self, value) -> bool:
        if self.domain_kind == "typed":
            return bool(self.source_predicate(value))
        return value in self.source_domain

    def apply(self, value):
        """Total on the declared space. A well-typed value outside the probe
        set transports normally; only a value outside the declared SPACE is a
        protocol error."""
        if not self.accepts(value):
            raise ObservationError(
                f"transport {self.transport_id} is declared on "
                f"{self.source_answer_type!r} ({self.domain_kind}); {value!r} is "
                "outside that space")
        return self.tau(value)

    def check_roundtrip(self, probes: Sequence[object] = ()) -> None:
        for p in (probes or self.source_domain):
            if self.tau_inverse(self.tau(p)) != p:
                raise ObservationError(f"transport not invertible at {p!r}")


def identity_transport(answer_type: str = "int", digest: str = "identity",
                       domain: Sequence = (1, 2)) -> Transport:
    """A FINITE identity transport for tests and exploratory use.

    Round-12: this is deliberately `finite`, not `typed`. Production typed
    transports come only from the sealed registry, which owns their symbolic
    definition and generates their implementations."""
    return Transport("id", "1", answer_type, answer_type, digest,
                     lambda v: v, lambda v: v, source_domain=tuple(domain),
                     domain_kind="finite", arrow_certificate_digest=digest)


def negation_transport(answer_type: str = "int", digest: str = "identity",
                       domain: Sequence = (1, 2)) -> Transport:
    """A FINITE negation transport for tests. Production typed negation is
    `BuiltinTransportId.INT_NEGATION_V1`, issued by the sealed registry."""
    dom = tuple(domain) + tuple(-v for v in domain)
    return Transport("neg", "1", answer_type, answer_type, digest,
                     lambda v: -v, lambda v: -v, source_domain=dom,
                     domain_kind="finite", arrow_certificate_digest=digest)


# ------------------------------------------------------------------ record --
@dataclass(frozen=True)
class SideResponse:
    """One side of one administered arrow, for one replicate.

    Round-11 section 3: provenance is mandatory for observed data. An
    `observed` record must carry a typed finish reason and a SHA-256 raw
    digest; a `synthetic` record must say so and is refused by the
    confirmatory route."""
    presentation_id: str
    status: Status
    normalized: object = None
    provider_finish_reason: object = ""
    raw_digest: str = ""
    record_kind: str = "synthetic"
    raw_finish_reason: str = ""

    def __post_init__(self):
        if not self.presentation_id:
            raise ObservationError("presentation_id must be non-empty")
        if not isinstance(self.status, Status):
            raise ObservationError(f"status must be a Status, got {self.status!r}")
        if self.record_kind not in ("observed", "synthetic"):
            raise ObservationError(
                f"record_kind must be 'observed' or 'synthetic', got "
                f"{self.record_kind!r}")
        fr = self.provider_finish_reason
        if isinstance(fr, str) and fr:
            try:
                fr = FinishReason(fr)
            except ValueError:
                object.__setattr__(self, "raw_finish_reason",
                                   self.raw_finish_reason or str(fr))
                fr = FinishReason.UNKNOWN
        if self.record_kind == "observed":
            if not isinstance(fr, FinishReason) or fr is FinishReason.SYNTHETIC:
                raise ObservationError(
                    f"{self.presentation_id}: an observed response requires a "
                    "typed provider finish reason")
            if not is_sha256(self.raw_digest):
                raise ObservationError(
                    f"{self.presentation_id}: an observed response requires a "
                    "SHA-256 raw-response digest (64 lowercase hex characters)")
            # truncation classification must agree with the finish reason
            if fr.is_budget_truncation and self.status not in (
                    Status.TRUNCATED_BUDGET, Status.PARSE_FAIL):
                raise ObservationError(
                    f"{self.presentation_id}: finish reason 'length' is budget "
                    f"truncation, which is a parse failure, not "
                    f"{self.status.value}")
            if fr.is_apparatus and self.status is not Status.APPARATUS:
                raise ObservationError(
                    f"{self.presentation_id}: finish reason {fr.value} is an "
                    f"apparatus failure, not {self.status.value}")
            if self.status is Status.APPARATUS and not fr.is_apparatus:
                raise ObservationError(
                    f"{self.presentation_id}: status APPARATUS requires an "
                    "apparatus finish reason")
        else:
            object.__setattr__(self, "raw_digest",
                               self.raw_digest or SYNTHETIC_DIGEST)
            if fr == "" or fr is None:
                fr = FinishReason.SYNTHETIC
        object.__setattr__(self, "provider_finish_reason", fr)
        if self.status is Status.OK:
            if self.normalized is PARSE_FAIL or self.normalized is None:
                raise ObservationError(
                    "status OK requires a well-typed normalized answer")
        else:
            if self.normalized is not None and self.normalized is not PARSE_FAIL:
                raise ObservationError(
                    f"status {self.status.value} must not carry a typed answer")

    def parsed(self) -> bool:
        return self.status is Status.OK and self.normalized is not PARSE_FAIL


class AntiCertificateRegistry:
    """The frozen registry of anti-equivalence certificates.

    Round-10 blocker 2. `anti_certificate_replayed` is no longer a constructor
    argument that the gate believes. The confirmatory route resolves the
    certificate id against this registry, checks its digest, and EXECUTES it;
    the replay result is produced here, never accepted."""

    def __init__(self, entries: Dict[str, Dict] = None):
        self._e = dict(entries or {})

    def register(self, cert_id: str, digest: str,
                 verifier: Callable[["NegativeControl"], bool]) -> None:
        if not cert_id or not digest:
            raise ObservationError("registry: id and digest must be non-empty")
        self._e[cert_id] = {"digest": digest, "verifier": verifier}

    def replay(self, control: "NegativeControl") -> Tuple[bool, str]:
        """Returns (replayed, reason_if_not)."""
        e = self._e.get(control.anti_certificate_id)
        if e is None:
            return False, (f"anti-certificate {control.anti_certificate_id!r} is "
                           "not in the frozen registry")
        if e["digest"] != control.anti_certificate_digest:
            return False, (f"anti-certificate digest mismatch for "
                           f"{control.anti_certificate_id!r}")
        try:
            ok = bool(e["verifier"](control))
        except Exception as exc:
            return False, f"anti-certificate raised on replay: {exc}"
        return (ok, "" if ok else "anti-certificate replay returned false")


@dataclass(frozen=True)
class NegativeControl:
    """The N-i control: a well-posed neighbouring problem with a DIFFERENT valid
    gold, which the apparatus must reject as non-equivalent.

    Round-9 derived the verdict from the record. Round-10 removes the last
    asserted premise: replay is performed by the registry, and the control must
    belong to its enclosing family and share that family's parser and renderer
    versions."""
    family_id: str
    presentation_id: str
    canonical_gold: object
    neighboring_gold: object
    anti_certificate_id: str
    anti_certificate_digest: str
    normalizer_status: Status
    normalizer_output: object
    parser_version: str
    renderer_version: str
    provenance_digest: str
    control_type: str = "N-i"

    def __post_init__(self):
        for f in ("family_id", "presentation_id", "anti_certificate_id",
                  "anti_certificate_digest", "parser_version",
                  "renderer_version", "provenance_digest"):
            if not getattr(self, f):
                raise ObservationError(f"N-i control: {f} must be non-empty")
        if self.control_type != "N-i":
            raise ObservationError(
                f"confirmatory runs require a type N-i control, got "
                f"{self.control_type!r}")
        if not isinstance(self.normalizer_status, Status):
            raise ObservationError("N-i control: normalizer_status must be a Status")

    def record_reasons(self) -> List[str]:
        """Reasons visible from the record alone (no replay)."""
        out = []
        if self.canonical_gold == self.neighboring_gold:
            out.append("neighbouring gold equals the canonical gold: the control "
                       "is not a distinct problem")
        if self.normalizer_status is not Status.OK:
            out.append(f"normalizer status {self.normalizer_status.value}: no "
                       "typed control answer was obtained")
        elif self.normalizer_output == self.canonical_gold:
            out.append("normalizer collapsed the control onto the canonical "
                       "gold: a degenerate normalizer is not excluded")
        elif self.normalizer_output != self.neighboring_gold:
            out.append("normalizer output does not match the neighbouring gold")
        return out

    def reasons(self, registry: "AntiCertificateRegistry" = None,
                family_id: str = None, parser_version: str = None,
                renderer_version: str = None) -> List[str]:
        out = list(self.record_reasons())
        if family_id is not None and self.family_id != family_id:
            out.append(f"control belongs to family {self.family_id!r}, not "
                       f"{family_id!r}")
        if parser_version is not None and self.parser_version != parser_version:
            out.append("control parser version differs from the family's")
        if renderer_version is not None and self.renderer_version != renderer_version:
            out.append("control renderer version differs from the family's")
        if registry is None:
            out.append("no anti-certificate registry supplied: replay could not "
                       "be performed, and replay is never assumed")
        else:
            replay = getattr(registry, "replay_anti", None) or registry.replay
            ok, why = replay(self)
            if not ok:
                out.append(why)
        return out

    def rejected(self, registry: "AntiCertificateRegistry" = None, **kw) -> bool:
        return not self.reasons(registry, **kw)


@dataclass(frozen=True)
class ArrowResponse:
    """THE immutable observation: both sides of one certified arrow, once."""
    family_id: str
    module: str
    arrow_id: str
    replicate: int
    certificate_digest: str
    canonical: SideResponse
    transformed: SideResponse
    canonical_gold: object
    transport: Transport
    judge_version: str
    parser_version: str
    renderer_version: str

    def __post_init__(self):
        if self.arrow_id not in REQUIRED_ARROWS:
            raise ObservationError(f"arrow_id {self.arrow_id!r} not in "
                                   f"{REQUIRED_ARROWS}")
        if not (0 <= self.replicate < R_REPLICATES):
            raise ObservationError(f"replicate {self.replicate} outside "
                                   f"0..{R_REPLICATES - 1}")
        for f in ("family_id", "module", "certificate_digest", "judge_version",
                  "parser_version", "renderer_version"):
            if not getattr(self, f):
                raise ObservationError(f"{f} must be non-empty")
        if not isinstance(self.transport, Transport):
            raise ObservationError("transport must be a Transport")
        if self.transport.arrow_certificate_digest != self.certificate_digest:
            raise ObservationError(
                f"{self.family_id}/{self.arrow_id}: the transport's arrow "
                "certificate digest does not match the arrow record's")

    @property
    def arm_kind(self) -> str:
        return ARM_KIND_OF[self.arrow_id]

    @property
    def transported_gold(self):
        return self.transport.apply(self.canonical_gold)

    # ---- derived quantities ------------------------------------------------
    def joint_parse(self) -> bool:
        return self.canonical.parsed() and self.transformed.parsed()

    def apparatus_lost(self) -> bool:
        return not (self.canonical.status.observed
                    and self.transformed.status.observed)

    def canonical_correct(self) -> Optional[bool]:
        if not self.canonical.parsed():
            return None
        return self.canonical.normalized == self.canonical_gold

    def transformed_correct(self) -> Optional[bool]:
        """Round-9 Patch Set 4. Behavioural non-response is CHARGED, not
        excused: refusal, parse failure, answer-type failure and budget
        truncation are incorrect (0). Only genuine apparatus or grader
        missingness is unobserved (None)."""
        st = self.transformed.status
        if not st.observed:
            return None
        if st.behavioural_defect:
            return False
        return self.transformed.normalized == self.transported_gold

    def canonical_correct_charged(self) -> Optional[bool]:
        st = self.canonical.status
        if not st.observed:
            return None
        if st.behavioural_defect:
            return False
        return self.canonical.normalized == self.canonical_gold

    def transport_discrepancy(self) -> Optional[float]:
        """delta_T = d( tau_T(Pi_x(M(x))), Pi_Tx(M(Tx)) ), with tau_T applied
        to the ACTUAL normalized canonical answer (round-8 B02)."""
        if not self.joint_parse():
            return None
        moved = self.transport.apply(self.canonical.normalized)
        return 0.0 if moved == self.transformed.normalized else 1.0

    def composite_defect(self) -> Optional[float]:
        if self.apparatus_lost():
            return None
        if (self.canonical.status.behavioural_defect
                or self.transformed.status.behavioural_defect):
            return 1.0
        d = self.transport_discrepancy()
        return d if d is not None else 1.0

    def failure_label(self) -> str:
        """Frozen precedence: refusal > parse > answer-type > canonical >
        transport. Mechanism and witness are obligation-record labels and are
        not derivable from a final-answer record; Appendix B says so."""
        for side in (self.canonical, self.transformed):
            if side.status is Status.REFUSAL:
                return "F_refusal"
        for side in (self.canonical, self.transformed):
            if side.status in (Status.PARSE_FAIL, Status.TRUNCATED_BUDGET):
                return "F_parse"
        for side in (self.canonical, self.transformed):
            if side.status is Status.ANSWER_TYPE:
                return "F_answer-type"
        if self.apparatus_lost():
            return "F_apparatus"
        if self.canonical_correct() is False:
            return "F_canonical"
        if self.transport_discrepancy():
            return "F_transport"
        return "none"


def majority(xs: Sequence[Optional[float]]) -> Tuple[Optional[float], bool]:
    """Conservative majority over OBSERVED replicates; ties -> defect.

    Retained for reporting. The confirmatory endpoint uses `majority_bounds`,
    because discarding missing replicates before voting is not sound."""
    v = [x for x in xs if x is not None]
    if not v:
        return None, False
    tie = (len(v) % 2 == 0) and (2 * sum(v) == len(v))
    return (1.0 if 2 * sum(v) >= len(v) else 0.0), tie


def majority_bounds(xs: Sequence[Optional[float]],
                    n_slots: int = R_REPLICATES) -> Tuple[float, float]:
    """Round-9 Patch Set 5: propagate MISSING replicates THROUGH the vote.

    Filling the missing slots with 0 gives the lower majority; filling them
    with 1 gives the upper. With outcomes [0, None, None, None] the completions
    [0,0,0,0] -> 0 and [0,1,1,1] -> 1 are both possible, so the arm is
    partially identified at [0, 1] rather than fixed at 0, which is what the
    previous reducer returned."""
    obs = [x for x in xs if x is not None]
    missing = max(0, n_slots - len(obs))
    s = sum(obs)
    lo = 1.0 if 2 * s >= n_slots else 0.0                    # missing -> 0
    hi = 1.0 if 2 * (s + missing) >= n_slots else 0.0        # missing -> 1
    return lo, hi


def _majority_bool(xs: Sequence[Optional[bool]]) -> Optional[bool]:
    """Majority CORRECTNESS over observed replicates; ties -> incorrect."""
    v = [1.0 if x else 0.0 for x in xs if x is not None]
    if not v:
        return None
    return (2 * sum(v) > len(v))


@dataclass(frozen=True)
class ObservedFamily:
    family_id: str
    module: str
    arrows: Tuple[ArrowResponse, ...]
    negative_control: Optional[NegativeControl] = None

    def __post_init__(self):
        seen = set()
        for a in self.arrows:
            if a.family_id != self.family_id:
                raise ObservationError(
                    f"arrow records family {a.family_id} inside family "
                    f"{self.family_id}")
            if a.module != self.module:
                raise ObservationError(
                    f"{self.family_id}: arrow module {a.module} != {self.module}")
            key = (a.arrow_id, a.replicate)
            if key in seen:
                raise ObservationError(f"{self.family_id}: duplicate {key}")
            seen.add(key)
        for aid in REQUIRED_ARROWS:
            recs = [a for a in self.arrows if a.arrow_id == aid]
            reps = sorted(a.replicate for a in recs)
            if reps != list(range(R_REPLICATES)):
                raise ObservationError(
                    f"{self.family_id}/{aid}: replicate set {reps} != "
                    f"{list(range(R_REPLICATES))}")
            # Round-9 Patch Set 3: one certified arrow and one apparatus per
            # arrow. A single replicate carrying a different certificate, judge,
            # parser, renderer or presentation pair was previously hidden by a
            # per-arrow audit dictionary that kept only the last record.
            for field in ("certificate_digest", "judge_version",
                          "parser_version", "renderer_version"):
                vals = {getattr(a, field) for a in recs}
                if len(vals) != 1:
                    raise ObservationError(
                        f"{self.family_id}/{aid}: mixed {field} across "
                        f"replicates: {sorted(vals)}")
            for side in ("canonical", "transformed"):
                pids = {getattr(a, side).presentation_id for a in recs}
                if len(pids) != 1:
                    raise ObservationError(
                        f"{self.family_id}/{aid}: mixed {side} presentation ids "
                        f"across replicates: {sorted(pids)}")
            tids = {(a.transport.transport_id, a.transport.transport_version,
                     a.transport.transport_certificate_digest,
                     a.transport.arrow_certificate_digest) for a in recs}
            if len(tids) != 1:
                raise ObservationError(
                    f"{self.family_id}/{aid}: mixed transport identity across "
                    f"replicates")
        if self.negative_control is not None:
            nc = self.negative_control
            if nc.family_id != self.family_id:
                raise ObservationError(
                    f"{self.family_id}: N-i control belongs to family "
                    f"{nc.family_id!r}")
            pv = {a.parser_version for a in self.arrows}
            rv = {a.renderer_version for a in self.arrows}
            if nc.parser_version not in pv or nc.renderer_version not in rv:
                raise ObservationError(
                    f"{self.family_id}: N-i control parser/renderer versions do "
                    "not match the family's apparatus")

    def by_arrow(self, arrow_id: str) -> List[ArrowResponse]:
        return [a for a in self.arrows if a.arrow_id == arrow_id]

    def arrow_ids(self) -> Tuple[str, ...]:
        return tuple(sorted({a.arrow_id for a in self.arrows}))

    # ---- level 1: replicate -> arrow ---------------------------------------
    def arm_defect(self, arrow_id: str) -> Tuple[Optional[float], bool]:
        return majority([a.composite_defect() for a in self.by_arrow(arrow_id)])

    def arm_defect_bounds(self, arrow_id: str) -> Tuple[float, float]:
        """Round-9 Patch Set 5: [lower, upper] arm defect, missing replicates
        propagated through the vote rather than discarded before it."""
        return majority_bounds([a.composite_defect()
                                for a in self.by_arrow(arrow_id)])

    def arrow_canonical_correct(self, arrow_id: str) -> Optional[bool]:
        return _majority_bool([a.canonical_correct()
                               for a in self.by_arrow(arrow_id)])

    def arrow_transformed_correct(self, arrow_id: str) -> Optional[bool]:
        """Charged correctness (Patch Set 4): refusals and parse failures count
        as incorrect, so a wholly refused arrow is a failure, not a dropout."""
        return _majority_bool([a.transformed_correct()
                               for a in self.by_arrow(arrow_id)])

    def arrow_canonical_correct_charged(self, arrow_id: str) -> Optional[bool]:
        return _majority_bool([a.canonical_correct_charged()
                               for a in self.by_arrow(arrow_id)])

    # ---- level 2: arrow -> family ------------------------------------------
    def structural_defect(self) -> Optional[float]:
        v = [self.arm_defect(t)[0] for t in ("T1", "T2")]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    def surface_defect(self) -> Optional[float]:
        return self.arm_defect("s")[0]

    def difference(self) -> Optional[float]:
        a, b = self.structural_defect(), self.surface_defect()
        return None if (a is None or b is None) else a - b

    def difference_bounds(self) -> Tuple[float, float]:
        """Directional family bounds built from ARM intervals (Patch Set 5):
        structural low with surface high gives the lower endpoint, and the
        reverse gives the upper."""
        s_lo = sum(self.arm_defect_bounds(t)[0] for t in ("T1", "T2")) / 2.0
        s_hi = sum(self.arm_defect_bounds(t)[1] for t in ("T1", "T2")) / 2.0
        u_lo, u_hi = self.arm_defect_bounds("s")
        return s_lo - u_hi, s_hi - u_lo

    def orbit_robust_bounds(self) -> Tuple[float, float]:
        """A_orb interval over {id, T1, T2}: each presentation reduced by the
        replicate rule with missing slots completed both ways."""
        parts = [_bool_bounds([a.canonical_correct_charged() for a in self.arrows
                               if a.arm_kind == "structural"],
                              n_slots=2 * R_REPLICATES)]
        for t in ("T1", "T2"):
            parts.append(_bool_bounds([a.transformed_correct()
                                       for a in self.by_arrow(t)]))
        lo = 1.0 if all(p[0] == 1.0 for p in parts) else 0.0
        hi = 1.0 if all(p[1] == 1.0 for p in parts) else 0.0
        return lo, hi

    def orbit_robust(self) -> Optional[float]:
        """A_orb at the FAMILY level over {id, T1, T2}, each presentation
        reduced by the declared replicate rule first (round-8 B08)."""
        canon = _majority_bool(
            [a.canonical_correct_charged() for a in self.arrows
             if a.arm_kind == "structural"])
        vals = [canon]
        for t in ("T1", "T2"):
            vals.append(self.arrow_transformed_correct(t))
        if any(v is None for v in vals):
            return None
        return 1.0 if all(vals) else 0.0


# ------------------------------------------------------------ metric suite --
def _bool_bounds(xs: Sequence[Optional[bool]],
                 n_slots: int = R_REPLICATES) -> Tuple[float, float]:
    """Correctness bounds for an arrow: missing replicates completed with
    incorrect and with correct, voted through the majority rule."""
    obs = [1.0 if x else 0.0 for x in xs if x is not None]
    missing = max(0, n_slots - len(obs))
    s = sum(obs)
    lo = 1.0 if 2 * s > n_slots else 0.0
    hi = 1.0 if 2 * (s + missing) > n_slots else 0.0
    return lo, hi


def metric_suite(fams: Sequence[ObservedFamily]) -> Dict[str, object]:
    """Co-reports at their DECLARED aggregation levels.

    Round-10: apparatus loss no longer disappears from the co-metrics. Each of
    D^u, A_G and A_orb is reported as an INTERVAL obtained by completing missing
    replicates with both outcomes, together with a complete-case point value
    that is explicitly labelled as such and restricted to identified arrows.
    An arrow with one observed success and three lost replicates is therefore
    partially identified everywhere, not a definite success in the co-metrics
    and an interval in the endpoint."""
    d_lo, d_hi, d_pt = [], [], []
    a_lo, a_hi, a_pt = [], [], []
    ties = {"structural": [0, 0], "surface": [0, 0]}
    labels: Dict[str, int] = {}
    by_class: Dict[str, Dict[str, int]] = {}
    jp_num = jp_den = 0
    cond_pt = []

    for f in fams:
        for aid in REQUIRED_ARROWS:
            kind = ARM_KIND_OF[aid]
            _d, tie = f.arm_defect(aid)
            ties[kind][1] += 1
            ties[kind][0] += int(tie)
            if kind != "structural":
                continue
            lo, hi = f.arm_defect_bounds(aid)
            d_lo.append(lo); d_hi.append(hi)
            if lo == hi:                                   # identified
                d_pt.append(lo)
                if f.arrow_canonical_correct_charged(aid) is True:
                    cond_pt.append(lo)
            alo, ahi = _bool_bounds([a.transformed_correct()
                                     for a in f.by_arrow(aid)])
            a_lo.append(alo); a_hi.append(ahi)
            if alo == ahi:
                a_pt.append(alo)
        for a in f.arrows:
            lab = a.failure_label()
            labels[lab] = labels.get(lab, 0) + 1
            by_class.setdefault(a.arm_kind, {})
            by_class[a.arm_kind][lab] = by_class[a.arm_kind].get(lab, 0) + 1
            jp_den += 1
            jp_num += int(a.joint_parse())

    orb_lo, orb_hi, orb_pt = [], [], []
    for f in fams:
        lo, hi = f.orbit_robust_bounds()
        orb_lo.append(lo); orb_hi.append(hi)
        if lo == hi:
            orb_pt.append(lo)

    def _m(xs):
        return (sum(xs) / len(xs)) if xs else None

    return {
        "D_u_bounds": (_m(d_lo), _m(d_hi)),
        "A_G_bounds": (_m(a_lo), _m(a_hi)),
        "A_orb_bounds": (_m(orb_lo), _m(orb_hi)),
        "D_u_complete_case": _m(d_pt), "D_t_complete_case": _m(cond_pt),
        "A_G_complete_case": _m(a_pt), "A_orb_complete_case": _m(orb_pt),
        "n_identified_arrows": len(d_pt), "n_arrow_units": len(d_lo),
        "aggregation": {
            "D_u": "arrow-level interval; missing replicates voted through",
            "D_t": "complete-case only, on identified arrows with canonical correct",
            "A_G": "arrow-level interval; behavioural non-response charged as 0",
            "A_orb": "family-level interval over {id, T1, T2}",
            "point_values": "complete-case: identified arrows/families only"},
        "joint_parse_rate": (jp_num / jp_den) if jp_den else None,
        "failure_counts": labels, "failure_counts_by_arm_class": by_class,
        "tie_rate_by_arm_class": {k: (v[0] / v[1]) if v[1] else None
                                  for k, v in ties.items()},
        "n_families": len(fams),
    }


def yield_rates(fams: Sequence[ObservedFamily]) -> Dict[str, Tuple[float, float]]:
    """Response and typed-parse rates per arm class, over ADMINISTERED responses.

    Round-10 blocker 3. The previous denominator excluded apparatus-lost sides,
    so a family with one observed and three lost replicates per arrow reported
    a response rate of 1.0 while only 25% of administered responses existed.
    Unresolved apparatus failure now LOWERS the response rate, which is what the
    paper's 'administered-arm response rate' means."""
    tally: Dict[str, List[int]] = {}
    for f in fams:
        for a in f.arrows:
            t = tally.setdefault(a.arm_kind, [0, 0, 0])
            for side in (a.canonical, a.transformed):
                t[0] += 1                                  # ADMINISTERED
                t[1] += int(side.status.observed
                            and side.status is not Status.REFUSAL)
                t[2] += int(side.parsed())
    return {k: ((v[1] / v[0]) if v[0] else 0.0, (v[2] / v[0]) if v[0] else 0.0)
            for k, v in tally.items()}


def yield_detail(fams: Sequence[ObservedFamily]) -> Dict[str, Dict[str, float]]:
    """Administered / observed / responded / parsed counts per arm class, so a
    reader can see the denominator rather than infer it."""
    out: Dict[str, Dict[str, float]] = {}
    for f in fams:
        for a in f.arrows:
            d = out.setdefault(a.arm_kind, {"administered": 0, "apparatus_lost": 0,
                                            "responded": 0, "parsed": 0})
            for side in (a.canonical, a.transformed):
                d["administered"] += 1
                if not side.status.observed:
                    d["apparatus_lost"] += 1
                    continue
                d["responded"] += int(side.status is not Status.REFUSAL)
                d["parsed"] += int(side.parsed())
    for d in out.values():
        n = d["administered"]
        d["response_rate"] = d["responded"] / n if n else 0.0
        d["parse_rate"] = d["parsed"] / n if n else 0.0
        d["apparatus_loss_rate"] = d["apparatus_lost"] / n if n else 0.0
    return out
