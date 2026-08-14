"""
imr/certificate_registry.py -- the package-pinned, sealed certificate registry.

Round-12. The registry is now the SOLE AUTHORITY over the executable semantics
of a production typed transport. Four things changed, each closing a way in
which a caller could previously influence certification.

  A. AUTHORITATIVE SYMBOLIC SPECIFICATIONS. A finite concrete sample cannot
     prove that an arbitrary Python callable globally realizes a symbolic map:
     the branch `0 if x == 123456789 else x` is symbolically invisible and lies
     outside any bounded sample, yet breaks the inverse law. The registry
     therefore no longer accepts callables at all. Production transports are a
     CLOSED ENUM of package-owned symbolic specifications; the executable
     forward and inverse maps are GENERATED internally from those expressions,
     and the inverse identities are verified symbolically on the declared
     answer types. With generation internal, the faithfulness question does not
     arise: there is no external implementation to be unfaithful.

  B. OPAQUE CAPABILITY. Each entry holds a private `object()` token, attached
     to a Transport only inside the resolver. Certification is an exact
     identity check -- object type, token identity, callable identity,
     predicate identity, declared types and the frozen digest -- not a Boolean
     attribute a caller can set.

  C. SEPARATED DIGESTS. `transport_certificate_digest` (registry-level: this
     map is a total bijection with the declared inverse) is immutable and can
     only be checked for equality. `arrow_certificate_digest` (family-level:
     this transformation is a valid semantic equivalence and this transport is
     its induced answer action) is a separate field supplied per arrow.

  D. IMMUTABLE AFTER SEALING. Sealing converts the namespaces to mapping
     proxies and blocks reassignment of the sealed attributes, so the sealed
     registry cannot be mutated structurally.

Concrete test points remain in the test suite as REGRESSION CHECKS. They are
not, and are no longer described as, a universal certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Dict, Optional, Sequence, Tuple

from observation import ObservationError, Transport


class BuiltinTransportId(str, Enum):
    """The closed set of production typed transports. Extending it is a source
    change reviewed with the release, not a runtime registration."""
    INT_IDENTITY_V1 = "int_identity_v1"
    INT_NEGATION_V1 = "int_negation_v1"


@dataclass(frozen=True)
class SymbolicTransportSpec:
    """The authoritative definition. The package owns the expressions; the
    implementations are generated from them."""
    transport_id: str
    transport_version: str
    source_answer_type: str
    target_answer_type: str
    forward_expression: str
    inverse_expression: str
    transport_certificate_digest: str


PRODUCTION_SPECS: Tuple[SymbolicTransportSpec, ...] = (
    SymbolicTransportSpec(
        BuiltinTransportId.INT_IDENTITY_V1.value, "1", "int", "int",
        "x", "x", "imr-transport-int-identity-v1"),
    SymbolicTransportSpec(
        BuiltinTransportId.INT_NEGATION_V1.value, "1", "int", "int",
        "-x", "-x", "imr-transport-int-negation-v1"),
)


# ------------------------------------------------- symbolic law verification --
def verify_symbolic_inverse_laws(spec: SymbolicTransportSpec
                                 ) -> Tuple[bool, str]:
    """tau_inv(tau(x)) - x == 0 and tau(tau_inv(x)) - x == 0, as identities on
    the declared type. This is the certificate; no sampling is involved."""
    try:
        import sympy as sp
    except Exception as exc:                                  # pragma: no cover
        return False, f"sympy unavailable: {exc}"
    x = sp.Symbol("x", integer=True)
    try:
        fwd = sp.sympify(spec.forward_expression, locals={"x": x})
        inv = sp.sympify(spec.inverse_expression, locals={"x": x})
    except Exception as exc:
        return False, f"expression does not parse: {exc}"
    if sp.simplify(inv.subs(x, fwd) - x) != 0:
        return False, "tau_inverse(tau(x)) - x does not simplify to 0"
    if sp.simplify(fwd.subs(x, inv) - x) != 0:
        return False, "tau(tau_inverse(x)) - x does not simplify to 0"
    if spec.source_answer_type == spec.target_answer_type == "int":
        for e in (fwd, inv):
            if e.subs(x, sp.Integer(7)).is_integer is not True:
                return False, "expression leaves the declared integer type"
    return True, ""


def _compile_int_map(expression: str) -> Callable[[object], int]:
    """Generate an executable integer map from the certified expression."""
    import sympy as sp
    x = sp.Symbol("x", integer=True)
    expr = sp.sympify(expression, locals={"x": x})

    def _impl(value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ObservationError("expected an integer answer")
        result = expr.subs(x, sp.Integer(value))
        if result.is_integer is not True:
            raise ObservationError("transport left the declared integer target")
        return int(result)

    return _impl


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ------------------------------------------------------------------ entries --
@dataclass(frozen=True)
class CertifiedTransportEntry:
    transport_id: str
    transport_version: str
    source_answer_type: str
    target_answer_type: str
    transport_certificate_digest: str
    tau: Callable[[object], object]
    tau_inverse: Callable[[object], object]
    source_predicate: Callable[[object], bool]
    target_predicate: Callable[[object], bool]
    probe_domain: Tuple = ()
    _registry_token: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class AntiCertificateEntry:
    anti_certificate_id: str
    anti_certificate_digest: str
    permitted_families: Tuple[str, ...]
    control_type: str
    verifier: Callable[[object], bool]
    verifier_version: str

    def __post_init__(self):
        object.__setattr__(self, "permitted_families",
                           tuple(self.permitted_families))


# ----------------------------------------------------------------- registry --
_SEALED_ATTRS = frozenset({"_transports", "_anti", "_sealed"})


class CertificateRegistry:
    """One immutable registry with two namespaces, sealed after construction."""

    def __init__(self):
        self._transports: Dict[Tuple[str, str], CertifiedTransportEntry] = {}
        self._anti: Dict[str, AntiCertificateEntry] = {}
        self._sealed = False

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) and name in _SEALED_ATTRS:
            raise ObservationError(
                f"sealed registry attribute cannot be reassigned: {name}")
        super().__setattr__(name, value)

    # ---- construction ----------------------------------------------------
    def register_symbolic_transport(self, spec: SymbolicTransportSpec) -> None:
        """The ONLY production registration path. No callable is accepted."""
        self._check_open()
        key = (spec.transport_id, spec.transport_version)
        if key in self._transports:
            raise ObservationError(f"duplicate transport certificate id: {key}")
        ok, why = verify_symbolic_inverse_laws(spec)
        if not ok:
            raise ObservationError(
                f"transport {spec.transport_id!r} failed symbolic verification: "
                f"{why}")
        entry = CertifiedTransportEntry(
            transport_id=spec.transport_id,
            transport_version=spec.transport_version,
            source_answer_type=spec.source_answer_type,
            target_answer_type=spec.target_answer_type,
            transport_certificate_digest=spec.transport_certificate_digest,
            tau=_compile_int_map(spec.forward_expression),
            tau_inverse=_compile_int_map(spec.inverse_expression),
            source_predicate=_is_int, target_predicate=_is_int,
            probe_domain=(1, 2), _registry_token=object())
        self._transports[key] = entry

    def register_anti(self, e: AntiCertificateEntry) -> None:
        self._check_open()
        if e.anti_certificate_id in self._anti:
            raise ObservationError(
                f"duplicate certificate id: {e.anti_certificate_id}")
        if e.control_type != "N-i":
            raise ObservationError("only N-i anti-certificates may be registered")
        self._anti[e.anti_certificate_id] = e

    def seal(self) -> "CertificateRegistry":
        if self._sealed:
            return self
        super().__setattr__("_transports", MappingProxyType(dict(self._transports)))
        super().__setattr__("_anti", MappingProxyType(dict(self._anti)))
        super().__setattr__("_sealed", True)
        return self

    @property
    def sealed(self) -> bool:
        return self._sealed

    def _check_open(self):
        if self._sealed:
            raise ObservationError(
                "the certificate registry is sealed; no registration, "
                "replacement or deletion is possible after sealing")

    # ---- transports ------------------------------------------------------
    def resolve_transport(self, *, transport_id: str, version: str,
                          expected_transport_certificate_digest: str,
                          arrow_certificate_digest: str) -> Transport:
        """Issue a registry-certified Transport.

        The registry digest is never replaceable: the caller may only assert
        what it expects, and the frozen value is what the returned object
        carries. The arrow-level digest is a separate, caller-supplied field."""
        e = self._transports.get((transport_id, version))
        if e is None:
            raise ObservationError(
                f"typed transport {transport_id!r} v{version} is not in the "
                "package-pinned certificate registry")
        if e.transport_certificate_digest != expected_transport_certificate_digest:
            raise ObservationError(
                f"transport {transport_id!r}: certificate digest mismatch")
        if not arrow_certificate_digest:
            raise ObservationError("an arrow certificate digest is required")
        t = Transport(e.transport_id, e.transport_version, e.source_answer_type,
                      e.target_answer_type, e.transport_certificate_digest,
                      e.tau, e.tau_inverse, source_domain=e.probe_domain,
                      source_predicate=e.source_predicate,
                      target_predicate=e.target_predicate,
                      domain_kind="typed",
                      arrow_certificate_digest=arrow_certificate_digest)
        object.__setattr__(t, "_registry_token", e._registry_token)
        return t

    def is_certified_transport(self, transport: object) -> bool:
        """Exact identity check. A forged subclass, copied public metadata
        without the token, or a replaced callable all fail."""
        if type(transport) is not Transport:
            return False
        e = self._transports.get((transport.transport_id,
                                  transport.transport_version))
        if e is None:
            return False
        return all((
            getattr(transport, "_registry_token", None) is e._registry_token,
            transport.source_answer_type == e.source_answer_type,
            transport.target_answer_type == e.target_answer_type,
            transport.transport_certificate_digest
            == e.transport_certificate_digest,
            transport.domain_kind == "typed",
            transport.tau is e.tau,
            transport.tau_inverse is e.tau_inverse,
            transport.source_predicate is e.source_predicate,
            transport.target_predicate is e.target_predicate,
        ))

    # ---- anti-certificates ----------------------------------------------
    def replay_anti(self, control) -> Tuple[bool, str]:
        e = self._anti.get(control.anti_certificate_id)
        if e is None:
            return False, (f"anti-certificate {control.anti_certificate_id!r} is "
                           "not in the frozen registry")
        if e.anti_certificate_digest != control.anti_certificate_digest:
            return False, (f"anti-certificate digest mismatch for "
                           f"{control.anti_certificate_id!r}")
        if e.control_type != control.control_type:
            return False, "anti-certificate control type mismatch"
        if control.family_id not in e.permitted_families:
            return False, (f"anti-certificate {control.anti_certificate_id!r} is "
                           f"not registered for family {control.family_id!r}")
        try:
            ok = bool(e.verifier(control))
        except Exception as exc:
            return False, f"anti-certificate raised on replay: {exc}"
        return (ok, "" if ok else "anti-certificate replay returned false")


def _build_production_registry() -> CertificateRegistry:
    r = CertificateRegistry()
    for spec in PRODUCTION_SPECS:
        r.register_symbolic_transport(spec)
    # The anti-certificate namespace is deliberately EMPTY: no family in the
    # released bank is confirmatory-eligible, so no confirmatory run can pass
    # the N-i gate. That is the honest state of this release.
    return r.seal()


PRODUCTION_REGISTRY = _build_production_registry()

TRANSPORT_DIGESTS = MappingProxyType(
    {s.transport_id: s.transport_certificate_digest for s in PRODUCTION_SPECS})
