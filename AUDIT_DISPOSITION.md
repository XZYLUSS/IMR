# Disposition Record — Round 12 (registry authenticity, final)

Manuscript: **What LLM Mathematics Benchmarks Measure** — **51 pp.**, 42 references. Ancillary bundle: **105 pinned files**, **302 tests**.

Input: *IMR Final Typed-Transport Patch Set — Round 12*. Patches A–E, ten required regression tests, the §12 obsolete-API sweep, and the §18 freeze criteria. Every vulnerability in §2 was reproduced here before repair.

---

## A. The four vulnerabilities, reproduced then closed

| § | Reproduced | Repair |
|---|---|---|
| **2.1** Finite sampling is not global certification | ✅ `tau = 0 if x == 123456789 else x` passed `prove_int_bijection` while `tau_inverse(tau(BAD)) == 0` | **Patch A.** The callable-registration path is *deleted*. Production transports are a closed `BuiltinTransportId` enum of package-owned `SymbolicTransportSpec`s; `register_symbolic_transport` takes **no `tau` or `tau_inverse` parameter at all**. Executable maps are generated internally from the certified expressions via SymPy, and the inverse identities are verified **symbolically** (`simplify(inv.subs(x, fwd) - x) == 0` and its dual), not sampled. With generation internal there is no external implementation that could be unfaithful. |
| **2.2** Certification forgeable through attributes | ✅ a `Transport` subclass carrying copied fields was certified | **Patch B.** Each entry holds a private `object()` token, attached to a `Transport` only inside the resolver via `object.__setattr__`; the public constructor **raises** if a token is passed. `is_certified_transport` requires `type(t) is Transport` (not `isinstance`), token identity, `tau`/`tau_inverse`/predicate **identity**, declared types, digest and `domain_kind`. No Boolean marker or tuple key decides anything. |
| **2.3** Registered digest replaceable | ✅ `certificate_digest="ANY-CALLER-VALUE"` was returned on the issued object | **Patch C.** `certificate_digest` is split into **`transport_certificate_digest`** (registry-level, immutable, copied from the frozen entry) and **`arrow_certificate_digest`** (family-level, supplied per arrow). The resolver takes `expected_transport_certificate_digest` and uses it **only for equality checking**; a mismatch raises. |
| **2.4** Sealed registry structurally mutable | ✅ `PRODUCTION_REGISTRY._anti["EVIL"] = …` succeeded | **Patch D.** `seal()` converts both namespaces to `MappingProxyType`, and `__setattr__` blocks reassignment of `_transports`, `_anti`, `_sealed`. Entries are frozen dataclasses with tuple collections. |

**Patch E** — `_confirmatory_core` validates every typed transport with `PRODUCTION_REGISTRY.is_certified_transport(...)`, never from a serialized field, and the simulator obtains its transports by `resolve_transport`, so simulated confirmatory families carry registry-issued objects.

## B. Required regression tests (10/10)

§9.1 `test_r12_branch_outside_faithfulness_sample_is_rejected` — asserts the production API has no `tau`/`tau_inverse` parameter (the patch's preferred outcome) and that the generated negation map is correct at `BAD` · §9.2 forged subclass **and** copied-public-metadata cases · §9.3 replaced `tau` · §9.4 replaced `tau_inverse` · §9.5 digest override refused, `certificate_digest` absent from the signature · §9.6 the two digests distinct on an issued object · §9.7 mapping-proxy immutability including `del` · §9.8 sealed attribute reassignment · §9.9 forged `registry_certified` / `registry_key` markers grant nothing · §9.10 confirmatory families carry certified transports with matching arrow digests.

## C. §12 obsolete-API sweep

`registry_certified`, `registry_key`, `is_certified(`, `prove_int_bijection` and the `[-1024, 1024]` sample region appear **nowhere outside the regression tests that prove they carry no authority**. Remaining `faithfulness` hits are one prose line in `certificate_registry.py` explaining that the question no longer arises, and the §9.1 test name. No transport-digest override remains; no arbitrary production callable input remains; no mutable sealed mapping remains.

## D. Manuscript wording (§10, §11)

The scope-table row and the transport paragraph now read: production typed transports are package-owned symbolic maps whose executable implementations are generated internally and issued only through the sealed registry; the inverse identities are verified symbolically on the declared type; confirmatory mode accepts no caller-supplied executable transport function. The claim that a finite sample proves global agreement is **withdrawn**, and concrete points are described only as regression checks. The two digests are defined and distinguished in the text.

## E. Release suite (§13–§16), from a clean extraction

`verify_all` PASS · `materialize_families` 24 folders · `dry_audit` PASS · `run_all_sympy_certificates` PASS · `analysis.py` PASS · `pytest -q` **302 passed** · `compileall` PASS · `freeze_manifest` **105** · `sha256sum -c` **105/105 OK** · TeX errors 0, overfull 0, underfull 0, undefined 0 · Type 3 fonts **none** · rebuilt `.ind` **byte-identical** · archive recreated from the corrected frozen tree as `arxiv-submission.tar.gz`.

## F. One deviation from the patch plan, stated plainly

§14 lists `pages = 50`. **The paper is now 51 pages.** The §10/§11 wording the patch itself requires added a paragraph, and the index consequently spills onto a second page. I tightened that paragraph once to try to recover page 50; it did not, and I did not delete substantive content to force the count, because the page number is a regression check against accidental structural damage and this change is the sanctioned honest-wording case named in the patch's own scope statement. Every other acceptance value is exactly as required. All records here, in BUILD.md, in DISPOSITION.md and in the arXiv comments say **51**.

## G. Freeze criteria (§18)

All eighteen boxes are satisfied, with the single documented exception of the page count in §F.

## H. Unchanged

The released 24-family bank encodes identity transport, every family is confirmatory-ineligible, the production anti-certificate namespace is deliberately empty, and **no model has been run**. No confirmatory verdict about any model is claimed anywhere in this package.
