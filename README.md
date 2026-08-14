# IMR ancillary bundle

Reference implementation accompanying

> **What LLM Mathematics Benchmarks Measure: Construct Validity, Artifact
> Projection, and Certified Semantic Transport**
> Shubham Kumar (JGU Mainz; UMGC Europe)

This directory is the complete released artifact set. It contains **no model
outputs** and produces **no empirical claim about any language model**. It
implements the measurement design of the paper so that the design can be
inspected, executed, and criticized. Start with `RUNNING.md`.

## File map

| File | Role |
|---|---|
| `core.py` | Measurement core: parse-fail sentinel, transforms and their composition law, typed answer transport, distances, gold rendering, two-stage failure router `route_failure`. |
| `families.py` | The 24 certified families: canonical values, transforms with declared transport, negative controls with anti-certificates, certificate rungs, answer types, parser aliases. |
| `admission.json` | The frozen, schema-validated (`imr-admission/2`) admission manifest keyed by family ID, carrying the **frozen six-module universe** and the required arm/replicate design: certificate rung, N-i control status, surface matching, non-identity transport, and confirmatory eligibility with reasons. `run_confirmatory_analysis` REQUIRES it and fails closed when it is absent. Every family in the calibration bank is recorded ineligible for confirmatory use, which is the honest status of the bank. |
| `proof_summary.py` | The Section 6.3 proof-summary layer: obligation vectors, the invariant/presentation-specific mask partition, `lambda_T` as a bijection on invariant coordinates, the three functor laws, and `delta_P` with its UNAVAILABLE and adjudication cases. The bank is final-answer only, so this layer is exercised by tests. |
| `prompts/` | The verbatim system prompt and format-repair prompt named by the generation protocol, plus the seed and context rules. |
| `certificate_registry.py` | **The package-pinned, sealed certificate registry.** Production typed transports are package-owned symbolic maps: the registry holds the certified expressions, generates the executable forward and inverse implementations internally, and verifies the inverse identities symbolically on the declared answer types. Confirmatory mode accepts no caller-supplied executable transport function, and anti-equivalence certificates are resolved, digest-checked, family-bound and replayed here. Duplicate ids are refused; the registry is sealed at import and cannot be mutated, replaced or supplied by a caller. |
| `observation.py` | **The paired IMR observation.** `ArrowResponse` carries BOTH sides of every administered arrow -- presentation ids, arrow id, certificate digest, typed normalizer outputs, canonical and transported golds, per-side typed `Status` -- and every quantity (JP(T), canonical correctness, transformed accuracy, transport discrepancy, composite defect, failure label, D^u/D^t/A_G/A_orb, mechanism and tie counts) is DERIVED from it, never supplied. |
| `analysis.py` | The Section 9 pipeline. Entry points: **`run_confirmatory_analysis`** (the only confirmatory route; loads the package-pinned manifest, no caller override) and `run_exploratory_analysis` (permissive, never returns a bare `admissible`). `_run_legacy_scalar_analysis` is **legacy/exploratory scalar**, is private, and is not a confirmatory entry point. The confirmatory route enforces the frozen `Admission` manifest before weights are computed, decides on the LOWER partial-identification endpoint, and recomputes the threshold curve and sensitivity with whichever engine decided. Composite defect; declared strata and empty-stratum failure; fixed pre-run weights; stratum-weighted estimate and family-level variance; percentile-t bound with exact degenerate-replicate handling; constrained stratified null-imposed wild-cluster fallback; directional armwise Manski bounds and the target-population mismatch diagnostic; threshold curve; family-cluster evaluator gate and finite-bank census; yield gate; differential-evaluator sensitivity; joint max-t with FWER critical value; simulation DGP with separate apparatus/refusal/parse mechanisms and `solve_beta_for_endpoint` (targets the realized directional lower endpoint) for target excess. Synthetic data only. |
| `verify_all.py` | Independent symbolic re-derivation of the 20 L2 reference values; consistency check of the 4 Module-D convention-ledger values. |
| `materialize_families.py` | Writes `families_repo/`: one folder per family with `problem.json`, `transform.json`, `certificate.sympy`; validates schema and bank balance. |
| `dry_audit.py` | The eight admission invariants over all 24 families, including negative-control rejection and anti-certificate replay. |
| `freeze_manifest.py` | Enumerates the intended released file set and writes `MANIFEST.sha256` over it, excluding itself. |
| `parser_aliases.py` | Alias table used by the normalizer for typed answer extraction. |
| `test_corpus_invariants.py` | tests: admission invariants, composition law, parser realism and non-coercion, anti-certificate replay, router branches, metric bounds, the analysis pipeline, release integrity, six adversarial tests for the statistical layer, and the round-4 tests for admission enforcement, gate simultaneity, per-arm-class yield, seed derivation, majority reduction, and common support. |
| `IMR_*_certificate.py`, `run_all_sympy_certificates.py` | Six standalone symbolic certificates for the designed non-identity-transport families (not yet in the bank; threat T2). |
| `problem.json`, `transform.json` | Schema exemplars for one family record. |
| `released_bank_records.json` | The 24 families actually in the released bank. |
| `planned_family_records.json` | Designed flagship families **not** in the bank (threat T2); design registry only. |
| `family_records.json` | Legacy registry. It contains only the six **designed flagship** records, never the released bank; earlier documentation described it as a bank snapshot, which was wrong. Use the two files above. |
| `families_repo/` | 24 materialized family folders (72 files). |
| `MANIFEST.sha256` | SHA-256 of every other released file. Root of trust. |
| `requirements.txt`, `RUNNING.md` | Dependency floor; reproduction sequence and expected output. |

## Paper-code contract

Six points where paper and code must agree, and do. Each is pinned by a test.

1. **Failure precedence.** `core.STAGE1` is
   `["refusal", "parse", "type", "canonical", "transport", "mechanism", "witness"]`,
   where `type` means **answer-type** failure, checked before canonical
   correctness because canonical equality is undefined on a wrongly typed
   answer. The paper uses the same order and names the obstruction-class
   judgment separately, as an obligation-vector coordinate rather than a router
   label.
2. **Decision constants.** `DELTA_SEM = 0.05`, `EVAL_GATE = 0.05`,
   `RESPONSE_FLOOR = 0.90`, `PARSE_FLOOR = 0.80`, `B_PRIM/B_WILD/B_CURVE/B_SENS = 20000/4000/4000/2000`,
   `BOOT_SEED = 20240517`.
3. **Evaluator-gate sampling unit and level.** The **family**, not the gold
   pair, and the gate is a maximum over modules, so the per-module bounds are
   Bonferroni-corrected. `required_cluster_count()` returns **94 families per
   module**; the pilot has four, where the simultaneous all-zero bound is
   about 0.698. The gate is unattainable at pilot scale: 94 vs 4 families per module
   is a 23.5-fold shortfall, about 1.4 orders of magnitude. The paper says so; the tests pin it.
4. **Exact evaluator interval.** `family_cluster_upper` uses the exact
   one-sided Clopper-Pearson limit at **every** count, not only at zero. At
   n = 94, k = 1, alpha = 0.05/6 it returns 0.0706 and fails the gate; a
   percentile bootstrap on the same data returns about 0.043 and would pass.
   A test pins this.
5. **Gate order.** `run_confirmatory_analysis` runs admission, strata, evaluator and
   yield gates BEFORE any estimation, and returns `estimation_performed: False`
   with no endpoint keys when a gate fails. Yield is derived from the immutable
   replicate records; a supplied `yield_rates` argument cannot override them.
6. **Directional bounds.** Because the surface term enters with a minus sign,
   sharp bounds are formed armwise: lower sets unobserved structural arms to 0
   and unobserved surface arms to 1, upper reverses both. A single constant on
   every missing arm is not a bound, and a test asserts that it is not.

## Fail-closed identity (round 6)

Admission is a **positive** test: a family is admitted only if its ID occurs in
the frozen manifest, its module matches the recorded module, and its record
carries `confirmatory_eligible: true`. Unknown IDs, duplicate IDs and module
mismatches raise `ProtocolError`. The Bonferroni denominator comes from the
manifest's frozen six-module universe, never from the observed input. The
confirmatory path additionally validates the experimental unit -- two
structural arms, one surface arm, four complete replicate records per arm,
binary finite defects -- and refuses the legacy scalar arm, which survives only
as a smoke-test adapter behind `Admission(permissive=True)`.

The evaluator gate consumes `EvaluatorAuditRecord`s keyed by family and arrow,
with provenance and Boolean outcomes; anonymous per-module vectors are refused
on the confirmatory path. Non-finite or out-of-domain audit values raise rather
than being coerced to successes.

## Round-8 contract points

* **Total transport.** `observation.Transport` is a total map on a declared
  answer type with an inverse, version, types and certificate digest. The
  discrepancy applies it to every normalized value, not only to the gold. A
  partial map is refused.
* **Record validation and joining.** `ArrowResponse` and `ObservedFamily`
  validate at construction (arrow id, replicate range and completeness, module
  and family agreement, non-empty provenance, certificate/transport digest
  equality). The confirmatory path additionally joins evaluator records to
  administered arrows on (family, arrow) and requires certificate, judge,
  parser and renderer equality.
* **N-i control is a gate.** A missing or non-rejected negative control makes
  the run inadmissible before the evaluator and yield gates.
* **Answer typing.** `Status.ANSWER_TYPE` routes a wrong-typed answer to
  `F_answer-type`, not `F_canonical`. Mechanism and witness are obligation
  labels and are not derivable from a final-answer record; the scope table
  says so.
* **Aggregation levels.** replicate -> arrow (conservative majority) -> family
  -> module -> model. `D_u`, `D_t`, `A_G` are arrow-level; `A_orb` is a
  family-level minimum over {id, T1, T2}.
* **Grid scope.** `run_grid` calibrates the inference engine on paired records
  and does **not** exercise the fail-closed gates; the confirmatory regression
  tests do that.

## Deliberately not implemented

Labelled as analysis plans in the paper, not as released procedures: the
latent-annotator (Dawid-Skene) aggregation of human obligation records, and the
H2/H3 exploratory comparisons, which require a common dataset and an extended
family schema the released bank does not have.

## Scope

Module D families carry V1 convention-ledger certificates and are excluded from
the confirmatory estimand. Negative controls are predominantly false-claim
(type N-ii), valid for the obstruction track but not for the confirmatory
anti-equivalence role. All but one family have identity answer transport, so
the typed-transport machinery is barely exercised at this scale. These limits
are stated in the paper (threats T1-T4, Appendix B) and are not worked around
here.
