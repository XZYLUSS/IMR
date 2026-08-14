# Running the ancillary bundle

Reference implementation accompanying *What LLM Mathematics Benchmarks
Measure: Construct Validity, Artifact Projection, and Certified Semantic
Transport*. See `README.md` for the file map and the paper-code contract.

**Requirements.** Python 3.11 or later (verified on CPython 3.12.3), plus the
floors in `requirements.txt`: `sympy>=1.12`, `numpy>=1.24`, `pytest>=7.4`. No
network access is required.

## Reproduction sequence

Run in order, from this directory. No copying, symlinking, patching, or
undocumented step is required at any point.

```sh
python3 -m pip install -r requirements.txt

python3 verify_all.py            # symbolic re-derivation of the 20 V2 values;
                                 # consistency check of the 4 Module-D values
python3 materialize_families.py  # writes families_repo/ : 24 family folders
python3 dry_audit.py             # 8 admission invariants x 24 families
python3 freeze_manifest.py       # writes MANIFEST.sha256 over the released set
python3 analysis.py              # fast smoke demo, ~8 s
# python3 analysis.py --full-grid  # the DECLARED factorial calibration:
#                                 # 972 cells, 2000 reps, B_prim/B_wild/B_curve/B_sens = 20,000/4,000/4,000/2,000. This is a
#                                 # multi-day sequential batch job; resumable,
#                                 # parallel with workers>1, and writes
#                                 # calibration_grid.json incrementally.
python3 -m pytest test_corpus_invariants.py -q     # expected: 302 passed
```

## Expected output

| Step | Expected |
|---|---|
| `verify_all.py` | all ten base families and all fourteen expansion families verified |
| `materialize_families.py` | 24 folders; missing artifacts: NONE; schema valid; bank balanced 4-per-module |
| `dry_audit.py` | every family: negative control rejected, anti-certificate replayed |
| `freeze_manifest.py` | `105 files pinned by SHA-256` |
| `analysis.py` | gate order, exact Clopper-Pearson counterchecks, derived yield, partial-identification endpoints, admission manifest, smoke grid; **~8 s** |
| `pytest` | `302 passed` |

`freeze_manifest.py` rewrites `MANIFEST.sha256`; running it before
`materialize_families.py` will pin fewer files. The order above is intended.

## The primary analysis

`analysis.run_confirmatory_analysis(fams, ...)` is the single entry point that
Section 9 of the paper describes. In pre-specified order it checks the declared
strata, runs the evaluator and yield gates, computes the stratum-weighted point
estimate with fixed pre-run weights, computes the directional sharp bounds and
the target-population mismatch diagnostic, computes the percentile-t lower bound, applies the
fallback rule, and returns the decision together with the basis on which it was
made. A failed stratum check or gate returns INADMISSIBLE; no endpoint is then
reported as a result.

Two numbers worth checking yourself:

```sh
python3 -c "import analysis as a; print(a.required_cluster_count())"                       # 94 per module
python3 -c "import analysis as a; print(a.family_cluster_upper([0.0]*4, alpha=0.05/6))"    # ~0.698
python3 -c "import analysis as a; print(a.family_cluster_upper([1.0]+[0.0]*93, alpha=0.05/6))"  # ~0.0706, fails the gate
```

The sampling unit for the evaluator gate is the **family**, not the gold pair,
the audited quantity is a binary family-level failure indicator, and the gate is
a maximum over modules, so the per-module bounds are Bonferroni-corrected. At
zero observed failures the exact one-sided bound over N clusters at level a is
1 - a**(1/N); with a = 0.05/6 the gate needs 94 families PER MODULE. The pilot
has four, where the simultaneous bound is about 0.698, so the pilot is
calibration-only. This is stated in the paper and pinned by tests.

## Measured runtimes (this container; your hardware will differ)

| Command | This environment |
|---|---|
| `verify_all.py` | ~20 s |
| `materialize_families.py` | ~10 s |
| `dry_audit.py` | ~10 s |
| `analysis.py` (smoke) | ~8 s |
| `pytest -q` | ~5 s |
| `analysis.py --full-grid` | measured per-cell cost x 972 cells x reps; multi-day sequential, `workers>1` parallelizes. Run `benchmark_analysis()` for your hardware. |

## Not implemented here

Four operational layers are analysis plans rather than released procedures, and
are labelled as such in the paper: the latent-annotator (Dawid-Skene)
aggregation of human obligation records; a live evaluator audit of a renderer,
parser, human grader or LLM judge -- `core.kernel_gold_self_consistency` is
named for what it is, a comparison of frozen reference objects through the same
equality kernel that defined them; the H2/H3 comparisons; and the generation
harness, since no model is run anywhere in this work.

## Verifying the release

`MANIFEST.sha256` pins every other released file and does not pin itself.

```sh
sha256sum MANIFEST.sha256      # compare with Appendix B of the paper
sha256sum -c MANIFEST.sha256   # expected: 105 lines, all OK
```

`test_manifest_covers_every_released_file` asserts entry-set equality with the
intended released set minus itself; `test_manifest_digests_match_disk`
recomputes every digest.

## What this bundle does not contain

No model outputs, no empirical results, no pre-registration. The Section 9
protocol is *pre-specified*, not pre-registered; a public timestamped deposit
must precede any confirmatory run. Module D is excluded from the confirmatory
estimand, negative controls are predominantly false-claim (type N-ii), and all
but one family have identity answer transport -- threats T1-T4 in the paper.
