"""Corpus dry audit: run the eight invariants on all twenty-four families and tabulate.
Invariant 2 (composition) is a FAMILY-LEVEL genuine composition-law check (FB-2);
invariants 1,3,4 are collapsed across the two certified transforms.
"""
from families import build_corpus
C = build_corpus()
print(f"{'family':12} {'mod':4} {'rung':4}  1 2 3 4 5 6 7 8   D_eval  verdict")
print("-" * 64)
allpass = True
for fid in sorted(C):
    fam = C[fid]; rep = fam.audit(); ch = rep["checks"]
    # collapse per-transform laws (identity/inverse/gold_transport) across transforms
    coll = {k: all(ch[f"functorial[{T.tid}]"][k] for T in fam.transforms)
            for k in ["identity", "inverse", "gold_transport"]}
    row = [coll["identity"], ch["composition"], coll["inverse"], coll["gold_transport"],
           ch["parser_soundness"], ch["parser_non_coercion"],
           ch["neg_control_rejected"], ch["kernel_gold_self_consistency"]]
    marks = " ".join("Y" if b else "X" for b in row)
    mod = fid.split("-")[1]
    print(f"{fid:12} {mod:4} {fam.rung:4}  {marks}   {fam.kernel_gold_self_consistency():.3f}   "
          f"{'PASS' if rep['passed'] else 'FAIL'}")
    allpass = allpass and rep["passed"]
print("-" * 64)
print(f"corpus dry-audit: {'ALL '+str(len(C))+' FAMILIES PASS ALL EIGHT INVARIANTS' if allpass else 'FAILURES PRESENT'}")
print()
print("Negative-control rejection sweep (vacuous-invariance loophole):")
for fid in sorted(C):
    fam = C[fid]
    accepted = fam.eqf(fam.neg_control_answer, fam.canonical_gold)
    anti_ok = bool(fam.anti_certificate(fam.neg_control_answer, fam.canonical_gold))
    print(f"  {fid:12}: n(x) accepted-as-equivalent = {accepted}  anti-certificate replayed = {anti_ok}  -> "
          f"{'rejected (good)' if (not accepted and anti_ok) else 'LEAK'}")
