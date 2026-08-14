# DISPOSITION.md — Final Submission Package (Round 12)

**Paper:** *What LLM Mathematics Benchmarks Measure: Construct Validity, Artifact Projection, and Certified Semantic Transport*
**Author:** Shubham Kumar (JGU Mainz; UMGC Europe) · **Freeze date:** 31 July 2026
**Status:** Patches A–E implemented, ten required regression tests added, §12 sweep clean, full release suite green from a clean extraction. **51 pages**, 42 references, **105 pinned files**, **302 tests**.

---

## 1. What changed

The remaining defect was exactly as stated: *the production registry could certify a typed transport that was not issued by it and did not faithfully implement the registered map.* Four things made that possible, and all four are closed.

**A finite sample cannot certify a program.** Your `BAD = 123456789` branch is symbolically invisible and outside any bounded region, and it breaks the inverse law there. Rather than widen the sample — which can never close the gap — the registry no longer accepts callables at all. Production transports are a **closed enum of package-owned symbolic specifications**; the executable forward and inverse maps are **generated internally** from those expressions; and the inverse identities are verified **symbolically** on the declared type. `register_symbolic_transport` has no `tau` parameter, so the attack has no surface.

**Certification is now a capability, not an attribute.** Each entry holds a private token attached only inside the resolver; the public constructor refuses one. `is_certified_transport` checks `type(t) is Transport`, token identity, callable and predicate **identity**, types and digest. Forged subclasses, copied public metadata, and replaced `tau` or `tau_inverse` all fail.

**The two digests are separate.** `transport_certificate_digest` is registry-level and immutable — the resolver accepts only `expected_transport_certificate_digest` for equality checking — and `arrow_certificate_digest` is supplied per arrow.

**The sealed registry is immutable.** `seal()` installs mapping proxies and `__setattr__` blocks reassignment of the sealed attributes.

## 2. Final hashes (§17)

| File | SHA-256 |
|---|---|
| `imr_critique.pdf` | `c89851fbe0b06fa7787aea541a755105c5ca5fb0e1b1518f24131aef43eea7d7` |
| `imr_critique.tex` | `4c011d6788b14b60a44bcc663ed2230d611096456c6f0d376c0f3d9b9b34f2eb` |
| `imr_critique.ind` | `77f8abc3a88ef8cad3fd06b8bbdecc70d3ba41e668fcb14fd80c1245f8583384` |
| `arxiv-submission.tar.gz` | `cda0f04d03f89c25b5500411661e40ced7033b6a4ccdd5ae98ae54b9dbcea98a` |
| `arxiv_metadata_abstract.tex` | `343edc19a79ef8030985f2dbb749b9ec1642c84deb9e3c165f4370174db2d3c1` |
| `BUILD.md` | `6660fccfb3340982fecffeb3d77809bb65505bbad3b9a524a0e790ad682cee72` |
| `AUDIT_DISPOSITION.md` | `fbe5d80c73a94eb3cda0b86fb062b2d5869ee851d4bef29f444f65e7b5d32882` |
| `anc/analysis.py` | `1feeb58ef6864fa67e9732f23f019b113e3ac2f5342abe3ec1f6273f69f3b591` |
| `anc/observation.py` | `29d17d7649c4a12dc7346fc912d22b570df7825a425fbef747793c25441972e7` |
| `anc/certificate_registry.py` | `0c8dee6129140635841a971e09424e3354dc6a5895fe5838a8cf5c7173592787` |
| `anc/admission.json` | `72b365a4d43306f5da2a65d5dd70325bab9ec7ca5ce1c04551c1df3604613730` |
| `anc/MANIFEST.sha256` | `7e54d898ff1a954adb4a43393ff59f9cbe50bdb34f3743058d73f2e752c209a4` |

**No file may be edited after this point.** Any edit invalidates the manifest and every value above.

## 3. Clean-extraction acceptance (§16)

`errors=0 overfull=0 underfull=0 undefined=0 pages=51` · Type 3 fonts **none** · rebuilt `.ind` **byte-identical** · `verify_all` · `materialize_families` · `dry_audit` · `run_all_sympy_certificates` · `analysis.py` · `pytest -q` **302 passed** · `compileall` · `sha256sum -c` **105/105 OK**.

## 4. One deviation, stated up front

**The paper is 51 pages, not the 50 in §14.** The §10/§11 wording that this patch requires added a paragraph, and the index now spills to a second page. I tightened that paragraph once and it did not recover page 50; I did not delete real content to force the number. The page count is a regression check against structural damage, and the patch's own scope statement sanctions wording needed to state the implementation honestly. Every other acceptance value is exactly as specified, and all records say 51.

## 5. Upload

1. Upload **`arxiv-submission.tar.gz`** — the archive was deleted and rebuilt from the corrected frozen tree, not renamed. If your browser appends a number, rename to exactly that filename.
2. Paste title, authors, comments and the 1,879-character ASCII abstract from `arxiv_metadata_abstract.tex`. Categories **cs.AI** primary, **cs.CL** cross-list.
3. Compile on arXiv; page through all **51**; confirm `anc/` is listed. License; submit.

## 6. Still yours, twelve rounds on

**Confirm the official affiliation strings and use a durable corresponding email.** It remains the only item in this package that I cannot verify, and a student address may lapse before the paper is read. Do it before you upload.

## 7. Closing

The principle this round enforced is the right one and worth stating in the paper's own terms: a confirmatory typed transport must not merely *claim* correspondence to a registry entry — its executable semantics must be *supplied* by that registry, and the runtime must verify the exact issued object rather than trusting caller-controlled metadata. That is now true of the code.

What has not changed, and is stated in the paper: the bank encodes identity transport, every family is confirmatory-ineligible, the anti-certificate namespace ships empty, and no model has been run. The instrument is sound and honest about its own limits; the empirical result is the next paper.
