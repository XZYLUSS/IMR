"""
imr/freeze_manifest.py  --  pin the frozen file set by SHA-256 (m-4).
Writes MANIFEST.sha256 over all source files and materialized family artifacts.
Verify with:   sha256sum -c MANIFEST.sha256   (from the imr/ directory)
"""
import hashlib, os, glob

ROOT = os.path.dirname(__file__)
SELF = "MANIFEST.sha256"

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# The intended released file set: every regular file under ROOT except the
# manifest itself and transient build artifacts. Defining it by enumeration of
# the directory (rather than by a hand-maintained list) is what makes the
# manifest a complete root of trust for the released bundle.
EXCLUDE_NAMES = {SELF}
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".log", ".tmp")

def intended_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn in EXCLUDE_NAMES or fn.endswith(EXCLUDE_SUFFIX):
                continue
            out.append(os.path.relpath(os.path.join(dirpath, fn), ROOT))
    return sorted(out)

def frozen_files():
    return intended_files()

def main():
    lines = []
    for rel in frozen_files():
        if rel == SELF:
            continue
        lines.append(f"{_sha256(os.path.join(ROOT, rel))}  {rel}")
    out = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, SELF), "w") as f:
        f.write(out)
    print(f"wrote {SELF}: {len(lines)} files pinned by SHA-256")
    # show a few representative digests
    for ln in lines[:3] + lines[-2:]:
        print("  ", ln)

if __name__ == "__main__":
    main()
