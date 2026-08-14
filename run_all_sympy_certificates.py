"""Run all six IMR flagship SymPy certificates."""
import importlib

MODULES = [
    "IMR_QD_005_certificate",
    "IMR_LG_006_certificate",
    "IMR_AN_005_certificate",
    "IMR_AM_005_certificate",
    "IMR_BV_005_certificate",
    "IMR_TN_005_certificate",
]


def main() -> None:
    for name in MODULES:
        mod = importlib.import_module(name)
        mod.main()
    print("ALL SIX FLAGSHIP SYMPY CERTIFICATES PASS")


if __name__ == "__main__":
    main()
