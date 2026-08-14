"""IMR-LG-006: de Rham period on S^1 and the local-to-global fallacy.
Executable SymPy certificate for period invariance and negative refutation.
"""
from sympy import symbols, sin, cos, pi, Rational, integrate, Integer, simplify


def main() -> None:
    t = symbols("t", real=True)

    # Canonical angular form: dtheta -> coefficient 1, period 2*pi.
    canonical_period = integrate(1, (t, 0, 2 * pi))
    assert canonical_period == 2 * pi

    # T1: orientation-preserving reparametrization theta -> theta + (1/2) sin theta.
    # Pullback coefficient: 1 + (1/2) cos theta.
    T1_period = integrate(1 + Rational(1, 2) * cos(t), (t, 0, 2 * pi))
    assert simplify(T1_period - canonical_period) == 0

    # T2: add exact form d(cos theta) = -sin theta dtheta.
    T2_period = integrate(1 - sin(t), (t, 0, 2 * pi))
    assert simplify(T2_period - canonical_period) == 0

    # Composite: both changes together.
    composite_period = integrate(1 + Rational(1, 2) * cos(t) - sin(t), (t, 0, 2 * pi))
    assert simplify(composite_period - canonical_period) == 0

    # Negative: local exactness => global exactness would force period 0.
    negative_period = Integer(0)
    assert negative_period != canonical_period
    assert simplify(canonical_period - 2 * pi) == 0

    print("IMR-LG-006 certificate PASS")


if __name__ == "__main__":
    main()
