"""IMR-AM-005: Fujikawa chiral rotation and affine theta transport.
Executable SymPy certificate for theta -> theta + 2*Nf*alpha, composition, and negative.
"""
from sympy import symbols, pi, simplify


def main() -> None:
    alpha1, alpha2, theta, Nf = symbols("alpha1 alpha2 theta N_f")

    def shift(alpha):
        return 2 * Nf * alpha

    def tau(th, alpha):
        return th + shift(alpha)

    # Composition law for affine theta shifts.
    chained = tau(tau(theta, alpha1), alpha2)
    composite = tau(theta, alpha1 + alpha2)
    assert simplify(chained - composite) == 0

    # T1: Nf=1, alpha=pi/2 => theta -> theta + pi.
    assert shift(pi / 2).subs(Nf, 1) == pi

    # T2: Nf=1, alpha=pi/4 => theta -> theta + pi/2.
    assert shift(pi / 4).subs(Nf, 1) == pi / 2

    # Composite: pi + pi/2 = 3*pi/2.
    assert simplify((shift(pi / 2) + shift(pi / 4)).subs(Nf, 1) - 3 * pi / 2) == 0

    # Negative: classical action-only reasoning reports Delta theta = 0.
    classical_delta = 0
    measure_delta_T1 = shift(pi / 2).subs(Nf, 1)
    assert classical_delta != measure_delta_T1

    print("IMR-AM-005 certificate PASS")


if __name__ == "__main__":
    main()
