"""IMR-QD-005: SU(2) holonomy, gauge conjugation, Wilson invariant.
Executable SymPy certificate for transported golds, composition, and negative control.
"""
from sympy import Matrix, I, sqrt, Rational, eye, diag, trace, simplify, zeros


def main() -> None:
    sx = Matrix([[0, 1], [1, 0]])
    sy = Matrix([[0, -I], [I, 0]])
    s2 = 1 / sqrt(2)
    ident = eye(2)

    # U0 = diag(exp(i*pi/3), exp(-i*pi/3)) but kept exact algebraic.
    e = Rational(1, 2) + I * sqrt(3) / 2
    U0 = diag(e, e.conjugate())

    g1 = s2 * (ident + I * sx)
    g2 = s2 * (ident + I * sy)

    assert simplify(g1.H * g1 - ident) == zeros(2)
    assert simplify(g2.H * g2 - ident) == zeros(2)
    assert simplify(g1.det() - 1) == 0
    assert simplify(g2.det() - 1) == 0

    # T1 and T2 transports are conjugations, not identity.
    U1 = simplify(g1.inv() * U0 * g1)
    U2 = simplify(g2.inv() * U0 * g2)

    assert simplify(U1 - U0) != zeros(2)
    assert simplify(U2 - U0) != zeros(2)

    # Wilson invariant descends to the gauge orbit.
    assert simplify(trace(U0) - 1) == 0
    assert simplify(trace(U1) - trace(U0)) == 0
    assert simplify(trace(U2) - trace(U0)) == 0

    # Composition: tau_g2(tau_g1(U)) = tau_{g1*g2}(U).
    chained = simplify(g2.inv() * U1 * g2)
    composite = simplify((g1 * g2).inv() * U0 * (g1 * g2))
    assert simplify(chained - composite) == zeros(2)

    # Negative control: reporting U0 after T1 is wrong even though trace agrees.
    negative_untransported = U0
    assert simplify(U1 - negative_untransported) != zeros(2)
    assert simplify(trace(U1) - trace(negative_untransported)) == 0

    print("IMR-QD-005 certificate PASS")


if __name__ == "__main__":
    main()
