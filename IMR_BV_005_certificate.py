"""IMR-BV-005: total derivative shifts boundary momentum but not bulk EL equation.
Executable SymPy certificate for bulk invariance, boundary transport, composition, and negative.
"""
from sympy import symbols, Function, Rational, diff, simplify
from sympy.calculus.euler import euler_equations


def main() -> None:
    t = symbols("t")
    x = Function("x")
    V = Function("V")

    L = Rational(1, 2) * diff(x(t), t) ** 2 - V(x(t))

    # T1: add d/dt(x^2)
    L_T1 = L + diff(x(t) ** 2, t)

    # T2: add d/dt(x^3)
    L_T2 = L + diff(x(t) ** 3, t)

    # Composite: add d/dt(x^2+x^3)
    L_comp = L + diff(x(t) ** 2 + x(t) ** 3, t)

    # Bulk EL equations are unchanged.
    EL = euler_equations(L, x(t), t)
    assert euler_equations(L_T1, x(t), t) == EL
    assert euler_equations(L_T2, x(t), t) == EL
    assert euler_equations(L_comp, x(t), t) == EL

    v = diff(x(t), t)
    p = diff(L, v)
    p_T1 = diff(L_T1, v)
    p_T2 = diff(L_T2, v)
    p_comp = diff(L_comp, v)

    # Boundary momentum transports nontrivially.
    assert simplify(p_T1 - p - 2 * x(t)) == 0
    assert simplify(p_T2 - p - 3 * x(t) ** 2) == 0
    assert simplify(p_comp - p - (2 * x(t) + 3 * x(t) ** 2)) == 0

    # Negative: total derivative has no boundary effect.
    negative_shift = 0
    assert simplify((p_T1 - p) - negative_shift) != 0

    print("IMR-BV-005 certificate PASS")


if __name__ == "__main__":
    main()
