"""IMR-TN-005: SL(2,Z) electric-magnetic charge transport and DSZ invariant.
Executable SymPy certificate for non-identity charge transport, pairing invariance, and composition.
"""
from sympy import Matrix, zeros, simplify


def main() -> None:
    S = Matrix([[0, 1], [-1, 0]])
    T = Matrix([[1, 1], [0, 1]])
    J = Matrix([[0, 1], [-1, 0]])

    v = Matrix([2, 1])  # canonical dyon charge
    w = Matrix([1, 3])  # reference charge

    assert S.det() == 1
    assert T.det() == 1

    # SL(2,Z) in dimension 2 preserves the DSZ symplectic form.
    assert simplify(S.T * J * S - J) == zeros(2)
    assert simplify(T.T * J * T - J) == zeros(2)

    Sv = S * v
    Tv = T * v
    assert Sv == Matrix([1, -2])
    assert Tv == Matrix([3, 1])
    assert Sv != v
    assert Tv != v

    dsz = lambda a, b: (a.T * J * b)[0]
    assert dsz(v, w) == 5
    assert dsz(S * v, S * w) == 5
    assert dsz(T * v, T * w) == 5

    # Composition coherence.
    assert (T * S) * v == T * (S * v)

    # Negative: identity transport in the dual frame.
    negative_untransported = v
    assert Sv != negative_untransported

    print("IMR-TN-005 certificate PASS")


if __name__ == "__main__":
    main()
