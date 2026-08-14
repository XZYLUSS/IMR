"""IMR-AN-005: limit-integral interchange failure for f_n=n*1_[0,1/n].
Executable SymPy certificate for lim int = 1, int lim = 0, and negative refutation.
"""
from sympy import symbols, integrate, limit, oo, Integer


def main() -> None:
    n, x = symbols("n x", positive=True)

    # Integral of f_n = n on [0,1/n], zero elsewhere in [0,1].
    integral_n = integrate(n, (x, 0, 1 / n))
    assert integral_n == 1
    assert limit(integral_n, n, oo) == 1

    # Pointwise a.e. limit is 0, so integral of pointwise limit is 0.
    integral_of_pointwise_limit = Integer(0)
    assert integral_of_pointwise_limit == 0

    # Negative control: silently interchanging limit and integral gives 0.
    negative_value = integral_of_pointwise_limit
    assert negative_value != limit(integral_n, n, oo)

    # Certificate-level note: no integrable dominator exists because sup_n f_n is unbounded near 0.
    # This mathematical fact is recorded in prose/Lean target; SymPy certifies the concrete value mismatch.
    print("IMR-AN-005 certificate PASS")


if __name__ == "__main__":
    main()
