# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""CKKS rotations and polynomials with a Session.

The first part multiplies a plaintext matrix with an encrypted vector by the
diagonal method, with hoisted rotations. The second part evaluates
polynomials: from monomial coefficients, from Chebyshev coefficients and by
interpolating a Python function (here the sigmoid) with a Chebyshev series.
"""

import numpy as np

from heongpu import ckks

D = 8  # the size of the matrix


def main():
    # The depth covers the degree 31 Chebyshev series (6 levels) and one
    # more product.
    session = ckks.Session(
        multiplicative_depth=7, scale_bit_size=40, rotations=list(range(1, D))
    )
    n = session.slot_count
    rng = np.random.default_rng(3)

    # ------------------------------------------------------------------
    # 1. Matrix-vector product by the diagonal method
    # ------------------------------------------------------------------
    # The vector v is repeated over all slots, so that a rotation of the
    # ciphertext rotates every copy of v. Then
    #   M v = sum_k diag_k * rotate(v, k),
    # with diag_k[i] = M[i, (i + k) mod D] (also repeated).
    matrix = rng.uniform(-1, 1, (D, D))
    vector = rng.uniform(-1, 1, D)
    encrypted = session.encrypt(np.tile(vector, n // D))

    rows = np.arange(D)
    diagonals = [
        np.tile(matrix[rows, (rows + k) % D], n // D) for k in range(D)
    ]
    # rotate_hoisted computes all rotations from one decomposition of the
    # ciphertext, which is faster than D separate rotations.
    rotations = [encrypted] + encrypted.rotate_hoisted(list(range(1, D)))
    result = rotations[0] * diagonals[0]
    for k in range(1, D):
        result += rotations[k] * diagonals[k]

    product = session.decrypt(result)[:D]
    print("M v (encrypted):", np.round(product, 5))
    print("M v (NumPy):    ", np.round(matrix @ vector, 5))
    np.testing.assert_allclose(product, matrix @ vector, atol=1e-5)

    # ------------------------------------------------------------------
    # 2. Polynomials
    # ------------------------------------------------------------------
    xs = np.linspace(-1.0, 1.0, n)
    x = session.encrypt(xs)

    # Monomial coefficients, constant term first: 0.5 - x + 0.25 x^2 + 2 x^3.
    coefficients = [0.5, -1.0, 0.25, 2.0]
    p = session.evaluate_polynomial(x, coefficients)
    expected = np.polynomial.polynomial.polyval(xs, coefficients)
    np.testing.assert_allclose(session.decrypt(p), expected, atol=1e-5)
    print(
        "p(x) = 0.5 - x + x^2/4 + 2x^3: levels used",
        x.levels_left - p.levels_left,
    )

    # Chebyshev coefficients on [a, b]: T0 + 0.5 T1 + 0.25 T2.
    q = session.evaluate_chebyshev(x, [1.0, 0.5, 0.25], -1.0, 1.0)
    np.testing.assert_allclose(
        session.decrypt(q),
        np.polynomial.chebyshev.chebval(xs, [1, 0.5, 0.25]),
        atol=1e-5,
    )

    # A function interpolated on [a, b] with a Chebyshev series of the given
    # degree. The values must be in [a, b].
    def sigmoid(t):
        return 1.0 / (1.0 + np.exp(-t))

    zs = rng.uniform(-6.0, 6.0, n)
    z = session.encrypt(zs)
    s = session.evaluate_chebyshev(z, sigmoid, -6.0, 6.0, degree=31)
    error = np.abs(session.decrypt(s) - sigmoid(zs)).max()
    print(
        f"sigmoid on [-6, 6], degree 31: max error {error:.2e},"
        f" levels used {z.levels_left - s.levels_left}"
    )
    assert error < 1e-3

    # The interpolation can also run on the CPU, for example to check the
    # approximation error before encrypting anything.
    coefficients = ckks.chebyshev_coefficients(sigmoid, -6.0, 6.0, 31)
    nodes = np.linspace(-6.0, 6.0, 1001)
    approximation = np.polynomial.chebyshev.chebval(nodes / 6.0, coefficients)
    error = np.abs(approximation - sigmoid(nodes)).max()
    print(f"  CPU interpolation error: {error:.2e}")
    print(
        "  depth of a degree 31 Chebyshev series on [-6, 6]:",
        ckks.ArithmeticOperator.polynomial_depth(31, True, -6.0, 6.0),
    )

    # The result is an ordinary ciphertext: keep computing with it while
    # levels_left is not negative (a ciphertext that runs out of levels
    # decrypts to wrong values).
    t = s * 2.0 - 1.0  # tanh(z / 2)
    assert t.levels_left >= 0
    np.testing.assert_allclose(session.decrypt(t), np.tanh(zs / 2), atol=2e-3)

    print("OK")


if __name__ == "__main__":
    main()
