# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""CKKS in the automatic rescale mode with the lower level API.

In the automatic rescale mode the operators rescale a product when it is
used next, bring the operands of an operation to the same level and give
every level a fixed scale. The application only relinearizes. Sessions use
this mode by default; this example shows it with the lower level classes.
"""

import numpy as np

import heongpu as hg
from heongpu import ckks


def main():
    params = ckks.Params(
        multiplicative_depth=4,
        scale_bit_size=40,
        rescale_mode=hg.RescaleMode.AUTOMATIC,
    )
    context = params.make_context()
    context.generate()
    top = context.top_level
    print("rescale mode:", context.rescale_mode, " top level:", top)
    for level in range(top, top - 3, -1):
        bits = np.log2(context.get_level_scale(level))
        print(f"  scale of level {level}: 2^{bits:.4f}")

    keygen = ckks.KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.public_key(secret_key)
    relin_key = keygen.relin_key(secret_key)
    galois_key = keygen.galois_key(secret_key, [2])
    encoder = ckks.Encoder(context)
    encryptor = ckks.Encryptor(context, public_key)
    decryptor = ckks.Decryptor(context, secret_key)
    operator = ckks.ArithmeticOperator(context, encoder)

    def decrypt(cipher):
        return encoder.decode(decryptor.decrypt(cipher))

    n = encoder.slot_count
    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)

    # The encoder takes the scale of the level, so no scale is passed.
    x = encryptor.encrypt(encoder.encode(xs))
    y = encryptor.encrypt(encoder.encode(ys))
    print("x: level", x.level)

    # x^2: only the relinearization is explicit. The product waits for its
    # rescale, which the next operation does.
    square = operator.multiply(x, x)
    operator.relinearize_inplace(square, relin_key)
    print(
        "x^2: level",
        square.level,
        " rescale required:",
        square.rescale_required,
    )

    # x^2 * y: the square is rescaled, y is brought to its level.
    product = operator.multiply(square, y)
    operator.relinearize_inplace(product, relin_key)
    # x^2 * y + x: x is brought down two levels.
    result = operator.add(product, x)
    print("x^2*y + x: level", result.level)
    np.testing.assert_allclose(decrypt(result), xs * xs * ys + xs, atol=1e-5)

    # Constants and plaintexts get the scale of the level as well.
    halved = operator.multiply_plain(x, 0.5)
    rotated = operator.rotate(halved, galois_key, 2)
    shifted = operator.add_plain(rotated, 0.25)
    np.testing.assert_allclose(
        decrypt(shifted), 0.5 * np.roll(xs, -2) + 0.25, atol=1e-6
    )
    # A plaintext for a ciphertext one level down: encode it at depth 1.
    plain = encoder.encode(ys, depth=1)
    np.testing.assert_allclose(
        decrypt(operator.multiply_plain(rotated, plain)),
        0.5 * np.roll(xs, -2) * ys,
        atol=1e-6,
    )

    # Polynomials: 1 + x + 0.5 x^2.
    poly = operator.evaluate_polynomial(x, [1.0, 1.0, 0.5], relin_key)
    np.testing.assert_allclose(decrypt(poly), 1 + xs + 0.5 * xs**2, atol=1e-5)
    print(
        "1 + x + x^2/2:",
        decrypt(poly)[:3],
        " expected",
        (1 + xs + 0.5 * xs**2)[:3],
    )

    # The operators check the scales: a ciphertext with another scale is
    # rejected.
    other = encryptor.encrypt(encoder.encode(ys, 1.001 * 2.0**40))
    try:
        operator.add(x, other)
    except hg.ParameterError as error:
        print("ParameterError:", str(error).splitlines()[0][:70])

    print("OK")


if __name__ == "__main__":
    main()
