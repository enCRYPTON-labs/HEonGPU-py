# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""BFV with the lower level API.

The lower level API gives every HEonGPU class: Context, KeyGenerator, the
keys, Encoder, Encryptor, Decryptor, ArithmeticOperator and LogicOperator.
The application relinearizes after ciphertext products and watches the
noise budget.
"""

import numpy as np

from heongpu import bfv

PLAIN_MODULUS = 786433  # a prime with t = 1 (mod 2N): batching


def main():
    # ------------------------------------------------------------------
    # 1. Parameters and context
    # ------------------------------------------------------------------
    params = bfv.Params(
        multiplicative_depth=3,
        plain_modulus=PLAIN_MODULUS,
        poly_modulus_degree=8192,
    )
    context = params.make_context()
    context.generate()
    t = context.plain_modulus
    context.print_parameters()

    # ------------------------------------------------------------------
    # 2. Keys and the classes that use them
    # ------------------------------------------------------------------
    keygen = bfv.KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.public_key(secret_key)
    relin_key = keygen.relin_key(secret_key)
    galois_key = keygen.galois_key(secret_key, [1, 3])

    encoder = bfv.Encoder(context)
    encryptor = bfv.Encryptor(context, public_key)
    decryptor = bfv.Decryptor(context, secret_key)
    operator = bfv.ArithmeticOperator(context, encoder)
    n = encoder.slot_count

    def encrypt(values):
        return encryptor.encrypt(encoder.encode(values))

    def decrypt(cipher, signed=False):
        return encoder.decode(decryptor.decrypt(cipher), signed=signed)

    # ------------------------------------------------------------------
    # 3. (x * y + z) * w
    # ------------------------------------------------------------------
    rng = np.random.default_rng(7)
    xs, ys, zs, ws = (rng.integers(0, 1000, n) for _ in range(4))
    x, y, z = encrypt(xs), encrypt(ys), encrypt(zs)
    print(
        "noise budget of a fresh ciphertext:",
        decryptor.remainder_noise_budget(x),
        "bits",
    )

    product = operator.multiply(x, y)  # 3 polynomials
    operator.relinearize_inplace(product, relin_key)  # back to 2
    total = operator.add(product, z)
    # A plaintext operand: multiply_plain needs no relinearization.
    result = operator.multiply_plain(total, encoder.encode(ws))
    expected = ((xs * ys + zs) % t) * ws % t
    assert np.array_equal(decrypt(result), expected)
    print("(x*y + z) * w   ", decrypt(result)[:6])
    print(
        "noise budget left:", decryptor.remainder_noise_budget(result), "bits"
    )

    # Negative numbers: encode takes them modulo t, decode(signed=True)
    # returns the centered values.
    difference = operator.sub(encrypt([3, 5, 7]), encrypt([10, 10, 10]))
    print("[3, 5, 7] - 10  ", decrypt(difference, signed=True)[:3])
    negated = operator.negate(encrypt([3, 5, 7]))
    assert list(decrypt(negated, signed=True)[:3]) == [-3, -5, -7]

    # ------------------------------------------------------------------
    # 4. Rotations, sums and key switching
    # ------------------------------------------------------------------
    rows = xs.reshape(2, -1)
    rotated = operator.rotate_rows(x, galois_key, 3)
    assert np.array_equal(
        decrypt(rotated), np.roll(rows, -3, axis=1).reshape(-1)
    )
    swapped = operator.rotate_columns(x, galois_key)
    assert np.array_equal(decrypt(swapped), rows[::-1].reshape(-1))

    # sum_slots needs the keys of the steps +-1, +-2, +-4, ..., +-N/4: a
    # GaloisKey with max_shift = log2(N/4) holds them.
    full_galois_key = bfv.GaloisKey(context, max_shift=n.bit_length() - 3)
    keygen.generate_galois_key(full_galois_key, secret_key)
    total = operator.sum_slots(x, full_galois_key)
    assert decrypt(total)[0] == xs.sum() % t
    print("sum of all slots", decrypt(total)[0])

    new_secret_key = keygen.secret_key()
    switch_key = bfv.SwitchKey(context)
    keygen.generate_switch_key(switch_key, new_secret_key, secret_key)
    switched = operator.keyswitch(x, switch_key)
    other_decryptor = bfv.Decryptor(context, new_secret_key)
    assert np.array_equal(encoder.decode(other_decryptor.decrypt(switched)), xs)
    print("rotations and key switching OK")

    # ------------------------------------------------------------------
    # 5. Logic gates on bits
    # ------------------------------------------------------------------
    logic = bfv.LogicOperator(context, encoder)
    a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
    ea, eb = encrypt(a), encrypt(b)
    gates = {
        "AND": a & b,
        "OR": a | b,
        "XOR": a ^ b,
        "NAND": 1 - (a & b),
        "NOR": 1 - (a | b),
        "XNOR": 1 - (a ^ b),
    }
    for name, truth in gates.items():
        result = getattr(logic, name)(ea, eb, relin_key)
        assert np.array_equal(decrypt(result), truth), name
    assert np.array_equal(decrypt(logic.NOT(ea)), 1 - a)
    print("logic gates OK")

    print("OK")


if __name__ == "__main__":
    main()
