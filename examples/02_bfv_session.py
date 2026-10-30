# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""BFV with a Session: exact arithmetic on integers modulo t.

A bfv.Session holds the context, the keys, the encoder, the encryptor, the
decryptor and the operators. Its ciphertexts support +, - and * with other
ciphertexts, integers and integer vectors; products are relinearized
automatically. All results are exact modulo the plaintext modulus t.
"""

import numpy as np

from heongpu import bfv

# A prime with t = 1 (mod 2N) for N up to 32768, so that every slot holds its
# own value (batching).
PLAIN_MODULUS = 786433


def main():
    # ------------------------------------------------------------------
    # 1. The session
    # ------------------------------------------------------------------
    params = bfv.Params(
        multiplicative_depth=2,
        plain_modulus=PLAIN_MODULUS,
        poly_modulus_degree=8192,
    )
    session = bfv.Session(params, rotations=[1, -2])
    t = session.plain_modulus
    n = session.slot_count
    print(session)
    # Without plain_modulus, Params picks a batching prime of
    # plain_modulus_bit_size bits (20 by default).

    # ------------------------------------------------------------------
    # 2. Encryption, decryption and the noise budget
    # ------------------------------------------------------------------
    rng = np.random.default_rng(2026)
    xs = rng.integers(0, 1000, n)
    ys = rng.integers(0, 1000, n)
    x = session.encrypt(xs)
    y = session.encrypt(ys)
    decrypted = session.decrypt(x)  # uint64 values in [0, t)
    assert np.array_equal(decrypted, xs)
    print("decrypt(x)[:6]       ", decrypted[:6], decrypted.dtype)

    # Negative values are taken modulo t; decrypt(signed=True) returns the
    # centered representatives in (-t/2, t/2].
    negative = session.encrypt([-5, -1, 0, 7])
    print("unsigned             ", session.decrypt(negative)[:4])
    print("signed               ", session.decrypt(negative, signed=True)[:4])

    # Each operation consumes noise budget (bits); at 0 decryption fails.
    print("noise budget: fresh =", session.noise_budget(x), "bits")

    # ------------------------------------------------------------------
    # 3. Arithmetic modulo t
    # ------------------------------------------------------------------
    z = x * x * y + 3 * x - y + 10
    expected = (xs * xs * ys + 3 * xs - ys + 10) % t
    assert np.array_equal(session.decrypt(z), expected)
    print("x*x*y + 3x - y + 10  ", session.decrypt(z)[:6])
    print("noise budget: after  =", session.noise_budget(z), "bits")

    # Vectors of integers act slot by slot, also on the left; -x and +x and
    # the in-place operators work as usual.
    w = ys * x - xs
    assert np.array_equal(session.decrypt(w), (ys * xs - xs) % t)
    w += 1
    w *= 2
    assert np.array_equal(session.decrypt(w), ((ys * xs - xs + 1) * 2) % t)

    # ------------------------------------------------------------------
    # 4. Rotations
    # ------------------------------------------------------------------
    # The N slots form a 2 x N/2 matrix. rotate(k) rotates both rows left by
    # k; rotate_columns() swaps the two rows.
    rows = xs.reshape(2, -1)
    r = x.rotate(1)
    assert np.array_equal(
        session.decrypt(r), np.roll(rows, -1, axis=1).reshape(-1)
    )
    r = x.rotate(-2)
    assert np.array_equal(
        session.decrypt(r), np.roll(rows, 2, axis=1).reshape(-1)
    )
    swapped = x.rotate_columns()
    assert np.array_equal(session.decrypt(swapped), rows[::-1].reshape(-1))
    print("row size:", session.row_size, " rotations OK")

    # Sum of all slots and inner products, in every slot.
    total = session.decrypt(x.sum_slots())[0]
    dot = session.decrypt(x.inner_product(y))[0]
    assert total == xs.sum() % t and dot == int(xs @ ys) % t
    print(f"sum(x) mod t = {total},  <x, y> mod t = {dot}")

    # ------------------------------------------------------------------
    # 5. Logic on bits
    # ------------------------------------------------------------------
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    ea, eb = session.encrypt(a), session.encrypt(b)
    circuit = bfv.XOR(bfv.AND(ea, eb), bfv.NOT(eb))
    assert np.array_equal(session.decrypt(circuit), (a & b) ^ (1 - b))
    print("(a AND b) XOR NOT b  ", session.decrypt(circuit)[:8])

    # ------------------------------------------------------------------
    # 6. Errors
    # ------------------------------------------------------------------
    try:
        session.encrypt([1.5])
    except TypeError as error:
        print("TypeError:", error)

    print("OK")


if __name__ == "__main__":
    main()
