# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""CKKS with a Session: approximate arithmetic on real and complex numbers.

A ckks.Session holds everything one party needs: the context, the keys, the
encoder, the encryptor, the decryptor and the operators. Its ciphertexts know
their session, so they support the Python operators +, -, * and ** directly.
By default a session works in the automatic rescale mode: products are
relinearized and rescaled, and operands at different levels are aligned,
without any further call.
"""

import numpy as np

import heongpu as hg
from heongpu import ckks


def show(name, values, count=4):
    print(f"{name:<28} {np.array2string(values[:count], precision=6)}")


def main():
    # ------------------------------------------------------------------
    # 1. The session
    # ------------------------------------------------------------------
    # Params chooses secure parameters (128-bit security by default) for a
    # multiplicative depth: four products in a row. The scale is 2^40.
    # rotations lists the rotation steps whose Galois keys are generated now;
    # other steps are generated the first time they are used.
    session = ckks.Session(
        ckks.Params(multiplicative_depth=4, scale_bit_size=40),
        rotations=[1, 2],
    )
    print(session)
    print("slots:", session.slot_count, " scale: 2^%d" % np.log2(session.scale))
    # The same with keywords: ckks.Session(multiplicative_depth=4, ...).

    # ------------------------------------------------------------------
    # 2. Encryption and decryption
    # ------------------------------------------------------------------
    # A NumPy array or a sequence gives one value per slot (the other slots
    # are 0); a scalar fills every slot.
    n = session.slot_count
    xs = np.linspace(-1.0, 1.0, n)
    ys = 0.5 * np.cos(np.arange(n) / 100.0)
    x = session.encrypt(xs)
    y = session.encrypt(ys)
    print(x)  # level, depth and scale of the ciphertext
    show("decrypt(x)", session.decrypt(x))
    np.testing.assert_allclose(session.decrypt(x), xs, atol=1e-6)

    # ------------------------------------------------------------------
    # 3. Arithmetic
    # ------------------------------------------------------------------
    # The operands can be ciphertexts, real constants or real values, one
    # per slot (a list or an array, also on the left of the operator).
    z = x * y + 0.5 * x - y
    expected = xs * ys + 0.5 * xs - ys
    show("x*y + 0.5x - y", session.decrypt(z))
    np.testing.assert_allclose(session.decrypt(z), expected, atol=1e-5)

    weights = np.full(n, 0.25)
    w = weights * x - 1.0  # a NumPy array on the left works too
    np.testing.assert_allclose(session.decrypt(w), 0.25 * xs - 1.0, atol=1e-5)

    # Powers use repeated squaring, division takes a real constant.
    cube = x**3 / 2
    np.testing.assert_allclose(session.decrypt(cube), xs**3 / 2, atol=1e-5)

    # In-place operators update the ciphertext.
    acc = session.encrypt(0.0)
    for k in range(1, 4):
        acc += k * x
    np.testing.assert_allclose(session.decrypt(acc), 6 * xs, atol=1e-5)

    # Each product uses one level. levels_left says how many are left.
    print(
        "levels left: x =",
        x.levels_left,
        " z =",
        z.levels_left,
        " cube =",
        cube.levels_left,
    )

    # ------------------------------------------------------------------
    # 4. Rotations and sums
    # ------------------------------------------------------------------
    # rotate(k) moves slot i + k to slot i (np.roll(values, -k)).
    r = x.rotate(1)
    np.testing.assert_allclose(session.decrypt(r), np.roll(xs, -1), atol=1e-5)
    # A step without a key gets one now (the session has the secret key).
    r = x.rotate(-3)
    np.testing.assert_allclose(session.decrypt(r), np.roll(xs, 3), atol=1e-5)
    # Several rotations of one ciphertext share its decomposition.
    r1, r2 = x.rotate_hoisted([1, 2])
    np.testing.assert_allclose(session.decrypt(r2), np.roll(xs, -2), atol=1e-5)

    # Sums of all slots and inner products land in every slot.
    total = session.decrypt(y.sum_slots())[0]
    dot = session.decrypt(x.inner_product(y))[0]
    print(
        f"sum(y) = {total:.4f} (expected {ys.sum():.4f}),"
        f" <x, y> = {dot:.4f} (expected {xs @ ys:.4f})"
    )
    assert abs(total - ys.sum()) < 1e-2 and abs(dot - xs @ ys) < 1e-2

    # ------------------------------------------------------------------
    # 5. Complex numbers
    # ------------------------------------------------------------------
    cs = xs + 1j * ys
    c = session.encrypt(cs)
    product = c * c.conjugate()  # |c|^2 in every slot
    np.testing.assert_allclose(
        session.decrypt_complex(product), np.abs(cs) ** 2, atol=1e-5
    )
    show("|x + iy|^2", session.decrypt(product))

    # ------------------------------------------------------------------
    # 6. Logic on bits
    # ------------------------------------------------------------------
    # Slots holding 0 or 1 can be combined with arithmetic gates
    # (AND = ab, XOR = a + b - 2ab, ...); each gate uses one level.
    a = np.arange(n) % 2
    b = (np.arange(n) // 2) % 2
    ea, eb = session.encrypt(a), session.encrypt(b)
    for name, gate, truth in (
        ("AND", ckks.AND, a & b),
        ("OR", ckks.OR, a | b),
        ("XOR", ckks.XOR, a ^ b),
        ("NAND", ckks.NAND, 1 - (a & b)),
    ):
        bits = np.round(session.decrypt(gate(ea, eb))).astype(int)
        assert np.array_equal(bits, truth), name
    not_a = np.round(session.decrypt(ckks.NOT(ea))).astype(int)
    assert np.array_equal(not_a, 1 - a)
    print("a[:6] =", a[:6], " b[:6] =", b[:6], " NOT(a)[:6] =", not_a[:6])

    # ------------------------------------------------------------------
    # 7. Errors are exceptions
    # ------------------------------------------------------------------
    other = ckks.Session(multiplicative_depth=1)
    try:
        x + other.encrypt(1.0)
    except hg.ContextMismatchError as error:
        print("ContextMismatchError:", str(error).splitlines()[0][:70])

    print("OK")


if __name__ == "__main__":
    main()
