# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""CKKS logic bootstrapping: bit and gate bootstrapping with a Session.

The slots hold bits (0 or 1).

- Bit bootstrapping refreshes a ciphertext of bits and reduces the error of
  the bits: bootstrap() of a session created from BootstrappingPreset.bit().
- Gate bootstrapping evaluates a gate while it bootstraps, as TFHE does:
  AND_bootstrapping(), OR_bootstrapping(), ... of a session created from
  BootstrappingPreset.gate(). The output is a fresh ciphertext of bits.

As in 11_ckks_bootstrapping.py, the presets use N = 2^16 and a session with
few levels needs a device memory pool of about 12 GiB; the sessions below
are created one after the other.
"""

import gc
import time

import numpy as np

from heongpu import ckks


def bit_bootstrapping():
    print("== Bit bootstrapping")
    session = ckks.Session(ckks.BootstrappingPreset.bit(2))
    levels = session.bootstrapping_levels()
    print(
        f"  levels: input {levels.input_level}, output {levels.output_level},"
        f" usable {levels.usable_levels}"
    )
    n = session.slot_count
    bits = (np.arange(n) % 3 == 0).astype(float)

    # Bits with an error: 0 and 1 plus up to +-0.05.
    noisy = bits + 0.05 * np.cos(np.arange(n))
    x = session.encrypt(noisy)
    start = time.time()
    clean = x.bootstrap()
    elapsed = time.time() - start
    error_before = np.abs(session.decrypt(x) - bits).max()
    error_after = np.abs(session.decrypt(clean) - bits).max()
    print(
        f"  error of the bits: {error_before:.1e} before, {error_after:.1e}"
        f" after ({elapsed:.2f} s)"
    )
    assert error_after < 0.25 * error_before

    # Arithmetic logic gates use levels; bit bootstrapping restores them.
    # x AND y = x y uses one level per gate.
    ys = ((np.arange(n) // 2) % 2).astype(float)
    y = session.encrypt(ys)
    result = clean
    for _ in range(4):
        if result.levels_left < 1:
            result = result.bootstrap()
        result = ckks.AND(result, y)
    expected = bits.astype(int) & ys.astype(int)
    assert np.array_equal(np.round(session.decrypt(result)), expected)
    print("  4 ANDs with bit bootstrapping OK")


def gate_bootstrapping():
    print("== Gate bootstrapping: adding 4-bit numbers")
    session = ckks.Session(ckks.BootstrappingPreset.gate(1))
    n = session.slot_count
    rng = np.random.default_rng(11)
    width = 4
    a = rng.integers(0, 2**width, n)
    b = rng.integers(0, 2**width, n)

    def encrypt_bit(values, i):
        return session.encrypt(((values >> i) & 1).astype(float))

    # A ripple carry adder; every gate bootstraps, so the depth of the
    # circuit does not matter.
    start = time.time()
    carry = None
    total = []
    for i in range(width):
        x, y = encrypt_bit(a, i), encrypt_bit(b, i)
        half = session.XOR_bootstrapping(x, y)
        if carry is None:
            total.append(half)
            carry = session.AND_bootstrapping(x, y)
        else:
            total.append(session.XOR_bootstrapping(half, carry))
            carry = session.OR_bootstrapping(
                session.AND_bootstrapping(x, y),
                session.AND_bootstrapping(half, carry),
            )
    total.append(carry)
    elapsed = time.time() - start

    result = sum(
        np.round(session.decrypt(bit)).astype(np.int64) << i
        for i, bit in enumerate(total)
    )
    assert np.array_equal(result, a + b)
    print(f"  {n} additions of 4-bit numbers in {elapsed:.2f} s:")
    print("  a     ", a[:8])
    print("  b     ", b[:8])
    print("  a + b ", result[:8])


def main():
    bit_bootstrapping()
    gc.collect()
    gate_bootstrapping()
    print("OK")


if __name__ == "__main__":
    main()
