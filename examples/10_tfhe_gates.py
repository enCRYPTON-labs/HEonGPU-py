# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""TFHE: Boolean gates with bootstrapping.

A TFHE ciphertext holds a vector of encrypted bits; every gate is applied to
all bits at once and bootstraps its output, so the noise does not grow with
the number of gates. Encryption uses the secret key; the gates use the
bootstrapping key, which can be given to a server.
"""

import numpy as np

import heongpu as hg
from heongpu import tfhe


def main():
    # ------------------------------------------------------------------
    # 1. Context and keys
    # ------------------------------------------------------------------
    context = tfhe.Context()  # the default parameter set of HEonGPU
    print(context)
    keygen = tfhe.KeyGenerator(context)
    secret_key = keygen.secret_key()
    boot_key = keygen.bootstrapping_key(secret_key)
    encryptor = tfhe.Encryptor(context, secret_key)
    decryptor = tfhe.Decryptor(context, secret_key)
    gates = tfhe.LogicOperator(context)

    # ------------------------------------------------------------------
    # 2. Truth tables
    # ------------------------------------------------------------------
    # encrypt() takes bools or 0/1 integers; decrypt() returns a bool array.
    a = encryptor.encrypt([0, 0, 1, 1])
    b = encryptor.encrypt([0, 1, 0, 1])
    print("a     ", decryptor.decrypt(a).astype(int))
    print("b     ", decryptor.decrypt(b).astype(int))
    for name in ("AND", "OR", "XOR", "NAND", "NOR", "XNOR"):
        result = getattr(gates, name)(a, b, boot_key)
        print(f"{name:<6}", decryptor.decrypt(result).astype(int))
    # NOT needs no bootstrapping key.
    print("NOT a ", decryptor.decrypt(gates.NOT(a)).astype(int))
    # MUX(x, y, c) = c ? x : y.
    c = encryptor.encrypt([1, 0, 1, 0])
    print(
        "MUX(a, b, c) with c = [1 0 1 0]:",
        decryptor.decrypt(gates.MUX(a, b, c, boot_key)).astype(int),
    )

    # ------------------------------------------------------------------
    # 3. Many bits at once and outputs into existing ciphertexts
    # ------------------------------------------------------------------
    rng = np.random.default_rng(5)
    size = 1024
    x = rng.integers(0, 2, size).astype(bool)
    y = rng.integers(0, 2, size).astype(bool)
    ex, ey = encryptor.encrypt(x), encryptor.encrypt(y)
    out = tfhe.Ciphertext(context)
    gates.XOR(ex, ey, boot_key, out=out)  # writes into out
    assert np.array_equal(decryptor.decrypt(out), x ^ y)
    print(f"XOR of {size} encrypted bit pairs OK")

    # ------------------------------------------------------------------
    # 4. A chain of gates: the parity and the conjunction of 8 bit vectors
    # ------------------------------------------------------------------
    # Each step combines the previous gate output with a fresh ciphertext.
    inputs = [rng.integers(0, 2, size).astype(bool) for _ in range(8)]
    encrypted = [encryptor.encrypt(bits) for bits in inputs]
    parity = gates.XOR(encrypted[0], encrypted[1], boot_key)
    every = gates.AND(encrypted[0], encrypted[1], boot_key)
    for cipher in encrypted[2:]:
        gates.XOR(parity, cipher, boot_key, out=parity)
        gates.AND(every, cipher, boot_key, out=every)
    assert np.array_equal(
        decryptor.decrypt(parity), np.bitwise_xor.reduce(inputs)
    )
    assert np.array_equal(
        decryptor.decrypt(every), np.logical_and.reduce(inputs)
    )
    print("parity and conjunction of 8 vectors OK")

    # ------------------------------------------------------------------
    # 5. Client and server
    # ------------------------------------------------------------------
    # The client sends the bootstrapping key and the ciphertexts as bytes;
    # the server computes without the secret key.
    key_bytes = boot_key.to_bytes()
    request = [encryptor.encrypt(x).to_bytes(), encryptor.encrypt(y).to_bytes()]

    server_context = tfhe.Context()
    server_key = tfhe.BootstrappingKey.from_bytes(key_bytes, server_context)
    server_gates = tfhe.LogicOperator(server_context)
    sx, sy = (
        tfhe.Ciphertext.from_bytes(data, server_context) for data in request
    )
    response = server_gates.NAND(sx, sy, server_key).to_bytes()

    result = decryptor.decrypt(tfhe.Ciphertext.from_bytes(response))
    assert np.array_equal(result, ~(x & y))
    print(
        f"server NAND OK (bootstrapping key {len(key_bytes) / 2**20:.0f} MiB)"
    )

    # ------------------------------------------------------------------
    # 6. Streams
    # ------------------------------------------------------------------
    with hg.Stream() as stream:
        ex = encryptor.encrypt(x, options=stream)
        ey = encryptor.encrypt(y, options=stream)
        result = gates.OR(ex, ey, boot_key, options=stream)
        bits = decryptor.decrypt(result, options=stream)
        stream.synchronize()
    assert np.array_equal(bits, x | y)
    print("OR on a stream OK")

    print("OK")


if __name__ == "__main__":
    main()
