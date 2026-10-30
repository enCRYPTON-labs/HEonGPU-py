# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""CKKS with the lower level API, in the manual rescale mode.

The lower level API gives every HEonGPU class: Context, KeyGenerator, the
keys, Encoder, Encryptor, Decryptor and ArithmeticOperator. In the manual
rescale mode the application relinearizes after products, rescales after
products and brings operands to the same level with mod_drop, as in the
HEonGPU C++ examples.
"""

import numpy as np

import heongpu as hg
from heongpu import ckks


def main():
    # ------------------------------------------------------------------
    # 1. Parameters and context
    # ------------------------------------------------------------------
    # Params chooses the ring size and the moduli for a depth and a scale.
    # The moduli can also be given explicitly, for example
    #   ckks.Params(poly_modulus_degree=16384,
    #               coeff_modulus_bit_sizes=[60, 40, 40, 40, 40],
    #               special_modulus_bit_sizes=[60])
    params = ckks.Params(multiplicative_depth=3, scale_bit_size=40)
    print(params.generate())  # the parameter set Params chooses
    context = params.make_context()
    context.generate()  # computes the tables on the GPU
    scale = params.scale()  # 2^40
    print("rescale mode:", context.rescale_mode, " slots:", context.slot_count)

    # ------------------------------------------------------------------
    # 2. Keys
    # ------------------------------------------------------------------
    keygen = ckks.KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.public_key(secret_key)
    relin_key = keygen.relin_key(secret_key)
    # Galois keys for the rotation steps 1 and -4, and the conjugation.
    galois_key = keygen.galois_key(secret_key, [1, -4])

    encoder = ckks.Encoder(context)
    encryptor = ckks.Encryptor(context, public_key)
    decryptor = ckks.Decryptor(context, secret_key)
    operator = ckks.ArithmeticOperator(context, encoder)

    def decrypt(cipher):
        return encoder.decode(decryptor.decrypt(cipher))

    # ------------------------------------------------------------------
    # 3. Encode and encrypt
    # ------------------------------------------------------------------
    n = encoder.slot_count
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.sin(np.linspace(0.0, 3.0, n))
    zs = np.full(n, 0.5)
    plain_x = encoder.encode(xs, scale)  # a Plaintext
    x = encryptor.encrypt(plain_x)  # a Ciphertext
    y = encryptor.encrypt(encoder.encode(ys, scale))
    z = encryptor.encrypt(encoder.encode(zs, scale))
    print(
        "x: level", x.level, " depth", x.depth, " scale 2^%d" % np.log2(x.scale)
    )

    # ------------------------------------------------------------------
    # 4. x * y + z with relinearization, rescaling and mod_drop
    # ------------------------------------------------------------------
    product = operator.multiply(x, y)  # 3 polynomials, scale 2^80
    print(
        "product: size",
        product.size,
        " relinearization required:",
        product.relinearization_required,
    )
    operator.relinearize_inplace(product, relin_key)  # back to 2
    operator.rescale_inplace(product)  # one level down, scale 2^40
    print("after rescale: level", product.level, " depth", product.depth)

    # z is still at the top level: drop it to the level of the product.
    try:
        operator.add(product, z)
    except hg.LevelMismatchError as error:
        print("LevelMismatchError:", str(error).splitlines()[0][:60], "...")
    operator.mod_drop_inplace(z)
    result = operator.add(product, z)
    np.testing.assert_allclose(decrypt(result), xs * ys + zs, atol=1e-6)
    print("x*y + z  ", decrypt(result)[:4])

    # ------------------------------------------------------------------
    # 5. Plaintexts and constants
    # ------------------------------------------------------------------
    # A plaintext or a constant: add_plain/sub_plain keep the scale,
    # multiply_plain multiplies the scales, so it is followed by a rescale.
    shifted = operator.add_plain(x, plain_x)  # 2x
    scaled = operator.multiply_plain(shifted, 0.25)  # 0.5x
    operator.rescale_inplace(scaled)
    np.testing.assert_allclose(decrypt(scaled), 0.5 * xs, atol=1e-6)

    # ------------------------------------------------------------------
    # 6. Rotations, conjugation and key switching
    # ------------------------------------------------------------------
    rotated = operator.rotate(x, galois_key, 1)
    np.testing.assert_allclose(decrypt(rotated), np.roll(xs, -1), atol=1e-6)
    rotated = operator.rotate(x, galois_key, -4)
    np.testing.assert_allclose(decrypt(rotated), np.roll(xs, 4), atol=1e-6)

    complex_values = xs + 1j * ys
    c = encryptor.encrypt(encoder.encode(complex_values, scale))
    conjugated = operator.conjugate(c, galois_key)
    np.testing.assert_allclose(
        encoder.decode_complex(decryptor.decrypt(conjugated)),
        np.conj(complex_values),
        atol=1e-6,
    )
    # mult_i multiplies by the imaginary unit without using a level.
    times_i = operator.mult_i(c)
    np.testing.assert_allclose(
        encoder.decode_complex(decryptor.decrypt(times_i)),
        1j * complex_values,
        atol=1e-6,
    )

    # A switch key moves a ciphertext to another secret key.
    new_secret_key = keygen.secret_key()
    switch_key = ckks.SwitchKey(context)
    keygen.generate_switch_key(switch_key, new_secret_key, secret_key)
    switched = operator.keyswitch(x, switch_key)
    new_decryptor = ckks.Decryptor(context, new_secret_key)
    np.testing.assert_allclose(
        encoder.decode(new_decryptor.decrypt(switched)), xs, atol=1e-6
    )
    print("rotations, conjugation and key switching OK")

    # ------------------------------------------------------------------
    # 7. Logic operations on bits
    # ------------------------------------------------------------------
    # LogicOperator works in the manual rescale mode. In HEonGPU 2.1.0 its
    # AND, NAND and NOT give correct results; see the Known limitations
    # section of the README for the other gates.
    logic = ckks.LogicOperator(context, encoder, scale)
    a = (np.arange(n) % 2).astype(float)
    b = ((np.arange(n) // 2) % 2).astype(float)
    ea = encryptor.encrypt(encoder.encode(a, scale))
    eb = encryptor.encrypt(encoder.encode(b, scale))
    nand = logic.NAND(ea, eb, relin_key)
    np.testing.assert_allclose(decrypt(nand), 1 - a * b, atol=1e-5)
    print("a NAND b ", np.round(decrypt(nand)[:4]).astype(int))

    print("OK")


if __name__ == "__main__":
    main()
