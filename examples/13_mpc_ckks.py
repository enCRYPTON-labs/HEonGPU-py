# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""Multiparty computation with CKKS (N-out-of-N threshold encryption).

Every party has its own secret key and nobody knows the collective secret
key, the sum of them. The parties:

1. build a collective public key and collective Galois keys from shares,
2. encrypt their inputs with the collective public key,
3. let a server compute on the ciphertexts,
4. decrypt the result together: every party computes a partial decryption,
   and the partial decryptions are combined,
5. refresh a ciphertext together (collective bootstrapping) instead of
   using CKKS bootstrapping.

All parties run in this process; in practice every share is serialized
(to_bytes()) and sent to the server.

The computation below needs no relinearization: it multiplies by
plaintexts, so the collective relinearization key is not generated.
"""

import numpy as np

import heongpu as hg
from heongpu import ckks

PARTIES = 3


class Party:
    """A participant: its secret key and its multiparty manager."""

    def __init__(self, context, scale):
        self.secret_key = ckks.KeyGenerator(context).secret_key()
        self.encoder = ckks.Encoder(context)
        self.manager = ckks.MultiPartyManager(context, self.encoder, scale)


def main():
    # All parties agree on the parameters and on a common random seed.
    params = ckks.Params(
        poly_modulus_degree=16384,
        coeff_modulus_bit_sizes=[60, 50, 50, 50],
        special_modulus_bit_sizes=[60],
    )
    scale = 2.0**50
    context = params.make_context()
    context.generate()
    seed = ckks.RNGSeed()  # shared by all parties (also as bytes: seed.key)
    parties = [Party(context, scale) for _ in range(PARTIES)]
    encoder = ckks.Encoder(context)
    server = ckks.MultiPartyManager(context, encoder, scale)
    n = context.slot_count

    # ------------------------------------------------------------------
    # 1. Collective keys
    # ------------------------------------------------------------------
    public_key_shares = [
        p.manager.generate_public_key_share(p.secret_key, seed) for p in parties
    ]
    public_key = server.assemble_public_key_share(public_key_shares)
    galois_key_shares = [
        p.manager.generate_galois_key_share(p.secret_key, seed, shifts=[1, -1])
        for p in parties
    ]
    galois_key = server.assemble_galois_key_share(galois_key_shares)
    share_bytes = public_key_shares[0].to_bytes()
    print(
        f"public key share: {len(share_bytes) / 2**10:.0f} KiB;"
        f" collective keys: {type(public_key).__name__},"
        f" {type(galois_key).__name__}"
    )
    # A share read back from bytes needs the context and the seed.
    ckks.MultipartyPublicKey.from_bytes(share_bytes, context, seed)

    # ------------------------------------------------------------------
    # 2. Every party encrypts its input with the collective public key
    # ------------------------------------------------------------------
    rng = np.random.default_rng(21)
    inputs = [rng.uniform(-1, 1, n) for _ in range(PARTIES)]
    encryptor = ckks.Encryptor(context, public_key)
    ciphertexts = [encryptor.encrypt(encoder.encode(v, scale)) for v in inputs]

    # ------------------------------------------------------------------
    # 3. The server: a weighted sum and a rotation
    # ------------------------------------------------------------------
    operator = ckks.ArithmeticOperator(context, encoder)
    weights = [0.5, 0.3, 0.2]
    result = None
    for weight, cipher in zip(weights, ciphertexts):
        term = operator.multiply_plain(
            cipher, encoder.encode(np.full(n, weight), scale)
        )
        operator.rescale_inplace(term)
        result = term if result is None else operator.add(result, term)
    result = operator.rotate(result, galois_key, 1)
    expected = np.roll(sum(w * v for w, v in zip(weights, inputs)), -1)

    # ------------------------------------------------------------------
    # 4. Collective decryption
    # ------------------------------------------------------------------
    def decrypt(cipher, who=parties):
        partials = [
            p.manager.decrypt_partial(cipher, p.secret_key) for p in who
        ]
        return encoder.decode(parties[0].manager.decrypt(partials))

    values = decrypt(result)
    print("weighted sum, rotated:", values[:4])
    print("expected:             ", expected[:4])
    np.testing.assert_allclose(values, expected, atol=1e-4)

    # Two of three parties cannot decrypt.
    wrong = decrypt(result, parties[:2])
    print(
        f"with 2 of 3 parties the error is {np.abs(wrong - expected).max():.1e}"
    )

    # ------------------------------------------------------------------
    # 5. Collective bootstrapping
    # ------------------------------------------------------------------
    # The result has used one level. The parties refresh it to the top
    # level with a fresh seed, without CKKS bootstrapping keys.
    print("level before:", result.level)
    refresh_seed = ckks.RNGSeed()
    shares = [
        p.manager.distributed_bootstrapping_participant(
            result, p.secret_key, refresh_seed
        )
        for p in parties
    ]
    refreshed = server.distributed_bootstrapping_coordinator(
        shares, result, refresh_seed
    )
    print("level after: ", refreshed.level)
    np.testing.assert_allclose(decrypt(refreshed), expected, atol=1e-3)

    # The refreshed ciphertext takes more operations.
    again = operator.rotate(refreshed, galois_key, -1)
    np.testing.assert_allclose(decrypt(again), np.roll(expected, 1), atol=1e-3)

    # Shares can also be computed on streams or stored in host memory.
    options = hg.ExecutionOptions(storage=hg.StorageType.HOST)
    share = parties[0].manager.generate_public_key_share(
        parties[0].secret_key, seed, options=options
    )
    print("share in host memory:", not share.is_on_device)

    print("OK")


if __name__ == "__main__":
    main()
