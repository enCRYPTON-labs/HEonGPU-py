# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""Multiparty computation with BFV (N-out-of-N threshold encryption).

Three parties count votes: every party encrypts a vector of 0/1 votes with
the collective public key, a server adds them, finds the unanimous votes
with products of ciphertexts and computes a weighted score, and the parties
decrypt the results together. A long chain of plaintext products shows the
collective bootstrapping, which restores the noise budget without the
secret keys leaving the parties.

A party keeps its multiparty manager between the two rounds of the
collective relinearization key.
"""

import numpy as np

from heongpu import bfv

PARTIES = 3
PLAIN_MODULUS = 786433  # a batching prime (t = 1 mod 2N)


class Party:
    def __init__(self, context):
        self.secret_key = bfv.KeyGenerator(context).secret_key()
        self.manager = bfv.MultiPartyManager(context)


def main():
    params = bfv.Params(
        multiplicative_depth=2,
        plain_modulus=PLAIN_MODULUS,
        poly_modulus_degree=8192,
    )
    context = params.make_context()
    context.generate()
    t = context.plain_modulus
    seed = bfv.RNGSeed()  # the common random seed of the parties
    parties = [Party(context) for _ in range(PARTIES)]
    server = bfv.MultiPartyManager(context)

    # ------------------------------------------------------------------
    # 1. Collective public, relinearization and Galois keys
    # ------------------------------------------------------------------
    public_key = server.assemble_public_key_share(
        [
            p.manager.generate_public_key_share(p.secret_key, seed)
            for p in parties
        ]
    )
    # The relinearization key takes two rounds: the server combines the
    # shares of the first round, and every party computes its share of the
    # second round from that combination.
    relin_key_common = server.assemble_relin_key_init(
        [p.manager.generate_relin_key_init(p.secret_key, seed) for p in parties]
    )
    relin_key = server.assemble_relin_key_share(
        [
            p.manager.generate_relin_key_share(relin_key_common, p.secret_key)
            for p in parties
        ],
        relin_key_common,
    )
    galois_key = server.assemble_galois_key_share(
        [
            p.manager.generate_galois_key_share(p.secret_key, seed, shifts=[1])
            for p in parties
        ]
    )

    encoder = bfv.Encoder(context)
    encryptor = bfv.Encryptor(context, public_key)
    operator = bfv.ArithmeticOperator(context, encoder)
    n = encoder.slot_count

    def decrypt(cipher):
        """Collective decryption: one partial decryption per party."""
        partials = [
            p.manager.decrypt_partial(cipher, p.secret_key) for p in parties
        ]
        return encoder.decode(server.decrypt(partials))

    # ------------------------------------------------------------------
    # 2. Votes: encrypted by each party, added and multiplied by the server
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    votes = [rng.integers(0, 2, n) for _ in range(PARTIES)]
    encrypted = [encryptor.encrypt(encoder.encode(v)) for v in votes]
    tally = encrypted[0]
    for cipher in encrypted[1:]:
        tally = operator.add(tally, cipher)
    # The unanimous votes: the product of the votes of all parties, each
    # product relinearized with the collective key.
    unanimous = encrypted[0]
    for cipher in encrypted[1:]:
        unanimous = operator.multiply(unanimous, cipher)
        operator.relinearize_inplace(unanimous, relin_key)
    # A weighted score with public weights, and the neighbour's score.
    weights = rng.integers(1, 10, n)
    score = operator.multiply_plain(tally, encoder.encode(weights))
    neighbour = operator.rotate_rows(score, galois_key, 1)

    expected_tally = sum(votes)
    expected_score = expected_tally * weights
    assert np.array_equal(decrypt(tally), expected_tally)
    assert np.array_equal(decrypt(unanimous), votes[0] * votes[1] * votes[2])
    assert np.array_equal(decrypt(score), expected_score)
    assert np.array_equal(
        decrypt(neighbour),
        np.roll(expected_score.reshape(2, -1), -1, axis=1).reshape(-1),
    )
    print("tally    ", decrypt(tally)[:8])
    print("unanimous", decrypt(unanimous)[:8])
    print("score    ", decrypt(score)[:8])

    # ------------------------------------------------------------------
    # 3. Collective bootstrapping
    # ------------------------------------------------------------------
    # 12 products with random plaintexts exceed the noise budget of these
    # parameters; a collective bootstrapping after each product keeps the
    # ciphertext decryptable.
    def bootstrap(cipher):
        refresh_seed = bfv.RNGSeed()
        shares = [
            p.manager.distributed_bootstrapping_participant(
                cipher, p.secret_key, refresh_seed
            )
            for p in parties
        ]
        return server.distributed_bootstrapping_coordinator(
            shares, cipher, refresh_seed
        )

    values = rng.integers(0, t, n)
    cipher = encryptor.encrypt(encoder.encode(values))
    unrefreshed = cipher
    expected = values.astype(object)
    for _ in range(12):
        factor = rng.integers(0, t, n)
        plain = encoder.encode(factor)
        cipher = bootstrap(operator.multiply_plain(cipher, plain))
        unrefreshed = operator.multiply_plain(unrefreshed, plain)
        expected = expected * factor.astype(object) % t
    expected = expected.astype(np.uint64)
    assert np.array_equal(decrypt(cipher), expected)
    print("12 products with collective bootstrapping: correct")
    print(
        "12 products without it: correct =",
        np.array_equal(decrypt(unrefreshed), expected),
    )

    print("OK")


if __name__ == "__main__":
    main()
