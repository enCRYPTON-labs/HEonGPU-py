# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

import concurrent.futures
import copy
import io
import pickle

import numpy as np
import pytest

import heongpu as hg
from heongpu import bfv

# The parameters of the multiparty examples of HEonGPU: room for the noise of
# the distributed decryption, and a prime t = 1 (mod 2N) for batching.
POLY_MODULUS_DEGREE = 8192
COEFF_MODULUS_BIT_SIZES = [43, 43, 44, 44]
SPECIAL_MODULUS_BIT_SIZES = [44]
PLAIN_MODULUS = 1032193
# t = 1 (mod 2N) for N = 16384.
PLAIN_MODULUS_LARGE = 786433



def make_context(poly_modulus_degree=POLY_MODULUS_DEGREE):
    if poly_modulus_degree == POLY_MODULUS_DEGREE:
        params = bfv.Params(
            poly_modulus_degree=poly_modulus_degree,
            coeff_modulus_bit_sizes=COEFF_MODULUS_BIT_SIZES,
            special_modulus_bit_sizes=SPECIAL_MODULUS_BIT_SIZES,
            plain_modulus=PLAIN_MODULUS,
        )
    else:
        params = bfv.Params(
            poly_modulus_degree=poly_modulus_degree,
            plain_modulus=PLAIN_MODULUS_LARGE,
            multiplicative_depth=3,
        )
    context = params.make_context()
    context.generate()
    return context


def disjoint_secret_keys(context, count, rng):
    """Ternary secret keys with disjoint supports, and their sum, which is
    ternary too: a secret key of the collective key, to check the protocols
    with a single-key decryptor."""
    n = context.poly_modulus_degree
    positions = rng.permutation(n)
    weight = n // (2 * count)
    coefficients = []
    for i in range(count):
        values = np.zeros(n, dtype=np.int64)
        support = positions[i * weight : (i + 1) * weight]
        values[support] = rng.choice([-1, 1], size=weight)
        coefficients.append(values)
    keys = [bfv.SecretKey.from_coefficients(c, context) for c in coefficients]
    collective = bfv.SecretKey.from_coefficients(sum(coefficients), context)
    return keys, collective


class Party:
    """A participant: its secret key and its own manager."""

    def __init__(self, context, secret_key):
        self.secret_key = secret_key
        self.manager = bfv.MultiPartyManager(context)


class MPCSetup:
    """Participants, a server and the collective keys of the BFV multiparty
    protocols, run as in example/mpc/1_multiparty_computation_bfv.cpp."""

    def __init__(
        self,
        count=3,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        shifts=(1, -2),
        rng=None,
    ):
        rng = np.random.default_rng(7) if rng is None else rng
        self.context = make_context(poly_modulus_degree)
        self.t = self.context.plain_modulus
        self.seed = bfv.RNGSeed()
        self.shifts = list(shifts)

        # Disjoint keys, so that collective_secret_key decrypts the
        # collective ciphertexts (for the noise budget checks).
        keys, self.collective_secret_key = disjoint_secret_keys(
            self.context, count, rng
        )
        self.parties = [Party(self.context, key) for key in keys]
        self.server = bfv.MultiPartyManager(self.context)

        # Collective public key.
        self.public_key_shares = [
            p.manager.generate_public_key_share(p.secret_key, self.seed)
            for p in self.parties
        ]
        self.public_key = self.server.assemble_public_key_share(
            self.public_key_shares
        )

        # Collective relinearization key, in two rounds.
        self.relin_key_round1 = [
            p.manager.generate_relin_key_init(p.secret_key, self.seed)
            for p in self.parties
        ]
        self.relin_key_common = self.server.assemble_relin_key_init(
            self.relin_key_round1
        )
        self.relin_key_round2 = [
            p.manager.generate_relin_key_share(
                self.relin_key_common, p.secret_key
            )
            for p in self.parties
        ]
        self.relin_key = self.server.assemble_relin_key_share(
            self.relin_key_round2, self.relin_key_common
        )

        # Collective Galois key.
        self.galois_key_shares = [
            p.manager.generate_galois_key_share(
                p.secret_key, self.seed, shifts=self.shifts
            )
            for p in self.parties
        ]
        self.galois_key = self.server.assemble_galois_key_share(
            self.galois_key_shares
        )

        self.encoder = bfv.Encoder(self.context)
        self.encryptor = bfv.Encryptor(self.context, self.public_key)
        self.operator = bfv.ArithmeticOperator(self.context, self.encoder)
        self.decryptor = bfv.Decryptor(
            self.context, self.collective_secret_key
        )
        self.slot_count = self.encoder.slot_count

    def encrypt(self, values):
        return self.encryptor.encrypt(self.encoder.encode(values))

    def partials(self, cipher, parties=None):
        parties = self.parties if parties is None else parties
        return [
            p.manager.decrypt_partial(cipher, p.secret_key) for p in parties
        ]

    def decrypt(self, cipher, parties=None):
        """Distributed decryption: every party decrypts partially, and the
        partial decryptions are fused."""
        plain = self.parties[0].manager.decrypt(self.partials(cipher, parties))
        return self.encoder.decode(plain)

    def bootstrap(self, cipher, seed=None):
        seed = bfv.RNGSeed() if seed is None else seed
        shares = [
            p.manager.distributed_bootstrapping_participant(
                cipher, p.secret_key, seed
            )
            for p in self.parties
        ]
        return self.server.distributed_bootstrapping_coordinator(
            shares, cipher, seed
        )

    def random(self, rng):
        return rng.integers(0, self.t, self.slot_count, dtype=np.uint64)


@pytest.fixture(scope="module")
def mpc():
    return MPCSetup()


# Keys are compared by the data after the header.
HEADER_BYTES = 64


def same_key_data(first, second):
    first, second = first.to_bytes(), second.to_bytes()
    return len(first) == len(second) and (
        first[HEADER_BYTES:] == second[HEADER_BYTES:]
    )


def mul_mod(a, b, t):
    return (a.astype(object) * b.astype(object) % t).astype(np.uint64)


def rotate_rows(values, shift):
    """Slot i of each row of N/2 slots gets slot i + shift."""
    return np.roll(values.reshape(2, -1), -shift, axis=1).reshape(-1)


# ----------------------------------------------------------------------
# Seed
# ----------------------------------------------------------------------


def test_seed_random_and_explicit():
    first, second = bfv.RNGSeed(), bfv.RNGSeed()
    assert len(first.key) == 16 and len(first.nonce) == 8
    assert first != second
    seed = bfv.RNGSeed(bytes(range(16)), b"12345678", b"bfv")
    assert seed.key == bytes(range(16))
    assert seed.nonce == b"12345678"
    assert seed.personalization_string == b"bfv"
    assert seed == bfv.RNGSeed(bytes(range(16)), b"12345678", b"bfv")
    assert "RNGSeed" in repr(seed)


def test_seed_copy_and_pickle():
    seed = bfv.RNGSeed()
    assert pickle.loads(pickle.dumps(seed)) == seed
    assert copy.copy(seed) == seed
    assert copy.deepcopy(seed) == seed
    assert seed.copy() == seed


def test_seed_invalid_key():
    with pytest.raises(ValueError):
        bfv.RNGSeed(b"short", b"12345678")


def test_seed_shared_with_ckks():
    from heongpu import ckks

    if hasattr(ckks, "RNGSeed"):
        assert ckks.RNGSeed is bfv.RNGSeed


# ----------------------------------------------------------------------
# Multiparty keys
# ----------------------------------------------------------------------


def test_key_classes(mpc):
    assert isinstance(mpc.public_key_shares[0], bfv.MultipartyPublicKey)
    assert isinstance(mpc.public_key_shares[0], bfv.PublicKey)
    assert type(mpc.public_key) is bfv.PublicKey
    assert isinstance(mpc.relin_key_round1[0], bfv.MultipartyRelinKey)
    assert isinstance(mpc.relin_key_common, bfv.MultipartyRelinKey)
    assert isinstance(mpc.relin_key_round2[0], bfv.RelinKey)
    assert type(mpc.relin_key) is bfv.RelinKey
    assert isinstance(mpc.galois_key_shares[0], bfv.MultipartyGaloisKey)
    assert isinstance(mpc.galois_key_shares[0], bfv.GaloisKey)
    assert type(mpc.galois_key) is bfv.GaloisKey


def test_key_seeds(mpc):
    assert mpc.public_key_shares[1].seed == mpc.seed
    assert mpc.relin_key_round1[2].seed == mpc.seed
    assert mpc.relin_key_common.seed == mpc.seed
    assert mpc.relin_key_round2[0].seed == mpc.seed
    assert mpc.galois_key_shares[0].seed == mpc.seed
    other = bfv.RNGSeed()
    common = mpc.server.assemble_relin_key_init(mpc.relin_key_round1, other)
    assert common.seed == other


def test_key_constructors(mpc):
    seed = bfv.RNGSeed()
    public = bfv.MultipartyPublicKey(mpc.context, seed)
    relin = bfv.MultipartyRelinKey(mpc.context, seed)
    galois = bfv.MultipartyGaloisKey(mpc.context, seed, shifts=[1, 2])
    elements = bfv.MultipartyGaloisKey(mpc.context, seed, galois_elements=[3])
    default = bfv.MultipartyGaloisKey(mpc.context, seed)
    for key in (public, relin, galois, elements, default):
        assert key.seed == seed
    assert public.ring_size == mpc.context.poly_modulus_degree
    with pytest.raises(ValueError):
        bfv.MultipartyGaloisKey(
            mpc.context, seed, shifts=[1], galois_elements=[3]
        )


def test_key_copy(mpc):
    for key in (
        mpc.public_key_shares[0],
        mpc.relin_key_common,
        mpc.galois_key_shares[0],
    ):
        for duplicate in (key.copy(), copy.copy(key), copy.deepcopy(key)):
            assert type(duplicate) is type(key)
            assert duplicate.seed == key.seed
            assert duplicate.to_bytes() == key.to_bytes()


# ----------------------------------------------------------------------
# Three-party computation
# ----------------------------------------------------------------------


def test_collective_public_key(mpc, rng):
    values = mpc.random(rng)
    cipher = mpc.encrypt(values)
    np.testing.assert_array_equal(mpc.decrypt(cipher), values)
    # The collective secret key is the sum of the parties' keys.
    np.testing.assert_array_equal(
        mpc.encoder.decode(mpc.decryptor.decrypt(cipher)), values
    )


def test_distributed_decryption_needs_every_party(mpc, rng):
    values = mpc.random(rng)
    cipher = mpc.encrypt(values)
    partial = mpc.decrypt(cipher, mpc.parties[:2])
    assert not np.array_equal(partial, values)
    # The partial decryptions can be fused by any manager, in any order.
    partials = mpc.partials(cipher)[::-1]
    plain = mpc.server.decrypt(partials)
    np.testing.assert_array_equal(mpc.encoder.decode(plain), values)


def test_partial_decryption_shape(mpc, rng):
    cipher = mpc.encrypt(mpc.random(rng))
    party = mpc.parties[0]
    partial = party.manager.decrypt_partial(cipher, party.secret_key)
    assert isinstance(partial, bfv.Ciphertext)
    assert partial.ring_size == cipher.ring_size
    assert partial.coeff_modulus_count == cipher.coeff_modulus_count


def test_three_party_flow(mpc, rng):
    """Each party encrypts its input with the collective public key; the
    server adds, multiplies by a plaintext and rotates with the collective
    Galois key; the parties decrypt together."""
    a, b, c = (mpc.random(rng) for _ in range(3))
    ca, cb, cc = mpc.encrypt(a), mpc.encrypt(b), mpc.encrypt(c)
    op = mpc.operator
    result = op.add(op.multiply_plain(ca, mpc.encoder.encode(b)), cc)
    result = op.rotate_rows(result, mpc.galois_key, 1)
    expected = rotate_rows((mul_mod(a, b, mpc.t) + c) % mpc.t, 1)
    np.testing.assert_array_equal(mpc.decrypt(result), expected)
    np.testing.assert_array_equal(
        mpc.decrypt(op.sub(ca, cb)), (a.astype(np.int64) - b) % mpc.t
    )


def test_three_party_multiply_relinearize_rotate(mpc, rng):
    """The flow of example/mpc/1_multiparty_computation_bfv.cpp."""
    a, b, c = (mpc.random(rng) for _ in range(3))
    ca, cb, cc = mpc.encrypt(a), mpc.encrypt(b), mpc.encrypt(c)
    op = mpc.operator
    product = op.multiply(ca, cb)
    assert product.relinearization_required
    op.relinearize_inplace(product, mpc.relin_key)
    assert not product.relinearization_required
    result = op.rotate_rows(op.add(product, cc), mpc.galois_key, 1)
    expected = rotate_rows((mul_mod(a, b, mpc.t) + c) % mpc.t, 1)
    np.testing.assert_array_equal(mpc.decrypt(result), expected)


def test_relin_key_protocol_output(mpc, rng):
    """Relinearization with the collective key gives a ciphertext of
    size 2."""
    product = mpc.operator.multiply(
        mpc.encrypt(mpc.random(rng)), mpc.encrypt(mpc.random(rng))
    )
    relinearized = mpc.operator.relinearize(product, mpc.relin_key)
    assert not relinearized.relinearization_required
    assert relinearized.coeff_modulus_count == product.coeff_modulus_count


@pytest.mark.parametrize("shift", [1, -2])
def test_rotation_collective_galois_key(mpc, rng, shift):
    values = mpc.random(rng)
    rotated = mpc.operator.rotate_rows(
        mpc.encrypt(values), mpc.galois_key, shift
    )
    np.testing.assert_array_equal(
        mpc.decrypt(rotated), rotate_rows(values, shift)
    )


def test_column_rotation_collective_galois_key(mpc, rng):
    values = mpc.random(rng)
    swapped = mpc.operator.rotate_columns(mpc.encrypt(values), mpc.galois_key)
    np.testing.assert_array_equal(
        mpc.decrypt(swapped), values.reshape(2, -1)[::-1].reshape(-1)
    )


def test_galois_key_share_for_elements(mpc):
    """Only the keys of rotation steps are combined: shares for explicit
    Galois elements are rejected."""
    shares = [
        p.manager.generate_galois_key_share(
            p.secret_key, mpc.seed, galois_elements=[3]
        )
        for p in mpc.parties
    ]
    assert all(type(share) is bfv.MultipartyGaloisKey for share in shares)
    with pytest.raises(hg.ParameterError, match="explicit Galois elements"):
        mpc.server.assemble_galois_key_share(shares)


# ----------------------------------------------------------------------
# Collective bootstrapping
# ----------------------------------------------------------------------


def test_collective_bootstrapping(mpc, rng):
    """The flow of example/mpc/3_mpc_collective_bootstrapping_bfv.cpp."""
    a, b, c = (mpc.random(rng) for _ in range(3))
    op = mpc.operator
    total = op.add(op.add(mpc.encrypt(a), mpc.encrypt(b)), mpc.encrypt(c))
    refreshed = mpc.bootstrap(total)
    assert isinstance(refreshed, bfv.Ciphertext)
    assert not refreshed.relinearization_required
    expected = (a.astype(object) + b + c) % mpc.t
    np.testing.assert_array_equal(
        mpc.decrypt(refreshed), expected.astype(np.uint64)
    )


def test_collective_bootstrapping_restores_noise_budget(mpc, rng):
    values = mpc.random(rng)
    factor = mpc.encoder.encode(mpc.random(rng))
    fresh = mpc.encrypt(values)
    fresh_budget = mpc.decryptor.remainder_noise_budget(fresh)
    cipher = fresh
    for _ in range(3):
        cipher = mpc.operator.multiply_plain(cipher, factor)
    used_budget = mpc.decryptor.remainder_noise_budget(cipher)
    assert used_budget < fresh_budget - 40
    refreshed = mpc.bootstrap(cipher)
    assert mpc.decryptor.remainder_noise_budget(refreshed) >= fresh_budget - 5
    np.testing.assert_array_equal(mpc.decrypt(refreshed), mpc.decrypt(cipher))


def test_collective_bootstrapping_shares(mpc, rng):
    cipher = mpc.encrypt(mpc.random(rng))
    seed = bfv.RNGSeed()
    party = mpc.parties[1]
    share = party.manager.distributed_bootstrapping_participant(
        cipher, party.secret_key, seed
    )
    assert isinstance(share, bfv.Ciphertext)
    assert share.ring_size == cipher.ring_size


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------


def test_public_key_share_serialization(mpc, tmp_path):
    share = mpc.public_key_shares[0]
    data = share.to_bytes()
    loaded = bfv.MultipartyPublicKey.from_bytes(data, mpc.context, mpc.seed)
    assert type(loaded) is bfv.MultipartyPublicKey
    assert loaded.seed == mpc.seed
    assert loaded.to_bytes() == data

    path = tmp_path / "public_key_share.bin"
    share.save(path)
    from_file = bfv.MultipartyPublicKey.load(path, mpc.context)
    assert from_file.to_bytes() == data

    shares = [loaded] + mpc.public_key_shares[1:]
    collective = mpc.server.assemble_public_key_share(shares)
    assert collective.to_bytes() == mpc.public_key.to_bytes()


def test_relin_key_share_serialization(mpc):
    buffer = io.BytesIO()
    mpc.relin_key_common.save(buffer)
    buffer.seek(0)
    common = bfv.MultipartyRelinKey.load(buffer, mpc.context, mpc.seed)
    assert common.seed == mpc.seed
    assert same_key_data(common, mpc.relin_key_common)

    shares = [
        bfv.MultipartyRelinKey.from_bytes(share.to_bytes(), mpc.context)
        for share in mpc.relin_key_round2
    ]
    collective = mpc.server.assemble_relin_key_share(shares, common)
    assert same_key_data(collective, mpc.relin_key)

    round1 = [
        bfv.MultipartyRelinKey.from_bytes(share.to_bytes(), mpc.context)
        for share in mpc.relin_key_round1
    ]
    again = mpc.server.assemble_relin_key_init(round1, mpc.seed)
    assert same_key_data(again, mpc.relin_key_common)


def test_galois_key_share_serialization(mpc, rng):
    shares = [
        bfv.MultipartyGaloisKey.from_bytes(
            share.to_bytes(), mpc.context, mpc.seed
        )
        for share in mpc.galois_key_shares
    ]
    assert all(type(share) is bfv.MultipartyGaloisKey for share in shares)
    collective = mpc.server.assemble_galois_key_share(shares)
    # The order of the rotation steps in the data can change on loading; the
    # keys are compared by their rotations.
    assert len(collective.to_bytes()) == len(mpc.galois_key.to_bytes())
    values = mpc.random(rng)
    rotated = mpc.operator.rotate_rows(mpc.encrypt(values), collective, -2)
    np.testing.assert_array_equal(
        mpc.decrypt(rotated), rotate_rows(values, -2)
    )


def test_collective_keys_serialization(mpc, rng):
    public_key = bfv.PublicKey.from_bytes(mpc.public_key.to_bytes())
    public_key.set_context(mpc.context)
    galois_key = pickle.loads(pickle.dumps(mpc.galois_key))
    galois_key.set_context(mpc.context)
    values = mpc.random(rng)
    cipher = bfv.Encryptor(mpc.context, public_key).encrypt(
        mpc.encoder.encode(values)
    )
    rotated = mpc.operator.rotate_rows(cipher, galois_key, 1)
    np.testing.assert_array_equal(
        mpc.decrypt(rotated), rotate_rows(values, 1)
    )


def test_partial_decryption_serialization(mpc, rng):
    values = mpc.random(rng)
    partials = [
        bfv.Ciphertext.from_bytes(partial.to_bytes())
        for partial in mpc.partials(mpc.encrypt(values))
    ]
    plain = mpc.server.decrypt(partials)
    np.testing.assert_array_equal(mpc.encoder.decode(plain), values)


@pytest.mark.parametrize(
    "name", ["public_key_shares", "relin_key_round1", "galois_key_shares"]
)
def test_multiparty_keys_not_picklable(mpc, name):
    with pytest.raises(TypeError, match="from_bytes"):
        pickle.dumps(getattr(mpc, name)[0])


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


def test_manager_needs_generated_context():
    params = bfv.Params(poly_modulus_degree=4096, plain_modulus=786433)
    with pytest.raises(hg.StateError):
        bfv.MultiPartyManager(params.make_context())


def test_manager_context(mpc):
    assert mpc.server.context.context_id == mpc.context.context_id


def test_empty_share_lists(mpc):
    server = mpc.server
    with pytest.raises(ValueError):
        server.assemble_public_key_share([])
    with pytest.raises(ValueError):
        server.assemble_relin_key_init([])
    with pytest.raises(ValueError):
        server.assemble_relin_key_share([], mpc.relin_key_common)
    with pytest.raises(ValueError):
        server.assemble_galois_key_share([])
    with pytest.raises(ValueError):
        server.decrypt([])
    with pytest.raises(ValueError):
        server.distributed_bootstrapping_coordinator(
            [], mpc.encrypt(1), mpc.seed
        )


def test_ungenerated_inputs(mpc):
    party = mpc.parties[0]
    with pytest.raises(hg.StateError):
        party.manager.generate_public_key_share(
            bfv.SecretKey(mpc.context), mpc.seed
        )
    with pytest.raises(hg.StateError):
        party.manager.generate_relin_key_share(
            bfv.MultipartyRelinKey(mpc.context, mpc.seed), party.secret_key
        )
    with pytest.raises(ValueError):
        mpc.server.assemble_public_key_share(
            [bfv.MultipartyPublicKey(mpc.context, mpc.seed)]
        )


def test_mismatched_galois_shares(mpc):
    party = mpc.parties[1]
    other = party.manager.generate_galois_key_share(
        party.secret_key, mpc.seed, shifts=[5]
    )
    with pytest.raises(hg.ParameterError, match="same rotation steps"):
        mpc.server.assemble_galois_key_share(
            [mpc.galois_key_shares[0], other]
        )
    with pytest.raises(hg.ParameterError, match="same rotation steps"):
        mpc.server.assemble_galois_key_share(
            [other, mpc.galois_key_shares[0]]
        )
    with pytest.raises(ValueError):
        party.manager.generate_galois_key_share(
            party.secret_key, mpc.seed, shifts=[1], galois_elements=[3]
        )


def test_unrelinearized_ciphertext(mpc, rng):
    product = mpc.operator.multiply(
        mpc.encrypt(mpc.random(rng)), mpc.encrypt(mpc.random(rng))
    )
    party = mpc.parties[0]
    with pytest.raises(hg.StateError):
        party.manager.decrypt_partial(product, party.secret_key)
    with pytest.raises(hg.StateError):
        party.manager.distributed_bootstrapping_participant(
            product, party.secret_key, mpc.seed
        )


def test_context_mismatch(mpc):
    params = bfv.Params(
        poly_modulus_degree=4096, plain_modulus=786433, multiplicative_depth=1
    )
    other = params.make_context()
    other.generate()
    keygen = bfv.KeyGenerator(other)
    secret_key = keygen.secret_key()
    encoder = bfv.Encoder(other)
    cipher = bfv.Encryptor(other, keygen.public_key(secret_key)).encrypt(
        encoder.encode(1)
    )
    party = mpc.parties[0]
    with pytest.raises(hg.ContextMismatchError):
        party.manager.generate_public_key_share(secret_key, mpc.seed)
    with pytest.raises(hg.ContextMismatchError):
        party.manager.generate_galois_key_share(secret_key, mpc.seed)
    with pytest.raises(hg.ContextMismatchError):
        party.manager.decrypt_partial(mpc.encrypt(1), secret_key)
    with pytest.raises(hg.ContextMismatchError):
        party.manager.decrypt_partial(cipher, party.secret_key)
    with pytest.raises(hg.ContextMismatchError):
        mpc.server.decrypt([cipher])
    with pytest.raises(hg.ContextMismatchError):
        party.manager.distributed_bootstrapping_participant(
            cipher, party.secret_key, mpc.seed
        )


# ----------------------------------------------------------------------
# Streams, storage and threads
# ----------------------------------------------------------------------


def test_stream_execution(mpc, rng):
    values = mpc.random(rng)
    with hg.Stream() as stream:
        cipher = mpc.encryptor.encrypt(
            mpc.encoder.encode(values, options=stream), options=stream
        )
        partials = [
            p.manager.decrypt_partial(cipher, p.secret_key, options=stream)
            for p in mpc.parties
        ]
        plain = mpc.server.decrypt(
            partials, options=hg.ExecutionOptions(stream=stream)
        )
        result = mpc.encoder.decode(plain, options=stream)
        seed = bfv.RNGSeed()
        shares = [
            p.manager.distributed_bootstrapping_participant(
                cipher, p.secret_key, seed, options=stream
            )
            for p in mpc.parties
        ]
        refreshed = mpc.server.distributed_bootstrapping_coordinator(
            shares, cipher, seed, options=stream
        )
        stream.synchronize()
    np.testing.assert_array_equal(result, values)
    np.testing.assert_array_equal(mpc.decrypt(refreshed), values)


def test_key_generation_on_stream(mpc, rng):
    seed = bfv.RNGSeed()
    with hg.Stream() as stream:
        shares = [
            p.manager.generate_public_key_share(
                p.secret_key, seed, options=stream
            )
            for p in mpc.parties
        ]
        public_key = mpc.server.assemble_public_key_share(
            shares, options=stream
        )
        galois_shares = [
            p.manager.generate_galois_key_share(
                p.secret_key, seed, shifts=[4], options=stream
            )
            for p in mpc.parties
        ]
        galois_key = mpc.server.assemble_galois_key_share(
            galois_shares, options=stream
        )
        stream.synchronize()
    values = mpc.random(rng)
    cipher = bfv.Encryptor(mpc.context, public_key).encrypt(
        mpc.encoder.encode(values)
    )
    rotated = mpc.operator.rotate_rows(cipher, galois_key, 4)
    np.testing.assert_array_equal(
        mpc.decrypt(rotated), rotate_rows(values, 4)
    )


def test_host_storage(mpc, rng):
    host = hg.ExecutionOptions(storage=hg.StorageType.HOST)
    seed = bfv.RNGSeed()
    shares = [
        p.manager.generate_public_key_share(p.secret_key, seed, options=host)
        for p in mpc.parties
    ]
    assert not any(share.is_on_device for share in shares)
    public_key = mpc.server.assemble_public_key_share(shares)
    assert public_key.is_on_device
    assert not any(share.is_on_device for share in shares)
    shares[0].store_in_device()
    assert shares[0].is_on_device

    values = mpc.random(rng)
    cipher = bfv.Encryptor(mpc.context, public_key).encrypt(
        mpc.encoder.encode(values)
    )
    partials = [
        p.manager.decrypt_partial(cipher, p.secret_key, options=host)
        for p in mpc.parties
    ]
    assert not any(partial.is_on_device for partial in partials)
    plain = mpc.server.decrypt(partials)
    np.testing.assert_array_equal(mpc.encoder.decode(plain), values)
    plain = mpc.server.decrypt(partials, options=host)
    assert not plain.is_on_device
    np.testing.assert_array_equal(mpc.encoder.decode(plain), values)

    shares = [
        p.manager.distributed_bootstrapping_participant(
            cipher, p.secret_key, seed, options=host
        )
        for p in mpc.parties
    ]
    assert not any(share.is_on_device for share in shares)
    refreshed = mpc.server.distributed_bootstrapping_coordinator(
        shares, cipher, seed, options=host
    )
    assert not refreshed.is_on_device
    np.testing.assert_array_equal(mpc.decrypt(refreshed), values)


def test_parties_in_threads(mpc, rng):
    """The GIL is released, so the parties can work in parallel threads."""
    values = [mpc.random(rng) for _ in range(4)]
    ciphers = [mpc.encrypt(v) for v in values]

    def decrypt(cipher):
        with concurrent.futures.ThreadPoolExecutor(len(mpc.parties)) as pool:
            partials = list(
                pool.map(
                    lambda p: p.manager.decrypt_partial(cipher, p.secret_key),
                    mpc.parties,
                )
            )
        return mpc.encoder.decode(mpc.server.decrypt(partials))

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(decrypt, ciphers))
    for result, expected in zip(results, values):
        np.testing.assert_array_equal(result, expected)


def test_bootstrapping_chain(mpc, rng):
    """A chain of plaintext multiplications far beyond the noise budget of
    the parameters, with a collective bootstrapping after each one."""
    values = mpc.random(rng)
    factors = [mpc.random(rng) for _ in range(12)]
    cipher = mpc.encrypt(values)
    unrefreshed = cipher
    expected = values
    for factor in factors:
        plain = mpc.encoder.encode(factor)
        cipher = mpc.bootstrap(mpc.operator.multiply_plain(cipher, plain))
        unrefreshed = mpc.operator.multiply_plain(unrefreshed, plain)
        expected = mul_mod(expected, factor, mpc.t)
    np.testing.assert_array_equal(mpc.decrypt(cipher), expected)
    assert not np.array_equal(mpc.decrypt(unrefreshed), expected)


# ----------------------------------------------------------------------
# Large ring
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("count", [2, 5])
def test_party_counts_large_ring(count):
    rng = np.random.default_rng(count)
    setup = MPCSetup(
        count=count, poly_modulus_degree=16384, shifts=[1], rng=rng
    )
    a, b = setup.random(rng), setup.random(rng)
    op = setup.operator
    result = op.rotate_rows(
        op.add(setup.encrypt(a), setup.encrypt(b)), setup.galois_key, 1
    )
    expected = rotate_rows((a.astype(object) + b) % setup.t, 1)
    np.testing.assert_array_equal(
        setup.decrypt(result), expected.astype(np.uint64)
    )
    refreshed = setup.bootstrap(result)
    np.testing.assert_array_equal(
        setup.decrypt(refreshed), expected.astype(np.uint64)
    )
