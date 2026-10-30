# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

import copy
import io
import pickle
import threading

import numpy as np
import pytest

import heongpu as hg
from heongpu import ckks

TOLERANCE = 1e-4
PARTIES = 3

needs_operator = pytest.mark.skipif(
    not hasattr(ckks, "ArithmeticOperator"),
    reason="ckks.ArithmeticOperator is not available",
)

# HEonGPU 2.1.0 samples only two of the 2 * d error polynomials of the round 2
# relinearization key share (HEMultiPartyManager::generate_relin_key_share);
# the others are uninitialized device memory. The collective key is only
# correct when that memory happens to be zero, as in a fresh memory pool.
relin_key_issue = pytest.mark.xfail(
    reason="HEonGPU 2.1.0: round 2 relinearization key share uses "
    "uninitialized error polynomials",
    raises=AssertionError,
    strict=False,
)

# A serialized relinearization key contains two header fields that the
# default key switching method does not use and HEonGPU 2.1.0 leaves
# uninitialized; keys are compared by the data after the header.
HEADER_BYTES = 64


def same_key_data(first, second):
    first, second = first.to_bytes(), second.to_bytes()
    return len(first) == len(second) and (
        first[HEADER_BYTES:] == second[HEADER_BYTES:]
    )


class Party:
    """One participant: its secret key, encoder and multiparty manager."""

    def __init__(self, context, scale):
        self.secret_key = ckks.KeyGenerator(context).secret_key()
        self.encoder = ckks.Encoder(context)
        self.manager = ckks.MultiPartyManager(context, self.encoder, scale)


class MPCSetup:
    """Participants and a server that build the collective keys following
    the protocol of HEonGPU's multiparty example."""

    def __init__(
        self,
        shifts=(1,),
        parties=PARTIES,
        poly_modulus_degree=8192,
        coeff_modulus_bit_sizes=(60, 50, 50, 50),
        scale_bit_size=50,
    ):
        self.params = ckks.Params(
            poly_modulus_degree=poly_modulus_degree,
            coeff_modulus_bit_sizes=list(coeff_modulus_bit_sizes),
            special_modulus_bit_sizes=[60],
            security_level=hg.SecurityLevel.NONE,
        )
        self.scale = 2.0**scale_bit_size
        self.context = self.params.make_context()
        self.context.generate()
        self.slot_count = self.context.slot_count
        self.shifts = list(shifts)
        self.seed = ckks.RNGSeed()

        self.parties = [Party(self.context, self.scale) for _ in range(parties)]
        self.encoder = ckks.Encoder(self.context)
        self.server = ckks.MultiPartyManager(
            self.context, self.encoder, self.scale
        )

        # Public key.
        self.public_key_shares = [
            p.manager.generate_public_key_share(p.secret_key, self.seed)
            for p in self.parties
        ]
        self.public_key = self.server.assemble_public_key_share(
            self.public_key_shares
        )

        # Relinearization key, round 1.
        self.relin_key_round1 = [
            p.manager.generate_relin_key_init(p.secret_key, self.seed)
            for p in self.parties
        ]
        self.relin_key_common = self.server.assemble_relin_key_init(
            self.relin_key_round1
        )
        # Round 2.
        self.relin_key_round2 = [
            p.manager.generate_relin_key_share(
                self.relin_key_common, p.secret_key
            )
            for p in self.parties
        ]
        self.relin_key = self.server.assemble_relin_key_share(
            self.relin_key_round2, self.relin_key_common
        )

        # Galois key.
        self.galois_key_shares = [
            p.manager.generate_galois_key_share(
                p.secret_key, self.seed, shifts=self.shifts
            )
            for p in self.parties
        ]
        self.galois_key = self.server.assemble_galois_key_share(
            self.galois_key_shares
        )

        self.encryptor = ckks.Encryptor(self.context, self.public_key)

    def encrypt(self, values, options=None):
        plain = self.encoder.encode(values, self.scale, options=options)
        return self.encryptor.encrypt(plain, options=options)

    def decrypt(self, cipher, parties=None, options=None):
        parties = self.parties if parties is None else parties
        partials = [
            p.manager.decrypt_partial(cipher, p.secret_key, options=options)
            for p in parties
        ]
        plain = self.parties[0].manager.decrypt(partials, options=options)
        return self.encoder.decode(plain, options=options)


@pytest.fixture(scope="module")
def mpc():
    return MPCSetup()


def random_vector(rng, size, low=-2.0, high=2.0):
    return rng.uniform(low, high, size)


# ----------------------------------------------------------------------
# RNGSeed
# ----------------------------------------------------------------------


def test_rng_seed_random():
    first, second = ckks.RNGSeed(), ckks.RNGSeed()
    assert len(first.key) == 16
    assert len(first.nonce) == 8
    assert first.personalization_string == b""
    assert first != second
    assert "RNGSeed" in repr(first)


def test_rng_seed_explicit():
    seed = ckks.RNGSeed(bytes(range(16)), b"\x01" * 8, b"heongpu")
    assert seed.key == bytes(range(16))
    assert seed.nonce == b"\x01" * 8
    assert seed.personalization_string == b"heongpu"
    assert seed == ckks.RNGSeed(bytes(range(16)), b"\x01" * 8, b"heongpu")
    assert pickle.loads(pickle.dumps(seed)) == seed
    assert copy.copy(seed) == seed
    assert copy.deepcopy(seed) == seed
    with pytest.raises(ValueError):
        ckks.RNGSeed(b"short", b"\x00" * 8)


# ----------------------------------------------------------------------
# Keys
# ----------------------------------------------------------------------


def test_key_types(mpc):
    assert all(
        isinstance(k, ckks.MultipartyPublicKey) for k in mpc.public_key_shares
    )
    assert isinstance(mpc.public_key_shares[0], ckks.PublicKey)
    assert isinstance(mpc.relin_key_round1[0], ckks.RelinKey)
    assert isinstance(mpc.relin_key_common, ckks.MultipartyRelinKey)
    assert isinstance(mpc.relin_key_round2[0], ckks.MultipartyRelinKey)
    assert isinstance(mpc.galois_key_shares[0], ckks.GaloisKey)
    assert type(mpc.public_key) is ckks.PublicKey
    assert type(mpc.relin_key) is ckks.RelinKey
    assert type(mpc.galois_key) is ckks.GaloisKey
    assert mpc.public_key_shares[0].seed == mpc.seed
    assert mpc.relin_key_common.seed == mpc.seed
    assert mpc.relin_key_round2[1].seed == mpc.seed
    assert mpc.galois_key_shares[2].seed == mpc.seed
    assert mpc.public_key_shares[0].ring_size == 8192
    assert mpc.public_key_shares[0].is_on_device


def test_manager_properties(mpc):
    manager = mpc.parties[0].manager
    assert manager.scale == mpc.scale
    assert manager.context.context_id == mpc.context.context_id


def test_key_constructors(mpc):
    key = ckks.MultipartyPublicKey(mpc.context, mpc.seed)
    assert key.seed == mpc.seed
    relin = ckks.MultipartyRelinKey(mpc.context, mpc.seed)
    assert relin.seed == mpc.seed
    ckks.MultipartyGaloisKey(mpc.context, mpc.seed)
    ckks.MultipartyGaloisKey(mpc.context, mpc.seed, shifts=[1, 2])
    ckks.MultipartyGaloisKey(mpc.context, mpc.seed, galois_elements=[5])
    with pytest.raises(ValueError):
        ckks.MultipartyGaloisKey(
            mpc.context, mpc.seed, shifts=[1], galois_elements=[5]
        )


def test_key_copy(mpc):
    share = mpc.public_key_shares[0]
    for duplicate in (copy.copy(share), copy.deepcopy(share), share.copy()):
        assert type(duplicate) is ckks.MultipartyPublicKey
        assert duplicate.seed == share.seed
        assert duplicate.to_bytes() == share.to_bytes()
    galois = copy.copy(mpc.galois_key_shares[0])
    assert type(galois) is ckks.MultipartyGaloisKey
    relin = mpc.relin_key_common.copy()
    assert type(relin) is ckks.MultipartyRelinKey


# ----------------------------------------------------------------------
# Collective encryption and decryption
# ----------------------------------------------------------------------


def test_collective_encrypt_decrypt(mpc, rng):
    values = random_vector(rng, mpc.slot_count)
    cipher = mpc.encrypt(values)
    np.testing.assert_allclose(mpc.decrypt(cipher), values, atol=TOLERANCE)


def test_partial_decryption_needs_every_party(mpc, rng):
    values = random_vector(rng, mpc.slot_count)
    cipher = mpc.encrypt(values)
    result = mpc.decrypt(cipher, parties=mpc.parties[:2])
    assert not np.allclose(result, values, atol=1e-2)
    single = ckks.Decryptor(mpc.context, mpc.parties[0].secret_key)
    result = mpc.encoder.decode(single.decrypt(cipher))
    assert not np.allclose(result, values, atol=1e-2)


def test_decrypt_partial_properties(mpc, rng):
    cipher = mpc.encrypt(random_vector(rng, mpc.slot_count))
    party = mpc.parties[1]
    partial = party.manager.decrypt_partial(cipher, party.secret_key)
    assert isinstance(partial, ckks.Ciphertext)
    assert partial.size == 2
    assert partial.depth == cipher.depth
    assert partial.scale == cipher.scale


def test_key_generation_with_stream(mpc, rng):
    stream = hg.Stream()
    party = mpc.parties[0]
    share = party.manager.generate_public_key_share(
        party.secret_key, mpc.seed, options=stream
    )
    galois = party.manager.generate_galois_key_share(
        party.secret_key, mpc.seed, shifts=[1], options=stream
    )
    relin = party.manager.generate_relin_key_init(
        party.secret_key, mpc.seed, options=stream
    )
    stream.synchronize()
    assert share.is_on_device and galois.is_on_device and relin.is_on_device


def test_shares_stored_in_host(mpc, rng):
    options = hg.ExecutionOptions(storage=hg.StorageType.HOST)
    shares = [
        p.manager.generate_public_key_share(
            p.secret_key, mpc.seed, options=options
        )
        for p in mpc.parties
    ]
    assert not any(share.is_on_device for share in shares)
    public_key = mpc.server.assemble_public_key_share(shares)
    assert public_key.is_on_device
    assert all(not share.is_on_device for share in shares)
    host_key = mpc.server.assemble_public_key_share(shares, options=options)
    assert not host_key.is_on_device
    assert same_key_data(host_key, public_key)

    values = random_vector(rng, mpc.slot_count)
    encryptor = ckks.Encryptor(mpc.context, public_key)
    cipher = encryptor.encrypt(mpc.encoder.encode(values, mpc.scale))
    np.testing.assert_allclose(mpc.decrypt(cipher), values, atol=TOLERANCE)


def test_galois_shares_stored_in_host(mpc):
    shares = [k.copy() for k in mpc.galois_key_shares]
    for share in shares:
        share.store_in_host()
    galois_key = mpc.server.assemble_galois_key_share(shares)
    assert same_key_data(galois_key, mpc.galois_key)


def test_partial_decryption_threads(mpc, rng):
    values = random_vector(rng, mpc.slot_count)
    cipher = mpc.encrypt(values)
    # The streams below do not wait for the default stream.
    hg.synchronize()
    partials = [None] * len(mpc.parties)
    # The partial decryptions keep using their streams: keep them open.
    streams = [hg.Stream() for _ in mpc.parties]
    errors = []

    def run(index):
        try:
            party = mpc.parties[index]
            partials[index] = party.manager.decrypt_partial(
                cipher, party.secret_key, options=streams[index]
            )
            streams[index].synchronize()
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [
        threading.Thread(target=run, args=(index,))
        for index in range(len(mpc.parties))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    plain = mpc.server.decrypt(partials)
    np.testing.assert_allclose(
        mpc.encoder.decode(plain), values, atol=TOLERANCE
    )


def test_decrypt_with_stream(mpc, rng):
    values = random_vector(rng, mpc.slot_count)
    stream = hg.Stream()
    cipher = mpc.encrypt(values, options=stream)
    result = mpc.decrypt(cipher, options=stream)
    stream.synchronize()
    np.testing.assert_allclose(result, values, atol=TOLERANCE)
    options = hg.ExecutionOptions()
    result = mpc.decrypt(cipher, options=options)
    np.testing.assert_allclose(result, values, atol=TOLERANCE)


# ----------------------------------------------------------------------
# Computation with the collective keys
# ----------------------------------------------------------------------


@needs_operator
@relin_key_issue
def test_multiply_relinearize(mpc, rng):
    a = random_vector(rng, mpc.slot_count)
    b = random_vector(rng, mpc.slot_count)
    c = random_vector(rng, mpc.slot_count)
    operator = ckks.ArithmeticOperator(mpc.context, mpc.encoder)
    product = operator.multiply(mpc.encrypt(a), mpc.encrypt(b))
    assert product.size == 3
    operator.relinearize_inplace(product, mpc.relin_key)
    assert product.size == 2
    operator.rescale_inplace(product)
    third = mpc.encrypt(c)
    operator.mod_drop_inplace(third)
    result = operator.add(product, third)
    np.testing.assert_allclose(mpc.decrypt(result), a * b + c, atol=TOLERANCE)


@needs_operator
def test_rotate(mpc, rng):
    values = random_vector(rng, mpc.slot_count)
    operator = ckks.ArithmeticOperator(mpc.context, mpc.encoder)
    rotated = operator.rotate(mpc.encrypt(values), mpc.galois_key, 1)
    np.testing.assert_allclose(
        mpc.decrypt(rotated), np.roll(values, -1), atol=TOLERANCE
    )


@needs_operator
@relin_key_issue
def test_multiply_rotate_three_inputs(mpc, rng):
    inputs = [random_vector(rng, mpc.slot_count) for _ in range(PARTIES)]
    # Every participant encrypts its own input with the collective key.
    ciphers = [mpc.encrypt(values) for values in inputs]
    operator = ckks.ArithmeticOperator(mpc.context, mpc.encoder)
    product = operator.multiply(ciphers[0], ciphers[1])
    operator.relinearize_inplace(product, mpc.relin_key)
    operator.rescale_inplace(product)
    operator.mod_drop_inplace(ciphers[2])
    total = operator.add(product, ciphers[2])
    rotated = operator.rotate(total, mpc.galois_key, 1)
    expected = np.roll(inputs[0] * inputs[1] + inputs[2], -1)
    np.testing.assert_allclose(mpc.decrypt(rotated), expected, atol=TOLERANCE)


# ----------------------------------------------------------------------
# Collective bootstrapping
# ----------------------------------------------------------------------


@needs_operator
def test_collective_bootstrapping(mpc, rng):
    inputs = [random_vector(rng, mpc.slot_count) for _ in range(PARTIES)]
    operator = ckks.ArithmeticOperator(mpc.context, mpc.encoder)
    total = mpc.encrypt(inputs[0])
    for values in inputs[1:]:
        total = operator.add(total, mpc.encrypt(values))
    operator.mod_drop_inplace(total)
    operator.mod_drop_inplace(total)
    assert total.depth == 2

    seed = ckks.RNGSeed()
    shares = [
        p.manager.distributed_bootstrapping_participant(
            total, p.secret_key, seed
        )
        for p in mpc.parties
    ]
    refreshed = mpc.server.distributed_bootstrapping_coordinator(
        shares, total, seed
    )
    assert refreshed.depth == 0
    assert refreshed.scale == mpc.scale
    expected = sum(inputs)
    np.testing.assert_allclose(mpc.decrypt(refreshed), expected, atol=1e-3)

    # The refreshed ciphertext is at the top level again.
    rotated = operator.rotate(refreshed, mpc.galois_key, 1)
    np.testing.assert_allclose(
        mpc.decrypt(rotated), np.roll(expected, -1), atol=1e-3
    )


@needs_operator
@relin_key_issue
def test_multiply_after_collective_bootstrapping(mpc, rng):
    values = random_vector(rng, mpc.slot_count, -1.0, 1.0)
    operator = ckks.ArithmeticOperator(mpc.context, mpc.encoder)
    cipher = mpc.encrypt(values)
    for _ in range(2):
        operator.mod_drop_inplace(cipher)
    seed = ckks.RNGSeed()
    shares = [
        p.manager.distributed_bootstrapping_participant(
            cipher, p.secret_key, seed
        )
        for p in mpc.parties
    ]
    refreshed = mpc.server.distributed_bootstrapping_coordinator(
        shares, cipher, seed
    )
    square = operator.multiply(refreshed, refreshed)
    operator.relinearize_inplace(square, mpc.relin_key)
    operator.rescale_inplace(square)
    np.testing.assert_allclose(mpc.decrypt(square), values * values, atol=1e-3)


def test_collective_bootstrapping_fresh(mpc, rng):
    values = random_vector(rng, mpc.slot_count)
    seed = ckks.RNGSeed()
    stream = hg.Stream()
    cipher = mpc.encrypt(values, options=stream)
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
    assert refreshed.depth == 0
    np.testing.assert_allclose(mpc.decrypt(refreshed), values, atol=1e-3)


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------


def test_public_key_share_serialization(mpc, rng, tmp_path):
    shares = []
    for index, share in enumerate(mpc.public_key_shares):
        data = share.to_bytes()
        assert isinstance(data, bytes)
        if index == 0:
            loaded = ckks.MultipartyPublicKey.from_bytes(
                data, mpc.context, mpc.seed
            )
            assert loaded.seed == mpc.seed
        elif index == 1:
            path = tmp_path / "share.bin"
            share.save(path)
            loaded = ckks.MultipartyPublicKey.load(path, mpc.context)
        else:
            buffer = io.BytesIO()
            share.save(buffer)
            buffer.seek(0)
            loaded = ckks.MultipartyPublicKey.load(buffer, mpc.context)
        assert type(loaded) is ckks.MultipartyPublicKey
        assert loaded.to_bytes() == data
        shares.append(loaded)
    public_key = mpc.server.assemble_public_key_share(shares)
    assert same_key_data(public_key, mpc.public_key)

    values = random_vector(rng, mpc.slot_count)
    encryptor = ckks.Encryptor(mpc.context, public_key)
    cipher = encryptor.encrypt(mpc.encoder.encode(values, mpc.scale))
    np.testing.assert_allclose(mpc.decrypt(cipher), values, atol=TOLERANCE)


def test_relin_key_serialization(mpc):
    round1 = [
        ckks.MultipartyRelinKey.from_bytes(k.to_bytes(), mpc.context)
        for k in mpc.relin_key_round1
    ]
    common = mpc.server.assemble_relin_key_init(round1, mpc.seed)
    assert same_key_data(common, mpc.relin_key_common)
    common = ckks.MultipartyRelinKey.from_bytes(
        common.to_bytes(), mpc.context, mpc.seed
    )
    round2 = [
        ckks.MultipartyRelinKey.from_bytes(k.to_bytes(), mpc.context)
        for k in mpc.relin_key_round2
    ]
    relin_key = mpc.server.assemble_relin_key_share(round2, common)
    assert same_key_data(relin_key, mpc.relin_key)
    # The collective key is a plain RelinKey.
    restored = ckks.RelinKey.from_bytes(relin_key.to_bytes())
    assert type(restored) is ckks.RelinKey


def test_galois_key_serialization(mpc):
    shares = [
        ckks.MultipartyGaloisKey.from_bytes(k.to_bytes(), mpc.context)
        for k in mpc.galois_key_shares
    ]
    assert all(type(k) is ckks.MultipartyGaloisKey for k in shares)
    galois_key = mpc.server.assemble_galois_key_share(shares)
    assert same_key_data(galois_key, mpc.galois_key)


def test_multiparty_key_pickle_unsupported(mpc):
    for key in (
        mpc.public_key_shares[0],
        mpc.relin_key_common,
        mpc.galois_key_shares[0],
    ):
        with pytest.raises(TypeError):
            pickle.dumps(key)
    # The collective keys are ordinary keys and can be pickled.
    restored = pickle.loads(pickle.dumps(mpc.public_key))
    assert restored.to_bytes() == mpc.public_key.to_bytes()


def test_multiparty_key_load_wrong_context(mpc, ckks_setup):
    data = mpc.public_key_shares[0].to_bytes()
    with pytest.raises(hg.Error):
        ckks.MultipartyPublicKey.from_bytes(data, ckks_setup.context)
    with pytest.raises(hg.SerializationError):
        ckks.MultipartyPublicKey.from_bytes(b"\x00" * 16, mpc.context)


def test_shares_on_host(mpc):
    shares = [k.copy() for k in mpc.public_key_shares]
    for share in shares:
        share.store_in_host()
        assert not share.is_on_device
    public_key = mpc.server.assemble_public_key_share(shares)
    assert same_key_data(public_key, mpc.public_key)


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


def test_empty_share_lists(mpc):
    with pytest.raises(ValueError):
        mpc.server.assemble_public_key_share([])
    with pytest.raises(ValueError):
        mpc.server.assemble_relin_key_init([])
    with pytest.raises(ValueError):
        mpc.server.assemble_galois_key_share([])
    with pytest.raises(ValueError):
        mpc.server.decrypt([])
    cipher = mpc.encrypt(1.0)
    with pytest.raises(ValueError):
        mpc.server.distributed_bootstrapping_coordinator([], cipher, mpc.seed)


def test_wrong_share_types(mpc):
    with pytest.raises(TypeError):
        mpc.server.assemble_public_key_share([mpc.public_key])
    with pytest.raises(TypeError):
        mpc.server.assemble_galois_key_share([mpc.galois_key])
    with pytest.raises(TypeError):
        mpc.server.assemble_public_key_share([mpc.public_key_shares[0], None])
    with pytest.raises(TypeError):
        mpc.server.decrypt([None])


def test_galois_share_arguments(mpc):
    party = mpc.parties[0]
    with pytest.raises(ValueError):
        party.manager.generate_galois_key_share(
            party.secret_key, mpc.seed, shifts=[1], galois_elements=[5]
        )


@needs_operator
def test_partials_at_different_levels(mpc, rng):
    party = mpc.parties[0]
    cipher = mpc.encrypt(random_vector(rng, mpc.slot_count))
    deeper = cipher.copy()
    operator = ckks.ArithmeticOperator(mpc.context, mpc.encoder)
    operator.mod_drop_inplace(deeper)
    partials = [
        party.manager.decrypt_partial(cipher, party.secret_key),
        party.manager.decrypt_partial(deeper, party.secret_key),
    ]
    with pytest.raises(ValueError):
        mpc.server.decrypt(partials)


def test_other_context(mpc, ckks_setup):
    party = mpc.parties[0]
    cipher = ckks_setup.encrypt(1.0)
    with pytest.raises(hg.Error):
        party.manager.decrypt_partial(cipher, party.secret_key)
    with pytest.raises(hg.Error):
        party.manager.generate_public_key_share(ckks_setup.secret_key, mpc.seed)


# ----------------------------------------------------------------------
# Larger setting
# ----------------------------------------------------------------------


@pytest.mark.slow
@needs_operator
def test_five_parties_large_ring(rng):
    setup = MPCSetup(
        shifts=(1, -3),
        parties=5,
        poly_modulus_degree=16384,
        coeff_modulus_bit_sizes=(60, 45, 45, 45, 45, 45),
        scale_bit_size=45,
    )
    inputs = [random_vector(rng, setup.slot_count) for _ in setup.parties]
    operator = ckks.ArithmeticOperator(setup.context, setup.encoder)
    total = setup.encrypt(inputs[0])
    for values in inputs[1:]:
        total = operator.add(total, setup.encrypt(values))
    expected = sum(inputs)
    np.testing.assert_allclose(setup.decrypt(total), expected, atol=1e-3)

    rotated = operator.rotate(total, setup.galois_key, -3)
    np.testing.assert_allclose(
        setup.decrypt(rotated), np.roll(expected, 3), atol=1e-3
    )

    for _ in range(4):
        operator.mod_drop_inplace(total)
    seed = ckks.RNGSeed()
    shares = [
        p.manager.distributed_bootstrapping_participant(
            total, p.secret_key, seed
        )
        for p in setup.parties
    ]
    refreshed = setup.server.distributed_bootstrapping_coordinator(
        shares, total, seed
    )
    assert refreshed.depth == 0
    np.testing.assert_allclose(setup.decrypt(refreshed), expected, atol=1e-3)
