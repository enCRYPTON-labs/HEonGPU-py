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

# A prime with t = 1 (mod 2N) for N up to 32768, so that it allows batching.
PLAIN_MODULUS = 786433


class BFVSetup:
    """A BFV context with its keys and helpers."""

    def __init__(self, depth=2, poly_modulus_degree=8192):
        self.params = bfv.Params(
            multiplicative_depth=depth,
            plain_modulus=PLAIN_MODULUS,
            poly_modulus_degree=poly_modulus_degree,
        )
        self.context = self.params.make_context()
        self.context.generate()
        self.t = self.context.plain_modulus
        self.keygen = bfv.KeyGenerator(self.context)
        self.secret_key = self.keygen.secret_key()
        self.public_key = self.keygen.public_key(self.secret_key)
        self.relin_key = self.keygen.relin_key(self.secret_key)
        self.galois_key = self.keygen.galois_key(self.secret_key, [1, 2, 3])
        self.switch_key = bfv.SwitchKey(self.context)
        self.keygen.generate_switch_key(
            self.switch_key, self.keygen.secret_key(), self.secret_key
        )
        self.encoder = bfv.Encoder(self.context)
        self.encryptor = bfv.Encryptor(self.context, self.public_key)
        self.decryptor = bfv.Decryptor(self.context, self.secret_key)
        self.slot_count = self.encoder.slot_count

    def encrypt(self, values):
        return self.encryptor.encrypt(self.encoder.encode(values))

    def decrypt(self, cipher, signed=False):
        return self.encoder.decode(
            self.decryptor.decrypt(cipher), signed=signed
        )

    def random(self, rng, size=None):
        size = self.slot_count if size is None else size
        return rng.integers(0, self.t, size, dtype=np.uint64)


@pytest.fixture(scope="module")
def setup():
    return BFVSetup()


def centered(values, t):
    """The representatives of values modulo t in (-t/2, t/2]."""
    values = np.mod(np.asarray(values, dtype=object), t)
    return np.where(values > t // 2, values - t, values).astype(np.int64)


# ----------------------------------------------------------------------
# Parameters and context
# ----------------------------------------------------------------------


def test_params_generate():
    params = bfv.Params(multiplicative_depth=2, plain_modulus_bit_size=20)
    parameters = params.generate()
    assert isinstance(parameters, bfv.ParameterSet)
    assert parameters.poly_modulus_degree >= 4096
    assert parameters.plain_modulus.bit_length() == 20
    assert parameters.plain_modulus % (2 * parameters.poly_modulus_degree) == 1
    assert len(parameters.coeff_modulus) >= 1
    assert len(parameters.special_modulus) >= 1
    assert parameters.total_bit_count <= parameters.max_bit_count
    assert parameters.digit_count >= 1
    assert parameters.security_level == hg.SecurityLevel.SEC128
    assert "BFV" in str(parameters)
    assert repr(parameters) == str(parameters)


def test_params_fields():
    params = bfv.Params()
    assert params.multiplicative_depth == 1
    assert params.plain_modulus == 0
    assert params.plain_modulus_bit_size == 20
    assert params.poly_modulus_degree == 0
    params.multiplicative_depth = 3
    params.plain_modulus = PLAIN_MODULUS
    params.coeff_modulus_bit_sizes = [50, 50]
    assert params.multiplicative_depth == 3
    assert params.plain_modulus == PLAIN_MODULUS
    assert params.coeff_modulus_bit_sizes == [50, 50]


def test_params_explicit_moduli():
    params = bfv.Params(
        poly_modulus_degree=8192,
        plain_modulus=PLAIN_MODULUS,
        coeff_modulus_bit_sizes=[54, 54],
        special_modulus_bit_sizes=[55],
    )
    parameters = params.generate()
    assert [q.bit_length() for q in parameters.coeff_modulus] == [54, 54]
    assert parameters.plain_modulus == PLAIN_MODULUS
    context = parameters.make_context()
    context.generate()
    assert context.poly_modulus_degree == 8192
    assert context.log_poly_modulus_degree == 13
    assert context.ciphertext_modulus_count == 2
    assert context.key_modulus_count == 3
    assert context.key_modulus[:2] == parameters.coeff_modulus
    assert context.plain_modulus == PLAIN_MODULUS


def test_parameter_set_apply():
    parameters = bfv.Params(
        multiplicative_depth=1, plain_modulus=PLAIN_MODULUS
    ).generate()
    context = bfv.Context()
    parameters.apply(context)
    context.generate()
    assert context.poly_modulus_degree == parameters.poly_modulus_degree
    assert context.plain_modulus == PLAIN_MODULUS


def test_params_too_deep():
    params = bfv.Params(multiplicative_depth=100, poly_modulus_degree=8192)
    with pytest.raises(hg.ParameterError):
        params.generate()


def test_parameter_set_wrong_security_level():
    parameters = bfv.Params(plain_modulus=PLAIN_MODULUS).generate()
    with pytest.raises(hg.ParameterError):
        parameters.apply(bfv.Context(hg.SecurityLevel.SEC256))


def test_context_setters():
    context = bfv.Context(hg.SecurityLevel.SEC128)
    context.set_poly_modulus_degree(8192)
    context.set_coeff_modulus_bit_sizes([54, 54], [55])
    context.set_plain_modulus(PLAIN_MODULUS)
    assert "not generated" in str(context)
    context.generate()
    assert context.slot_count == 8192
    assert context.plain_modulus == PLAIN_MODULUS
    assert context.security_level == hg.SecurityLevel.SEC128
    assert len(context.key_modulus) == 3
    assert context.context_id != 0
    assert "heongpu.bfv.Context" in repr(context)
    assert f"t={PLAIN_MODULUS}" in repr(context)
    assert "BFV" in str(context)


def test_context_coeff_modulus_values():
    reference = bfv.Params(
        poly_modulus_degree=8192,
        plain_modulus=PLAIN_MODULUS,
        coeff_modulus_bit_sizes=[54, 54],
        special_modulus_bit_sizes=[55],
    ).generate()
    context = bfv.Context()
    context.set_poly_modulus_degree(8192)
    context.set_coeff_modulus_values(
        np.array(reference.coeff_modulus, dtype=np.uint64),
        reference.special_modulus,
    )
    context.set_plain_modulus(PLAIN_MODULUS)
    context.generate()
    assert context.key_modulus == (
        reference.coeff_modulus + reference.special_modulus
    )


def test_context_default_values():
    context = bfv.Context()
    context.set_poly_modulus_degree(8192)
    context.set_coeff_modulus_default_values(1)
    context.set_plain_modulus(PLAIN_MODULUS)
    context.generate()
    assert context.key_modulus_count == context.ciphertext_modulus_count + 1


def test_context_plain_modulus_fixed():
    context = bfv.Context()
    context.set_plain_modulus(PLAIN_MODULUS)
    with pytest.raises(RuntimeError):
        context.set_plain_modulus(65537)


def test_context_incomplete_generate(setup):
    context = bfv.Context()
    context.set_poly_modulus_degree(8192)
    with pytest.raises(Exception):
        context.generate()
    with pytest.raises(hg.StateError):
        bfv.Ciphertext(context)


def test_context_serialization(setup, tmp_path):
    context = setup.context
    data = context.to_bytes()
    assert isinstance(data, bytes)
    loaded = bfv.Context.from_bytes(data)
    assert loaded.context_id == context.context_id
    assert loaded.plain_modulus == context.plain_modulus
    buffer = io.BytesIO()
    context.save(buffer)
    buffer.seek(0)
    assert bfv.Context.load(buffer).context_id == context.context_id
    path = tmp_path / "context.bin"
    context.save(path)
    assert bfv.Context.load(path).context_id == context.context_id
    assert pickle.loads(pickle.dumps(context)).context_id == context.context_id


def test_loaded_context_works_with_keys(setup, rng):
    context = bfv.Context.from_bytes(setup.context.to_bytes())
    values = setup.random(rng)
    cipher = setup.encrypt(values)
    decryptor = bfv.Decryptor(context, setup.secret_key)
    encoder = bfv.Encoder(context)
    np.testing.assert_array_equal(
        encoder.decode(decryptor.decrypt(cipher)), values
    )


def test_print_parameters(setup, capsys):
    setup.context.print_parameters()
    output = capsys.readouterr().out
    assert "BFV" in output
    assert str(PLAIN_MODULUS) in output


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------


def test_encoder_properties(setup):
    assert setup.encoder.slot_count == setup.context.poly_modulus_degree
    assert setup.encoder.row_size == setup.encoder.slot_count // 2
    assert setup.encoder.plain_modulus == PLAIN_MODULUS


def test_encode_decode_exact(setup, rng):
    values = setup.random(rng)
    plain = setup.encoder.encode(values)
    assert isinstance(plain, bfv.Plaintext)
    assert plain.size == setup.slot_count
    assert not plain.in_ntt_domain
    result = setup.encoder.decode(plain)
    assert result.dtype == np.uint64
    assert result.shape == (setup.slot_count,)
    np.testing.assert_array_equal(result, values)


def test_encode_negative_values(setup, rng):
    t = setup.t
    values = rng.integers(-(t // 2), t // 2, setup.slot_count)
    plain = setup.encoder.encode(values)
    signed = setup.encoder.decode(plain, signed=True)
    assert signed.dtype == np.int64
    np.testing.assert_array_equal(signed, values)
    np.testing.assert_array_equal(
        setup.encoder.decode(plain), np.mod(values, t).astype(np.uint64)
    )


def test_encode_plain_modulus_wrap(setup):
    t = setup.t
    values = [t, t + 5, -1, -t - 1, 3 * t + 7, 2**62, -(2**62)]
    expected = [v % t for v in values]
    result = setup.encoder.decode(setup.encoder.encode(values))
    np.testing.assert_array_equal(result[: len(values)], expected)
    np.testing.assert_array_equal(result[len(values) :], 0)
    # Unsigned values above 2^63 and Python ints of any size.
    big = np.array([2**64 - 1, 2**63 + 1], dtype=np.uint64)
    result = setup.encoder.decode(setup.encoder.encode(big))
    np.testing.assert_array_equal(
        result[:2], [(2**64 - 1) % t, (2**63 + 1) % t]
    )
    huge = [2**100 + 3, -(2**90)]
    result = setup.encoder.decode(setup.encoder.encode(huge))
    np.testing.assert_array_equal(result[:2], [v % t for v in huge])


def test_encode_centered_boundary(setup):
    t = setup.t
    values = [t // 2, t // 2 + 1, -(t // 2), 0, 1, -1]
    result = setup.encoder.decode(setup.encoder.encode(values), signed=True)
    np.testing.assert_array_equal(result[:6], centered(values, t))


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.uint8, np.uint32, np.int64]
)
def test_encode_dtypes(setup, dtype):
    values = np.arange(-5, 5).astype(dtype)
    result = setup.encoder.decode(setup.encoder.encode(values))
    expected = np.mod(values.astype(np.int64), setup.t).astype(np.uint64)
    np.testing.assert_array_equal(result[:10], expected)


def test_encode_bool_list_scalar_empty(setup):
    result = setup.encoder.decode(setup.encoder.encode([True, False, True]))
    np.testing.assert_array_equal(result[:4], [1, 0, 1, 0])
    result = setup.encoder.decode(setup.encoder.encode(7))
    np.testing.assert_array_equal(result, 7)
    result = setup.encoder.decode(setup.encoder.encode(-2), signed=True)
    np.testing.assert_array_equal(result, -2)
    result = setup.encoder.decode(setup.encoder.encode([]))
    np.testing.assert_array_equal(result, 0)


def test_encode_rejects_bad_input(setup):
    with pytest.raises(TypeError):
        setup.encoder.encode([1.5, 2.0])
    with pytest.raises(TypeError):
        setup.encoder.encode(np.zeros(4))
    with pytest.raises(TypeError):
        setup.encoder.encode("text")
    with pytest.raises(TypeError):
        setup.encoder.encode([1, 2.5, 2**70])
    with pytest.raises(ValueError):
        setup.encoder.encode(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError):
        setup.encoder.encode(np.zeros(setup.slot_count + 1, dtype=np.int64))


def test_plaintext_properties(setup, rng):
    plain = setup.encoder.encode(setup.random(rng, 16))
    assert plain.is_on_device
    assert "Plaintext" in repr(plain)
    data = plain.get_data()
    assert data.dtype == np.uint64
    assert data.shape == (plain.size,)
    assert np.all(data < setup.t)
    empty = bfv.Plaintext(setup.context)
    assert empty.size == setup.slot_count


# ----------------------------------------------------------------------
# Encryption and decryption
# ----------------------------------------------------------------------


def test_encrypt_decrypt(setup, rng):
    values = setup.random(rng)
    cipher = setup.encrypt(values)
    assert isinstance(cipher, bfv.Ciphertext)
    assert cipher.size == 2
    assert cipher.ring_size == setup.slot_count
    assert cipher.coeff_modulus_count == setup.context.ciphertext_modulus_count
    assert not cipher.in_ntt_domain
    assert not cipher.relinearization_required
    assert cipher.is_on_device
    assert "Ciphertext" in repr(cipher)
    np.testing.assert_array_equal(setup.decrypt(cipher), values)


def test_encrypt_decrypt_signed(setup, rng):
    t = setup.t
    values = rng.integers(-(t // 2), t // 2, setup.slot_count)
    np.testing.assert_array_equal(
        setup.decrypt(setup.encrypt(values), signed=True), values
    )


def test_noise_budget(setup, rng):
    cipher = setup.encrypt(setup.random(rng))
    budget = setup.decryptor.remainder_noise_budget(cipher)
    assert isinstance(budget, int)
    assert budget > 20


def test_empty_ciphertext(setup):
    cipher = bfv.Ciphertext(setup.context)
    assert cipher.size == 2
    assert cipher.ring_size == setup.slot_count


def test_ciphertext_raw_data(setup, rng):
    cipher = setup.encrypt(setup.random(rng, 4))
    data = cipher.get_data()
    assert data.dtype == np.uint64
    assert data.shape == (
        cipher.size,
        cipher.coeff_modulus_count,
        cipher.ring_size,
    )
    moduli = np.array(
        setup.context.key_modulus[: cipher.coeff_modulus_count], dtype=np.uint64
    )
    assert np.all(data < moduli[None, :, None])


def test_context_mismatch(setup, rng):
    other = bfv.Params(
        poly_modulus_degree=8192,
        plain_modulus=PLAIN_MODULUS,
        coeff_modulus_bit_sizes=[50, 50],
        special_modulus_bit_sizes=[51],
    ).make_context()
    other.generate()
    assert other.context_id != setup.context.context_id
    decryptor = bfv.Decryptor(other, bfv.KeyGenerator(other).secret_key())
    with pytest.raises(hg.ContextMismatchError):
        decryptor.decrypt(setup.encrypt([1, 2, 3]))
    with pytest.raises(hg.ContextMismatchError):
        bfv.Encoder(other).decode(setup.encoder.encode([1]))
    with pytest.raises(hg.ContextMismatchError):
        bfv.Encryptor(other, setup.public_key)


# ----------------------------------------------------------------------
# Keys
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["secret_key", "public_key", "relin_key", "galois_key", "switch_key"],
)
def test_key_serialization(setup, name, tmp_path):
    key = getattr(setup, name)
    data = key.to_bytes()
    assert isinstance(data, bytes) and len(data) > 0

    # The Galois keys are kept in a hash map, so their order in the data can
    # change after loading.
    def same(other):
        assert type(other) is type(key)
        other_data = other.to_bytes()
        if name == "galois_key":
            return len(other_data) == len(data)
        return other_data == data

    assert same(type(key).from_bytes(data))
    path = tmp_path / f"{name}.bin"
    key.save(path)
    assert same(type(key).load(str(path)))
    buffer = io.BytesIO()
    key.save(buffer)
    buffer.seek(0)
    assert same(type(key).load(buffer))
    assert same(pickle.loads(pickle.dumps(key)))
    assert same(copy.copy(key))
    assert same(copy.deepcopy(key))
    assert same(key.copy())


def test_loaded_keys_work(setup, rng):
    secret_key = bfv.SecretKey.from_bytes(setup.secret_key.to_bytes())
    public_key = bfv.PublicKey.from_bytes(setup.public_key.to_bytes())
    secret_key.set_context(setup.context)
    public_key.set_context(setup.context)
    assert secret_key.ring_size == setup.slot_count
    assert public_key.ring_size == setup.slot_count
    assert public_key.coeff_modulus_count == setup.context.key_modulus_count
    encryptor = bfv.Encryptor(setup.context, public_key)
    decryptor = bfv.Decryptor(setup.context, secret_key)
    values = setup.random(rng)
    cipher = encryptor.encrypt(setup.encoder.encode(values))
    np.testing.assert_array_equal(
        setup.encoder.decode(decryptor.decrypt(cipher)), values
    )


def test_generate_into_existing_keys(setup, rng):
    keygen = setup.keygen
    secret_key = bfv.SecretKey(setup.context)
    keygen.generate_secret_key(secret_key)
    public_key = bfv.PublicKey(setup.context)
    keygen.generate_public_key(public_key, secret_key)
    relin_key = bfv.RelinKey(setup.context)
    keygen.generate_relin_key(relin_key, secret_key)
    galois_key = bfv.GaloisKey(setup.context, shifts=[1])
    keygen.generate_galois_key(galois_key, secret_key)
    for key in (secret_key, public_key, relin_key, galois_key):
        assert key.is_on_device
    values = setup.random(rng)
    cipher = bfv.Encryptor(setup.context, public_key).encrypt(
        setup.encoder.encode(values)
    )
    plain = bfv.Decryptor(setup.context, secret_key).decrypt(cipher)
    np.testing.assert_array_equal(setup.encoder.decode(plain), values)


def test_sparse_secret_key(setup, rng):
    key = bfv.SecretKey(setup.context, 64)
    setup.keygen.generate_secret_key(key)
    assert key.ring_size == setup.slot_count
    assert key.coeff_modulus_count == setup.context.key_modulus_count
    public_key = setup.keygen.public_key(key)
    values = setup.random(rng)
    cipher = bfv.Encryptor(setup.context, public_key).encrypt(
        setup.encoder.encode(values)
    )
    plain = bfv.Decryptor(setup.context, key).decrypt(cipher)
    np.testing.assert_array_equal(setup.encoder.decode(plain), values)


def test_secret_key_from_coefficients(setup, rng):
    coefficients = rng.integers(-1, 2, setup.slot_count)
    key = bfv.SecretKey.from_coefficients(coefficients, setup.context)
    assert key.ring_size == setup.slot_count
    public_key = setup.keygen.public_key(key)
    values = setup.random(rng)
    cipher = bfv.Encryptor(setup.context, public_key).encrypt(
        setup.encoder.encode(values)
    )
    plain = bfv.Decryptor(setup.context, key).decrypt(cipher)
    np.testing.assert_array_equal(setup.encoder.decode(plain), values)
    again = bfv.SecretKey.from_coefficients(coefficients, setup.context)
    assert again.to_bytes() == key.to_bytes()


def test_galois_key_constructors(setup):
    bfv.GaloisKey(setup.context)
    bfv.GaloisKey(setup.context, shifts=[1, -2, 5])
    bfv.GaloisKey(setup.context, max_shift=4)
    bfv.GaloisKey(setup.context, galois_elements=[3, 9])
    with pytest.raises(ValueError):
        bfv.GaloisKey(setup.context, shifts=[1], max_shift=8)
    with pytest.raises(hg.ParameterError):
        bfv.GaloisKey(setup.context, max_shift=-1)


def test_keygen_galois_shifts(setup):
    key = setup.keygen.galois_key(setup.secret_key, [4, -4])
    assert key.is_on_device
    assert len(key.to_bytes()) > 0


def test_ungenerated_key_rejected(setup):
    with pytest.raises(hg.StateError):
        bfv.Decryptor(setup.context, bfv.SecretKey(setup.context))


# ----------------------------------------------------------------------
# Serialization of plaintexts and ciphertexts
# ----------------------------------------------------------------------


def test_ciphertext_serialization(setup, rng, tmp_path):
    values = setup.random(rng)
    cipher = setup.encrypt(values)
    for loaded in (
        bfv.Ciphertext.from_bytes(cipher.to_bytes()),
        pickle.loads(pickle.dumps(cipher)),
        copy.copy(cipher),
        copy.deepcopy(cipher),
        cipher.copy(),
    ):
        assert isinstance(loaded, bfv.Ciphertext)
        np.testing.assert_array_equal(setup.decrypt(loaded), values)
    path = tmp_path / "cipher.bin"
    with open(path, "wb") as file:
        cipher.save(file)
    with open(path, "rb") as file:
        loaded = bfv.Ciphertext.load(file)
    np.testing.assert_array_equal(setup.decrypt(loaded), values)
    assert bfv.Ciphertext.load(path).to_bytes() == cipher.to_bytes()


def test_copy_is_independent(setup, rng):
    values = setup.random(rng)
    cipher = setup.encrypt(values)
    duplicate = cipher.copy()
    operator = bfv.ArithmeticOperator(setup.context, setup.encoder)
    operator.negate_inplace(cipher)
    np.testing.assert_array_equal(setup.decrypt(duplicate), values)


def test_plaintext_serialization(setup, rng, tmp_path):
    values = setup.random(rng)
    plain = setup.encoder.encode(values)
    for loaded in (
        bfv.Plaintext.from_bytes(plain.to_bytes()),
        pickle.loads(pickle.dumps(plain)),
        copy.deepcopy(plain),
        plain.copy(),
    ):
        assert loaded.to_bytes() == plain.to_bytes()
        np.testing.assert_array_equal(setup.encoder.decode(loaded), values)
    path = tmp_path / "plain.bin"
    plain.save(path)
    np.testing.assert_array_equal(
        setup.encoder.decode(bfv.Plaintext.load(path)), values
    )


def test_serialization_error():
    # Some malformed inputs are reported as plain RuntimeError by HEonGPU.
    with pytest.raises((hg.SerializationError, RuntimeError)):
        bfv.Ciphertext.from_bytes(b"not a ciphertext")
    with pytest.raises((hg.SerializationError, RuntimeError)):
        bfv.SecretKey.from_bytes(b"")


def test_wrong_scheme_rejected(setup):
    with pytest.raises((hg.SerializationError, RuntimeError)):
        hg.ckks.Ciphertext.from_bytes(setup.encrypt([1]).to_bytes())


def test_text_file_rejected(setup, tmp_path):
    path = tmp_path / "cipher.bin"
    setup.encrypt([1]).save(path)
    with open(path, "r", encoding="latin-1") as file:
        with pytest.raises(TypeError):
            bfv.Ciphertext.load(file)


# ----------------------------------------------------------------------
# Storage, streams and threads
# ----------------------------------------------------------------------


def test_storage_moves(setup, rng):
    values = setup.random(rng)
    cipher = setup.encrypt(values)
    cipher.store_in_host()
    assert not cipher.is_on_device
    # Inputs in host memory are moved to the device for the operation.
    np.testing.assert_array_equal(setup.decrypt(cipher), values)
    assert not cipher.is_on_device
    cipher.store_in_device()
    assert cipher.is_on_device
    np.testing.assert_array_equal(setup.decrypt(cipher), values)


def test_key_storage_moves(setup, rng):
    for key in (
        setup.secret_key.copy(),
        setup.public_key.copy(),
        setup.relin_key.copy(),
        setup.galois_key.copy(),
        setup.switch_key.copy(),
    ):
        data = key.to_bytes()
        key.store_in_host()
        assert not key.is_on_device
        key.store_in_device()
        assert key.is_on_device
        if not isinstance(key, bfv.GaloisKey):
            assert key.to_bytes() == data
    public_key = setup.public_key.copy()
    public_key.store_in_host()
    values = setup.random(rng)
    cipher = bfv.Encryptor(setup.context, public_key).encrypt(
        setup.encoder.encode(values)
    )
    np.testing.assert_array_equal(setup.decrypt(cipher), values)


def test_host_output_option(setup, rng):
    values = setup.random(rng)
    host = hg.ExecutionOptions(storage=hg.StorageType.HOST)
    plain = setup.encoder.encode(values, options=host)
    assert not plain.is_on_device
    cipher = setup.encryptor.encrypt(plain, host)
    assert not cipher.is_on_device
    assert not bfv.Ciphertext(setup.context, host).is_on_device
    np.testing.assert_array_equal(setup.decrypt(cipher), values)


def test_stream_execution(setup, rng):
    values = setup.random(rng)
    with hg.Stream() as stream:
        plain = setup.encoder.encode(values, options=stream)
        cipher = setup.encryptor.encrypt(
            plain, options=hg.ExecutionOptions(stream=stream)
        )
        assert cipher.stream == stream.handle
        budget = setup.decryptor.remainder_noise_budget(cipher, options=stream)
        assert budget > 0
        result = setup.encoder.decode(
            setup.decryptor.decrypt(cipher, options=stream), options=stream
        )
        stream.synchronize()
    np.testing.assert_array_equal(result, values)


def test_stream_handle_and_switch(setup, rng):
    values = setup.random(rng)
    with hg.Stream() as stream:
        cipher = setup.encryptor.encrypt(
            setup.encoder.encode(values), options=stream.handle
        )
        assert cipher.stream == stream.handle
        cipher.switch_stream(None)
        assert cipher.stream == 0
        stream.synchronize()
    np.testing.assert_array_equal(setup.decrypt(cipher), values)


def test_threads(setup, rng):
    inputs = [setup.random(rng) for _ in range(4)]
    ciphers = [setup.encrypt(values) for values in inputs]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        results = list(pool.map(setup.decrypt, ciphers))
    for values, result in zip(inputs, results):
        np.testing.assert_array_equal(result, values)


@pytest.mark.slow
def test_large_ring(rng):
    large = BFVSetup(depth=8, poly_modulus_degree=32768)
    assert large.slot_count == 32768
    values = large.random(rng)
    np.testing.assert_array_equal(large.decrypt(large.encrypt(values)), values)
