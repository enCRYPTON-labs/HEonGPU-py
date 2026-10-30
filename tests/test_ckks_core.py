# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

import copy
import io
import pickle

import numpy as np
import pytest

import heongpu as hg
from heongpu import ckks

TOLERANCE = 1e-6


def test_params_generate():
    params = ckks.Params(multiplicative_depth=3, scale_bit_size=40)
    parameters = params.generate()
    assert parameters.multiplicative_depth == 3
    assert parameters.poly_modulus_degree >= 8192
    assert len(parameters.coeff_modulus) == 4
    assert parameters.total_bit_count <= parameters.max_bit_count
    assert "CKKS" in str(parameters)
    assert params.scale() == 2.0**40


def test_params_explicit_bit_sizes():
    params = ckks.Params(
        poly_modulus_degree=8192,
        coeff_modulus_bit_sizes=[60, 40, 40],
        special_modulus_bit_sizes=[60],
    )
    context = params.make_context()
    context.generate()
    assert context.poly_modulus_degree == 8192
    assert context.ciphertext_modulus_count == 3
    assert context.key_modulus_count == 4
    assert len(context.key_modulus) == 4


def test_params_too_deep():
    params = ckks.Params(multiplicative_depth=100, poly_modulus_degree=8192)
    with pytest.raises(hg.ParameterError):
        params.generate()


def test_context_setters():
    context = ckks.Context(hg.SecurityLevel.SEC128)
    context.set_poly_modulus_degree(8192)
    context.set_coeff_modulus_bit_sizes([60, 40, 40], [60])
    context.generate()
    assert context.slot_count == 4096
    assert context.security_level == hg.SecurityLevel.SEC128
    assert context.rescale_mode == hg.RescaleMode.MANUAL
    assert "Context" in repr(context)
    assert str(context)


def test_context_serialization(ckks_setup):
    data = ckks_setup.context.to_bytes()
    loaded = ckks.Context.from_bytes(data)
    assert loaded.context_id == ckks_setup.context.context_id
    buffer = io.BytesIO()
    ckks_setup.context.save(buffer)
    buffer.seek(0)
    assert ckks.Context.load(buffer).context_id == ckks_setup.context.context_id


def test_encrypt_decrypt_real(ckks_setup, rng):
    values = rng.uniform(-1, 1, ckks_setup.slot_count)
    cipher = ckks_setup.encrypt(values)
    assert cipher.size == 2
    assert cipher.depth == 0
    assert cipher.level == ckks_setup.context.ciphertext_modulus_count - 1
    assert cipher.scale == ckks_setup.scale
    assert cipher.is_on_device
    result = ckks_setup.decrypt(cipher)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, values, atol=TOLERANCE)


def test_encrypt_decrypt_complex(ckks_setup, rng):
    values = rng.uniform(-1, 1, ckks_setup.slot_count) + 1j * rng.uniform(
        -1, 1, ckks_setup.slot_count
    )
    result = ckks_setup.decrypt_complex(ckks_setup.encrypt(values))
    assert result.dtype == np.complex128
    np.testing.assert_allclose(result, values, atol=TOLERANCE)


def test_encode_list_and_scalar(ckks_setup):
    result = ckks_setup.decrypt(ckks_setup.encrypt([0.5, -0.25, 0.125]))
    np.testing.assert_allclose(result[:3], [0.5, -0.25, 0.125], atol=TOLERANCE)
    np.testing.assert_allclose(result[3:], 0.0, atol=TOLERANCE)
    result = ckks_setup.decrypt(ckks_setup.encrypt(0.75))
    np.testing.assert_allclose(result, 0.75, atol=TOLERANCE)


def test_encode_at_depth(ckks_setup):
    plain = ckks_setup.encoder.encode([1.0, 2.0], ckks_setup.scale, depth=1)
    assert plain.depth == 1
    cipher = ckks_setup.encryptor.encrypt(plain)
    assert cipher.depth == 1
    np.testing.assert_allclose(
        ckks_setup.decrypt(cipher)[:2], [1.0, 2.0], atol=TOLERANCE
    )


def test_encode_rejects_bad_input(ckks_setup):
    with pytest.raises(ValueError):
        ckks_setup.encoder.encode(np.zeros((2, 2)), ckks_setup.scale)
    with pytest.raises(TypeError):
        ckks_setup.encoder.encode("text", ckks_setup.scale)
    with pytest.raises(ValueError):
        ckks_setup.encoder.encode(
            np.zeros(ckks_setup.slot_count + 1), ckks_setup.scale
        )


def test_encode_without_scale_needs_automatic_mode(ckks_setup):
    with pytest.raises(hg.StateError):
        ckks_setup.encoder.encode([1.0])


def test_automatic_mode_encode(ckks_auto_setup, rng):
    values = rng.uniform(-1, 1, 16)
    result = ckks_auto_setup.decrypt(ckks_auto_setup.encrypt(values))
    np.testing.assert_allclose(result[:16], values, atol=TOLERANCE)
    assert ckks_auto_setup.context.rescale_mode == hg.RescaleMode.AUTOMATIC


@pytest.mark.parametrize(
    "name", ["secret_key", "public_key", "relin_key", "galois_key"]
)
def test_key_serialization(ckks_setup, name, tmp_path):
    key = getattr(ckks_setup, name)
    data = key.to_bytes()
    assert isinstance(data, bytes) and len(data) > 0
    # The Galois keys are kept in a hash map, so their order in the data can
    # change after loading.
    def same(other):
        other_data = other.to_bytes()
        if name == "galois_key":
            return len(other_data) == len(data)
        return other_data == data

    assert same(type(key).from_bytes(data))
    path = tmp_path / f"{name}.bin"
    key.save(path)
    assert same(type(key).load(str(path)))
    assert same(pickle.loads(pickle.dumps(key)))


def test_loaded_keys_work(ckks_setup, rng):
    secret_key = ckks.SecretKey.from_bytes(ckks_setup.secret_key.to_bytes())
    public_key = ckks.PublicKey.from_bytes(ckks_setup.public_key.to_bytes())
    encryptor = ckks.Encryptor(ckks_setup.context, public_key)
    decryptor = ckks.Decryptor(ckks_setup.context, secret_key)
    values = rng.uniform(-1, 1, 8)
    cipher = encryptor.encrypt(
        ckks_setup.encoder.encode(values, ckks_setup.scale)
    )
    result = ckks_setup.encoder.decode(decryptor.decrypt(cipher))
    np.testing.assert_allclose(result[:8], values, atol=TOLERANCE)


def test_ciphertext_serialization(ckks_setup, rng, tmp_path):
    values = rng.uniform(-1, 1, 8)
    cipher = ckks_setup.encrypt(values)
    for loaded in (
        ckks.Ciphertext.from_bytes(cipher.to_bytes()),
        pickle.loads(pickle.dumps(cipher)),
        copy.deepcopy(cipher),
        cipher.copy(),
    ):
        np.testing.assert_allclose(
            ckks_setup.decrypt(loaded)[:8], values, atol=TOLERANCE
        )
    path = tmp_path / "cipher.bin"
    with open(path, "wb") as file:
        cipher.save(file)
    with open(path, "rb") as file:
        loaded = ckks.Ciphertext.load(file)
    np.testing.assert_allclose(
        ckks_setup.decrypt(loaded)[:8], values, atol=TOLERANCE
    )


def test_serialization_error():
    # Some malformed inputs are reported as plain RuntimeError by HEonGPU.
    with pytest.raises((hg.SerializationError, RuntimeError)):
        ckks.Ciphertext.from_bytes(b"not a ciphertext")


def test_text_file_rejected(ckks_setup, tmp_path):
    path = tmp_path / "cipher.bin"
    ckks_setup.encrypt([1.0]).save(path)
    with open(path, "r", encoding="latin-1") as file:
        with pytest.raises(TypeError):
            ckks.Ciphertext.load(file)


def test_storage_moves(ckks_setup, rng):
    values = rng.uniform(-1, 1, 8)
    cipher = ckks_setup.encrypt(values)
    cipher.store_in_host()
    assert not cipher.is_on_device
    cipher.store_in_device()
    assert cipher.is_on_device
    np.testing.assert_allclose(
        ckks_setup.decrypt(cipher)[:8], values, atol=TOLERANCE
    )


def test_ciphertext_raw_data(ckks_setup):
    cipher = ckks_setup.encrypt([1.0])
    data = cipher.get_data()
    assert data.dtype == np.uint64
    assert data.shape == (
        cipher.size,
        cipher.coeff_modulus_count - cipher.depth,
        cipher.ring_size,
    )


def test_plaintext_properties(ckks_setup):
    plain = ckks_setup.encoder.encode([1.0], ckks_setup.scale)
    assert plain.depth == 0
    assert plain.scale == ckks_setup.scale
    assert plain.encoding == hg.Encoding.SLOT
    assert plain.get_data().shape == (plain.size,)
    assert pickle.loads(pickle.dumps(plain)).to_bytes() == plain.to_bytes()


def test_context_mismatch(ckks_setup):
    other = ckks.Params(
        poly_modulus_degree=16384,
        coeff_modulus_bit_sizes=[60, 30, 30],
        special_modulus_bit_sizes=[60],
    ).make_context()
    other.generate()
    decryptor = ckks.Decryptor(other, ckks.KeyGenerator(other).secret_key())
    with pytest.raises(hg.ContextMismatchError):
        decryptor.decrypt(ckks_setup.encrypt([1.0]))


def test_sparse_secret_key(ckks_setup):
    key = ckks.SecretKey(ckks_setup.context, 64)
    ckks_setup.keygen.generate_secret_key(key)
    assert key.ring_size == ckks_setup.context.poly_modulus_degree


def test_switch_key(ckks_setup, rng):
    keygen = ckks_setup.keygen
    new_key = keygen.secret_key()
    switch_key = ckks.SwitchKey(ckks_setup.context)
    keygen.generate_switch_key(switch_key, new_key, ckks_setup.secret_key)
    assert switch_key.is_on_device
    assert len(switch_key.to_bytes()) > 0


def test_galois_key_constructors(ckks_setup):
    ckks.GaloisKey(ckks_setup.context, shifts=[1, 2, 3])
    ckks.GaloisKey(ckks_setup.context, max_shift=8)
    with pytest.raises(ValueError):
        ckks.GaloisKey(ckks_setup.context, shifts=[1], max_shift=8)


def test_stream_execution(ckks_setup, rng):
    values = rng.uniform(-1, 1, 8)
    with hg.Stream() as stream:
        plain = ckks_setup.encoder.encode(
            values, ckks_setup.scale, options=stream
        )
        cipher = ckks_setup.encryptor.encrypt(
            plain, options=hg.ExecutionOptions(stream=stream)
        )
        assert cipher.stream == stream.handle
        result = ckks_setup.encoder.decode(
            ckks_setup.decryptor.decrypt(cipher, options=stream),
            options=stream,
        )
        stream.synchronize()
    np.testing.assert_allclose(result[:8], values, atol=TOLERANCE)


def test_context_pickle(ckks_setup):
    loaded = pickle.loads(pickle.dumps(ckks_setup.context))
    assert loaded.context_id == ckks_setup.context.context_id


def test_raw_data_waits_for_the_stream(ckks_setup, rng):
    keygen = ckks_setup.keygen
    galois_key = keygen.galois_key(ckks_setup.secret_key, [1])
    operator = ckks.ArithmeticOperator(ckks_setup.context, ckks_setup.encoder)
    cipher = ckks_setup.encrypt(rng.uniform(-1, 1, 8))
    hg.synchronize()
    stream = hg.Stream()
    options = hg.ExecutionOptions(stream=stream)
    for _ in range(50):
        operator.rotate_inplace(cipher, galois_key, 1, options=options)
    data = cipher.get_data()
    stream.synchronize()
    np.testing.assert_array_equal(data, cipher.get_data())
