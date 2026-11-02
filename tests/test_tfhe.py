# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

import copy
import io
import pickle
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import heongpu as hg
from heongpu import tfhe

SIZE = 64

BINARY_GATES = {
    "NAND": lambda x, y: ~(x & y),
    "AND": lambda x, y: x & y,
    "NOR": lambda x, y: ~(x | y),
    "OR": lambda x, y: x | y,
    "XNOR": lambda x, y: ~(x ^ y),
    "XOR": lambda x, y: x ^ y,
}


class TFHESetup:
    """A TFHE context with its keys, encryptor, decryptor and operator."""

    def __init__(self):
        self.context = tfhe.Context()
        self.keygen = tfhe.KeyGenerator(self.context)
        self.secret_key = self.keygen.secret_key()
        self.boot_key = self.keygen.bootstrapping_key(self.secret_key)
        self.encryptor = tfhe.Encryptor(self.context, self.secret_key)
        self.decryptor = tfhe.Decryptor(self.context, self.secret_key)
        self.operator = tfhe.LogicOperator(self.context)

    def encrypt(self, bits, options=None):
        return self.encryptor.encrypt(bits, options=options)

    def decrypt(self, cipher, options=None):
        return self.decryptor.decrypt(cipher, options=options)


@pytest.fixture(scope="module")
def setup():
    return TFHESetup()


def random_bits(rng, size=SIZE):
    return rng.integers(0, 2, size).astype(bool)


# ----------------------------------------------------------------------
# Context and keys
# ----------------------------------------------------------------------


def test_context(setup):
    context = setup.context
    assert context.lwe_size == 512
    assert context.tlwe_size == 1024
    assert "lwe_size=512" in repr(context)
    loaded = pickle.loads(pickle.dumps(context))
    assert loaded.lwe_size == context.lwe_size


def test_context_with_pool_config():
    context = tfhe.Context(hg.MemoryPoolConfig(max_device_fraction=0.3))
    assert context.lwe_size == 512


def test_secret_key(setup):
    key = setup.secret_key
    assert key.is_generated
    assert key.is_on_device
    assert key.lwe_size == 512
    assert key.tlwe_size == 1024
    lwe = key.get_lwe_key()
    tlwe = key.get_tlwe_key()
    assert lwe.dtype == np.int32 and lwe.shape == (512,)
    assert tlwe.dtype == np.int32 and tlwe.shape == (1024,)
    assert set(np.unique(lwe)) <= {0, 1}
    assert set(np.unique(tlwe)) <= {0, 1}
    assert 0 < lwe.sum() < 512
    assert 0 < tlwe.sum() < 1024


def test_secret_keys_are_independent(setup):
    other = setup.keygen.secret_key()
    assert not np.array_equal(
        other.get_lwe_key(), setup.secret_key.get_lwe_key()
    )


def test_generate_into_empty_keys(setup, rng):
    secret_key = tfhe.SecretKey(setup.context)
    assert not secret_key.is_generated
    setup.keygen.generate_secret_key(secret_key)
    assert secret_key.is_generated
    boot_key = tfhe.BootstrappingKey(setup.context)
    assert not boot_key.is_generated
    setup.keygen.generate_bootstrapping_key(boot_key, secret_key)
    assert boot_key.is_generated
    assert boot_key.is_on_device

    encryptor = tfhe.Encryptor(setup.context, secret_key)
    decryptor = tfhe.Decryptor(setup.context, secret_key)
    x, y = random_bits(rng), random_bits(rng)
    result = setup.operator.AND(
        encryptor.encrypt(x), encryptor.encrypt(y), boot_key
    )
    assert_array_equal(decryptor.decrypt(result), x & y)


def test_default_constructed_keys(setup):
    secret_key = tfhe.SecretKey()
    secret_key.set_context(setup.context)
    setup.keygen.generate_secret_key(secret_key)
    assert secret_key.is_generated
    boot_key = tfhe.BootstrappingKey()
    boot_key.set_context(setup.context)
    assert not boot_key.is_generated


def test_secret_key_in_host_memory(setup, rng):
    key = setup.keygen.secret_key(
        options=hg.ExecutionOptions(storage=hg.StorageType.HOST)
    )
    assert not key.is_on_device
    lwe = key.get_lwe_key()
    encryptor = tfhe.Encryptor(setup.context, key)
    decryptor = tfhe.Decryptor(setup.context, key)
    bits = random_bits(rng)
    assert_array_equal(decryptor.decrypt(encryptor.encrypt(bits)), bits)
    key.store_in_device()
    assert key.is_on_device
    assert_array_equal(key.get_lwe_key(), lwe)
    key.store_in_host()
    assert not key.is_on_device
    assert_array_equal(key.get_lwe_key(), lwe)


def test_key_copies(setup):
    key = copy.deepcopy(setup.secret_key)
    assert_array_equal(key.get_lwe_key(), setup.secret_key.get_lwe_key())
    assert_array_equal(
        key.copy().get_tlwe_key(), setup.secret_key.get_tlwe_key()
    )
    boot_key = copy.copy(setup.boot_key)
    assert boot_key.is_generated


# ----------------------------------------------------------------------
# Encryption
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bits",
    [
        [True, False, True, True, False],
        [1, 0, 0, 1],
        np.array([True, False, False]),
        np.array([1, 1, 0, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.uint8),
        (False, True),
    ],
)
def test_encrypt_inputs(setup, bits):
    cipher = setup.encrypt(bits)
    expected = np.asarray(bits).astype(bool)
    assert cipher.size == len(expected)
    assert len(cipher) == len(expected)
    result = setup.decrypt(cipher)
    assert result.dtype == np.bool_
    assert_array_equal(result, expected)


@pytest.mark.parametrize("bit", [True, False, 1, 0])
def test_encrypt_scalar(setup, bit):
    cipher = setup.encrypt(bit)
    assert cipher.size == 1
    assert_array_equal(setup.decrypt(cipher), [bool(bit)])


def test_encrypt_random(setup, rng):
    bits = random_bits(rng, 1000)
    cipher = setup.encrypt(bits)
    assert cipher.is_generated
    assert cipher.is_on_device
    assert cipher.lwe_size == 512
    assert "size=1000" in repr(cipher)
    assert_array_equal(setup.decrypt(cipher), bits)


def test_encryption_is_randomized(setup):
    bits = [True, False, True]
    a1, _ = setup.encrypt(bits).get_data()
    a2, _ = setup.encrypt(bits).get_data()
    assert not np.array_equal(a1, a2)


def test_ciphertext_data_decrypts_with_the_key(setup, rng):
    bits = random_bits(rng)
    a, b = setup.encrypt(bits).get_data()
    assert a.shape == (SIZE, 512) and a.dtype == np.int32
    assert b.shape == (SIZE,) and b.dtype == np.int32
    key = setup.secret_key.get_lwe_key().astype(np.int64)
    phase = (b.astype(np.int64) - a.astype(np.int64) @ key) % 2**32
    phase = phase.astype(np.uint32).view(np.int32)
    assert_array_equal(phase > 0, bits)
    # The messages are encoded as +-1/8 of the torus with small noise.
    eighth = 2**29
    assert np.all(np.abs(np.abs(phase.astype(np.int64)) - eighth) < 2**24)


@pytest.mark.parametrize(
    "bits, error",
    [
        ([], hg.ParameterError),
        (np.zeros(0, dtype=bool), hg.ParameterError),
        ([0, 2, 1], ValueError),
        ([-1], ValueError),
        ([[0, 1], [1, 0]], ValueError),
        ([0.0, 1.0], TypeError),
        (["a"], TypeError),
    ],
)
def test_encrypt_invalid_bits(setup, bits, error):
    with pytest.raises(error):
        setup.encrypt(bits)


def test_ungenerated_secret_key(setup):
    key = tfhe.SecretKey(setup.context)
    with pytest.raises(hg.StateError):
        tfhe.Encryptor(setup.context, key)
    with pytest.raises(hg.StateError):
        tfhe.Decryptor(setup.context, key)
    with pytest.raises(hg.StateError):
        setup.keygen.bootstrapping_key(key)
    with pytest.raises(hg.StateError):
        key.get_lwe_key()


def test_generate_twice(setup):
    with pytest.raises(hg.StateError):
        setup.keygen.generate_secret_key(setup.secret_key)
    with pytest.raises(hg.StateError):
        setup.keygen.generate_bootstrapping_key(
            setup.boot_key, setup.secret_key
        )


def test_empty_ciphertext(setup):
    cipher = tfhe.Ciphertext(setup.context)
    assert cipher.size == 0
    assert not cipher.is_generated
    assert cipher.lwe_size == 512
    with pytest.raises(hg.StateError):
        setup.decrypt(cipher)
    with pytest.raises(hg.StateError):
        cipher.get_data()
    with pytest.raises(hg.StateError):
        cipher.to_bytes()


# ----------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------


@pytest.mark.parametrize("gate", sorted(BINARY_GATES))
def test_binary_gate(setup, rng, gate):
    x, y = random_bits(rng), random_bits(rng)
    result = getattr(setup.operator, gate)(
        setup.encrypt(x), setup.encrypt(y), setup.boot_key
    )
    assert result.size == SIZE
    assert result.is_generated
    assert_array_equal(setup.decrypt(result), BINARY_GATES[gate](x, y))


def test_binary_gates_truth_table(setup):
    x = np.array([False, False, True, True])
    y = np.array([False, True, False, True])
    cx, cy = setup.encrypt(x), setup.encrypt(y)
    for gate, function in BINARY_GATES.items():
        result = getattr(setup.operator, gate)(cx, cy, setup.boot_key)
        assert_array_equal(setup.decrypt(result), function(x, y), gate)


def test_not(setup, rng):
    x = random_bits(rng)
    result = setup.operator.NOT(setup.encrypt(x))
    assert_array_equal(setup.decrypt(result), ~x)


def test_mux(setup, rng):
    x, y, control = random_bits(rng), random_bits(rng), random_bits(rng)
    result = setup.operator.MUX(
        setup.encrypt(x),
        setup.encrypt(y),
        setup.encrypt(control),
        setup.boot_key,
    )
    assert_array_equal(setup.decrypt(result), np.where(control, x, y))


def test_gates_do_not_change_inputs(setup, rng):
    x, y = random_bits(rng), random_bits(rng)
    cx, cy = setup.encrypt(x), setup.encrypt(y)
    data = cx.get_data()
    setup.operator.XOR(cx, cy, setup.boot_key)
    setup.operator.NOT(cx)
    setup.operator.MUX(cx, cy, cx, setup.boot_key)
    assert_array_equal(cx.get_data()[0], data[0])
    assert_array_equal(cx.get_data()[1], data[1])
    assert_array_equal(setup.decrypt(cy), y)


def test_gate_into_out(setup, rng):
    x, y = random_bits(rng), random_bits(rng)
    cx, cy = setup.encrypt(x), setup.encrypt(y)
    out = tfhe.Ciphertext(setup.context)
    result = setup.operator.NAND(cx, cy, setup.boot_key, out=out)
    assert result is out
    assert out.is_generated
    assert_array_equal(setup.decrypt(out), ~(x & y))
    # An output that already holds data is overwritten.
    setup.operator.OR(cx, cy, setup.boot_key, out=out)
    assert_array_equal(setup.decrypt(out), x | y)


def test_gate_into_input(setup, rng):
    x, y, control = random_bits(rng), random_bits(rng), random_bits(rng)
    cx, cy, cc = setup.encrypt(x), setup.encrypt(y), setup.encrypt(control)
    setup.operator.AND(cx, cy, setup.boot_key, out=cx)
    x = x & y
    assert_array_equal(setup.decrypt(cx), x)
    setup.operator.XNOR(cx, cy, setup.boot_key, out=cy)
    y = ~(x ^ y)
    assert_array_equal(setup.decrypt(cy), y)
    setup.operator.NOT(cx, out=cx)
    x = ~x
    assert_array_equal(setup.decrypt(cx), x)
    setup.operator.MUX(cx, cy, cc, setup.boot_key, out=cc)
    assert_array_equal(setup.decrypt(cc), np.where(control, x, y))


def test_gate_output_phase(setup, rng):
    """A gate output decrypts to the right bits, with small noise."""
    x = random_bits(rng, 1000)
    cipher = setup.operator.AND(
        setup.encrypt(x), setup.encrypt(x), setup.boot_key
    )
    a, b = cipher.get_data()
    key = setup.secret_key.get_lwe_key().astype(np.int64)
    phase = (b.astype(np.int64) - a.astype(np.int64) @ key) % 2**32
    phase = phase.astype(np.uint32).view(np.int32) / 2**32
    assert_array_equal(phase > 0, x)
    assert np.std(phase[x]) < 0.01
    assert np.std(phase[~x]) < 0.01


def test_gate_on_gate_output_and_fresh_input(setup, rng):
    x, y = random_bits(rng, 1000), random_bits(rng, 1000)
    cx, cy = setup.encrypt(x), setup.encrypt(y)
    refreshed = setup.operator.AND(cx, cx, setup.boot_key)
    for gate, function in BINARY_GATES.items():
        result = getattr(setup.operator, gate)(refreshed, cy, setup.boot_key)
        assert_array_equal(setup.decrypt(result), function(x, y), gate)


@pytest.mark.parametrize("gate", sorted(BINARY_GATES))
def test_gate_on_gate_outputs(setup, rng, gate):
    x, y = random_bits(rng, 1000), random_bits(rng, 1000)
    op, bk = setup.operator, setup.boot_key
    gx = op.AND(setup.encrypt(x), setup.encrypt(x), bk)
    gy = op.AND(setup.encrypt(y), setup.encrypt(y), bk)
    result = getattr(op, gate)(gx, gy, bk)
    assert_array_equal(setup.decrypt(result), BINARY_GATES[gate](x, y))


def test_ripple_carry_adder(setup, rng):
    """Adds two vectors of 8-bit numbers with a circuit of gates."""
    bits, lanes = 8, 32
    x = rng.integers(0, 256, lanes)
    y = rng.integers(0, 256, lanes)
    op, bk = setup.operator, setup.boot_key
    x_bits = [setup.encrypt((x >> i) & 1) for i in range(bits)]
    y_bits = [setup.encrypt((y >> i) & 1) for i in range(bits)]
    carry = setup.encrypt(np.zeros(lanes, dtype=bool))
    total = []
    for i in range(bits):
        half = op.XOR(x_bits[i], y_bits[i], bk)
        total.append(op.XOR(half, carry, bk))
        carry = op.OR(
            op.AND(x_bits[i], y_bits[i], bk), op.AND(half, carry, bk), bk
        )
    total.append(carry)
    result = sum(
        setup.decrypt(bit).astype(np.int64) << i for i, bit in enumerate(total)
    )
    assert_array_equal(result, x + y)


def test_gate_errors(setup, rng):
    op, bk = setup.operator, setup.boot_key
    small = setup.encrypt(random_bits(rng, 4))
    large = setup.encrypt(random_bits(rng, 5))
    empty = tfhe.Ciphertext(setup.context)
    for gate in BINARY_GATES:
        with pytest.raises(hg.ParameterError):
            getattr(op, gate)(small, large, bk)
        with pytest.raises(hg.StateError):
            getattr(op, gate)(small, empty, bk)
    with pytest.raises(hg.ParameterError):
        op.MUX(small, small, large, bk)
    with pytest.raises(hg.StateError):
        op.MUX(small, small, empty, bk)
    with pytest.raises(hg.StateError):
        op.NOT(empty)
    with pytest.raises(hg.StateError):
        op.AND(small, small, tfhe.BootstrappingKey(setup.context))
    with pytest.raises(TypeError):
        op.AND(small, small, bk, out=3)
    with pytest.raises(TypeError):
        op.AND(small, [1, 0, 1, 0], bk)


# ----------------------------------------------------------------------
# Storage and copies
# ----------------------------------------------------------------------


def test_ciphertext_storage(setup, rng):
    x, y = random_bits(rng), random_bits(rng)
    cx, cy = setup.encrypt(x), setup.encrypt(y)
    data = cx.get_data()
    cx.store_in_host()
    assert not cx.is_on_device
    assert "host" in repr(cx)
    assert_array_equal(cx.get_data()[0], data[0])
    assert_array_equal(setup.decrypt(cx), x)
    result = setup.operator.XOR(cx, cy, setup.boot_key)
    assert_array_equal(setup.decrypt(result), x ^ y)
    assert not cx.is_on_device
    cx.store_in_device()
    assert cx.is_on_device
    assert_array_equal(cx.get_data()[1], data[1])


def test_output_in_host_memory(setup, rng):
    x, y = random_bits(rng), random_bits(rng)
    options = hg.ExecutionOptions(storage=hg.StorageType.HOST)
    cipher = setup.encrypt(x, options=options)
    assert not cipher.is_on_device
    result = setup.operator.OR(
        cipher, setup.encrypt(y), setup.boot_key, options=options
    )
    assert not result.is_on_device
    assert_array_equal(setup.decrypt(result), x | y)


def test_boot_key_in_host_memory(setup, rng):
    boot_key = setup.boot_key.copy()
    boot_key.store_in_host()
    assert not boot_key.is_on_device
    x, y = random_bits(rng), random_bits(rng)
    result = setup.operator.AND(setup.encrypt(x), setup.encrypt(y), boot_key)
    assert_array_equal(setup.decrypt(result), x & y)
    boot_key.store_in_device()
    assert boot_key.is_on_device


def test_ciphertext_copies(setup, rng):
    x = random_bits(rng)
    cipher = setup.encrypt(x)
    for duplicate in (cipher.copy(), copy.copy(cipher), copy.deepcopy(cipher)):
        assert_array_equal(duplicate.get_data()[0], cipher.get_data()[0])
        setup.operator.NOT(duplicate, out=duplicate)
        assert_array_equal(setup.decrypt(duplicate), ~x)
    assert_array_equal(setup.decrypt(cipher), x)


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------


def test_ciphertext_serialization(setup, rng):
    x = random_bits(rng)
    cipher = setup.encrypt(x)
    data = cipher.to_bytes()
    assert isinstance(data, bytes)
    assert data.startswith(b"HGPYTFHE")
    loaded = tfhe.Ciphertext.from_bytes(data, setup.context)
    assert loaded.size == SIZE
    assert_array_equal(loaded.get_data()[0], cipher.get_data()[0])
    assert_array_equal(setup.decrypt(loaded), x)
    # Without a context, a new one is created.
    assert_array_equal(setup.decrypt(tfhe.Ciphertext.from_bytes(data)), x)
    unpickled = pickle.loads(pickle.dumps(cipher))
    assert_array_equal(setup.decrypt(unpickled), x)


def test_serialized_ciphertext_in_gates(setup, rng):
    x, y = random_bits(rng), random_bits(rng)
    cx = pickle.loads(pickle.dumps(setup.encrypt(x)))
    cy = pickle.loads(pickle.dumps(setup.encrypt(y)))
    result = setup.operator.NOR(cx, cy, setup.boot_key)
    result = pickle.loads(pickle.dumps(result))
    assert_array_equal(setup.decrypt(result), ~(x | y))


def test_ciphertext_serialization_keeps_storage(setup, rng):
    x = random_bits(rng)
    cipher = setup.encrypt(x)
    cipher.store_in_host()
    loaded = pickle.loads(pickle.dumps(cipher))
    assert not loaded.is_on_device
    assert_array_equal(setup.decrypt(loaded), x)


def test_ciphertext_file(setup, rng, tmp_path):
    x = random_bits(rng)
    cipher = setup.encrypt(x)
    path = tmp_path / "cipher.bin"
    cipher.save(path)
    assert_array_equal(setup.decrypt(tfhe.Ciphertext.load(path)), x)
    cipher.save(str(path))
    assert_array_equal(
        setup.decrypt(tfhe.Ciphertext.load(str(path), setup.context)), x
    )
    buffer = io.BytesIO()
    cipher.save(buffer)
    buffer.seek(0)
    assert_array_equal(setup.decrypt(tfhe.Ciphertext.load(buffer)), x)


def test_secret_key_serialization(setup, rng):
    key = setup.secret_key
    for loaded in (
        tfhe.SecretKey.from_bytes(key.to_bytes(), setup.context),
        tfhe.SecretKey.from_bytes(key.to_bytes()),
        pickle.loads(pickle.dumps(key)),
    ):
        assert loaded.is_generated
        assert_array_equal(loaded.get_lwe_key(), key.get_lwe_key())
        assert_array_equal(loaded.get_tlwe_key(), key.get_tlwe_key())
    bits = random_bits(rng)
    decryptor = tfhe.Decryptor(setup.context, loaded)
    assert_array_equal(decryptor.decrypt(setup.encrypt(bits)), bits)
    encryptor = tfhe.Encryptor(setup.context, loaded)
    assert_array_equal(setup.decrypt(encryptor.encrypt(bits)), bits)


def test_secret_key_serialization_keeps_storage(setup):
    key = setup.secret_key.copy()
    key.store_in_host()
    loaded = pickle.loads(pickle.dumps(key))
    assert not loaded.is_on_device
    assert_array_equal(loaded.get_lwe_key(), setup.secret_key.get_lwe_key())


def test_boot_key_serialization(setup, rng, tmp_path):
    path = tmp_path / "boot_key.bin"
    setup.boot_key.save(path)
    loaded = tfhe.BootstrappingKey.load(path, setup.context)
    assert loaded.is_generated
    assert loaded.is_on_device
    x, y, control = random_bits(rng), random_bits(rng), random_bits(rng)
    cx, cy, cc = setup.encrypt(x), setup.encrypt(y), setup.encrypt(control)
    assert_array_equal(setup.decrypt(setup.operator.XOR(cx, cy, loaded)), x ^ y)
    unpickled = pickle.loads(pickle.dumps(setup.boot_key))
    assert_array_equal(
        setup.decrypt(setup.operator.MUX(cx, cy, cc, unpickled)),
        np.where(control, x, y),
    )


def test_client_server_roundtrip(setup, rng):
    """The client keeps the secret key; the server only gets the
    bootstrapping key and the ciphertexts as bytes."""
    x, y = random_bits(rng), random_bits(rng)
    boot_key_bytes = setup.boot_key.to_bytes()
    x_bytes = setup.encrypt(x).to_bytes()
    y_bytes = setup.encrypt(y).to_bytes()

    server_context = tfhe.Context()
    server_key = tfhe.BootstrappingKey.from_bytes(
        boot_key_bytes, server_context
    )
    server = tfhe.LogicOperator(server_context)
    result = server.XOR(
        tfhe.Ciphertext.from_bytes(x_bytes, server_context),
        tfhe.Ciphertext.from_bytes(y_bytes, server_context),
        server_key,
    )
    result_bytes = result.to_bytes()

    assert_array_equal(
        setup.decrypt(tfhe.Ciphertext.from_bytes(result_bytes)), x ^ y
    )


def test_serialization_errors(setup, rng):
    cipher_bytes = setup.encrypt(random_bits(rng)).to_bytes()
    key_bytes = setup.secret_key.to_bytes()
    with pytest.raises(hg.SerializationError):
        tfhe.Ciphertext.from_bytes(b"")
    with pytest.raises(hg.SerializationError):
        tfhe.Ciphertext.from_bytes(b"not a ciphertext at all")
    with pytest.raises(hg.SerializationError):
        tfhe.Ciphertext.from_bytes(cipher_bytes[:-1])
    with pytest.raises(hg.SerializationError):
        tfhe.Ciphertext.from_bytes(cipher_bytes + b"\0")
    with pytest.raises(hg.SerializationError):
        tfhe.Ciphertext.from_bytes(key_bytes)
    with pytest.raises(hg.SerializationError):
        tfhe.BootstrappingKey.from_bytes(cipher_bytes)
    with pytest.raises(hg.SerializationError):
        tfhe.SecretKey.from_bytes(key_bytes[:100])
    # An unknown format version.
    data = bytearray(cipher_bytes)
    data[8:12] = struct.pack("<I", 99)
    with pytest.raises(hg.SerializationError, match="version"):
        tfhe.Ciphertext.from_bytes(bytes(data))
    # Parameters that differ from those of the context.
    data = bytearray(cipher_bytes)
    data[16:20] = struct.pack("<i", 256)
    with pytest.raises(hg.ContextMismatchError):
        tfhe.Ciphertext.from_bytes(bytes(data))
    data = bytearray(key_bytes)
    data[16:20] = struct.pack("<i", 256)
    with pytest.raises(hg.ContextMismatchError):
        tfhe.SecretKey.from_bytes(bytes(data))
    # A secret key must be binary.
    data = bytearray(key_bytes)
    data[-4:] = struct.pack("<i", 2)
    with pytest.raises(hg.SerializationError, match="binary"):
        tfhe.SecretKey.from_bytes(bytes(data))
    with pytest.raises(hg.StateError):
        tfhe.SecretKey(setup.context).to_bytes()
    with pytest.raises(hg.StateError):
        tfhe.BootstrappingKey(setup.context).to_bytes()


# ----------------------------------------------------------------------
# Streams and threads
# ----------------------------------------------------------------------


def test_stream_execution(setup, rng):
    x, y, control = random_bits(rng), random_bits(rng), random_bits(rng)
    with hg.Stream() as stream:
        cx = setup.encrypt(x, options=stream)
        cy = setup.encrypt(y, options=hg.ExecutionOptions(stream=stream))
        cc = setup.encrypt(control, options=stream)
        assert cx.stream == stream.handle
        result = setup.operator.AND(cx, cy, setup.boot_key, options=stream)
        assert result.stream == stream.handle
        assert_array_equal(setup.decrypt(result, options=stream), x & y)
        result = setup.operator.MUX(cx, cy, cc, setup.boot_key, options=stream)
        assert_array_equal(
            setup.decrypt(result, options=stream), np.where(control, x, y)
        )
        result = setup.operator.NOT(cx, options=stream)
        assert_array_equal(setup.decrypt(result, options=stream), ~x)
        stream.synchronize()


def test_switch_stream(setup, rng):
    x = random_bits(rng)
    cipher = setup.encrypt(x)
    hg.synchronize()
    with hg.Stream() as stream:
        cipher.switch_stream(stream)
        assert cipher.stream == stream.handle
        assert_array_equal(setup.decrypt(cipher, options=stream), x)
        cipher.switch_stream(None)
        stream.synchronize()
    assert cipher.stream == 0
    key = setup.secret_key.copy()
    key.switch_stream(0)
    assert key.stream == 0
    boot_key = tfhe.BootstrappingKey(setup.context)
    boot_key.switch_stream(0)
    assert boot_key.stream == 0


def test_threads_with_streams(setup, rng):
    inputs = [(random_bits(rng), random_bits(rng)) for _ in range(4)]

    def work(pair):
        x, y = pair
        with hg.Stream() as stream:
            cx = setup.encrypt(x, options=stream)
            cy = setup.encrypt(y, options=stream)
            result = setup.operator.XOR(cx, cy, setup.boot_key, options=stream)
            result = setup.operator.AND(
                result, cy, setup.boot_key, options=stream
            )
            return setup.decrypt(result, options=stream)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, inputs))
    for (x, y), result in zip(inputs, results):
        assert_array_equal(result, (x ^ y) & y)


# ----------------------------------------------------------------------
# Long tests
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_large_batch(setup, rng):
    # Bootstrapping needs about 100 KB of temporary GPU memory per bit.
    size = 1 << 12
    x, y, control = (
        random_bits(rng, size),
        random_bits(rng, size),
        random_bits(rng, size),
    )
    cx, cy, cc = setup.encrypt(x), setup.encrypt(y), setup.encrypt(control)
    for gate, function in BINARY_GATES.items():
        result = getattr(setup.operator, gate)(cx, cy, setup.boot_key)
        assert_array_equal(setup.decrypt(result), function(x, y), gate)
    result = setup.operator.MUX(cx, cy, cc, setup.boot_key)
    assert_array_equal(setup.decrypt(result), np.where(control, x, y))


@pytest.mark.slow
def test_long_gate_chain(setup, rng):
    """Bootstrapping keeps the noise bounded over a long chain of gates,
    each with one fresh input."""
    x = random_bits(rng, 256)
    expected = x.copy()
    cipher = setup.encrypt(x)
    op, bk = setup.operator, setup.boot_key
    gates = sorted(BINARY_GATES)
    for step in range(300):
        y = random_bits(rng, 256)
        gate = gates[step % len(gates)]
        cipher = getattr(op, gate)(cipher, setup.encrypt(y), bk)
        expected = BINARY_GATES[gate](expected, y)
        if step % 50 == 49:
            assert_array_equal(setup.decrypt(cipher), expected, step)
    assert_array_equal(setup.decrypt(cipher), expected)


@pytest.mark.slow
def test_adder_32_bit(setup, rng):
    bits, lanes = 32, 256
    x = rng.integers(0, 2**31, lanes, dtype=np.int64)
    y = rng.integers(0, 2**31, lanes, dtype=np.int64)
    op, bk = setup.operator, setup.boot_key
    carry = setup.encrypt(np.zeros(lanes, dtype=bool))
    total = []
    for i in range(bits):
        cx = setup.encrypt((x >> i) & 1)
        cy = setup.encrypt((y >> i) & 1)
        half = op.XOR(cx, cy, bk)
        total.append(op.XOR(half, carry, bk))
        carry = op.MUX(carry, cx, half, bk)
    total.append(carry)
    result = sum(
        setup.decrypt(bit).astype(np.int64) << i for i, bit in enumerate(total)
    )
    assert_array_equal(result, x + y)
