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
RING_SIZE = 8192
ROTATIONS = [1, 3, -2]


def make_params(depth=2, poly_modulus_degree=RING_SIZE):
    return bfv.Params(
        multiplicative_depth=depth,
        plain_modulus=PLAIN_MODULUS,
        poly_modulus_degree=poly_modulus_degree,
    )


@pytest.fixture(scope="module")
def session():
    return bfv.Session(make_params(), rotations=ROTATIONS)


@pytest.fixture(scope="module")
def evaluator_bytes(session):
    return session.to_bytes(hg.SessionRole.EVALUATOR)


@pytest.fixture(scope="module")
def evaluator(evaluator_bytes):
    return bfv.Session.from_bytes(evaluator_bytes)


@pytest.fixture(scope="module")
def encryptor(session):
    return bfv.Session.from_bytes(
        session.to_bytes(hg.SessionRole.ENCRYPTOR),
        expected_role=hg.SessionRole.ENCRYPTOR,
    )


def random_values(rng, size=RING_SIZE):
    return rng.integers(0, PLAIN_MODULUS, size, dtype=np.uint64)


def padded(values, size=RING_SIZE):
    """The slot values of a vector: the values, then zeros."""
    values = np.mod(np.asarray(values, dtype=object), PLAIN_MODULUS)
    result = np.zeros(size, dtype=object)
    result[: len(values)] = values
    return result


def expected(values):
    """values (any integers, as Python objects) modulo t, as uint64."""
    return np.mod(np.asarray(values, dtype=object), PLAIN_MODULUS).astype(
        np.uint64
    )


def centered(values):
    values = np.mod(np.asarray(values, dtype=object), PLAIN_MODULUS)
    half = PLAIN_MODULUS // 2
    return np.where(values > half, values - PLAIN_MODULUS, values).astype(
        np.int64
    )


def rotate_rows(values, shift):
    """Both rows of N/2 slots rotated left by shift."""
    rows = np.asarray(values, dtype=object).reshape(2, -1)
    return np.roll(rows, -shift, axis=1).reshape(-1)


# ----------------------------------------------------------------------
# Construction and properties
# ----------------------------------------------------------------------


def test_session_properties(session):
    assert session.role == hg.SessionRole.OWNER
    assert session.slot_count == RING_SIZE
    assert session.row_size == RING_SIZE // 2
    assert session.plain_modulus == PLAIN_MODULUS
    assert session.stream == 0
    assert repr(session) == (
        f"<heongpu.bfv.Session role=OWNER slot_count={RING_SIZE} "
        f"plain_modulus={PLAIN_MODULUS}>"
    )


def test_session_from_parameter_set(rng):
    parameters = make_params(depth=1, poly_modulus_degree=4096).generate()
    s = bfv.Session(parameters)
    assert s.slot_count == 4096
    assert s.plain_modulus == PLAIN_MODULUS
    x = rng.integers(0, PLAIN_MODULUS, 4096, dtype=np.uint64)
    np.testing.assert_array_equal(s.decrypt(s.encrypt(x)), x)


def test_session_generated_plain_modulus():
    s = bfv.Session(bfv.Params(multiplicative_depth=1))
    t = s.plain_modulus
    assert t.bit_length() == 20
    assert t % (2 * s.slot_count) == 1


def test_session_too_deep():
    params = make_params(depth=200, poly_modulus_degree=4096)
    with pytest.raises(hg.ParameterError):
        bfv.Session(params)


# ----------------------------------------------------------------------
# Encryption and decryption
# ----------------------------------------------------------------------


def test_encrypt_decrypt(session, rng):
    x = random_values(rng)
    c = session.encrypt(x)
    assert isinstance(c, bfv.SessionCiphertext)
    assert not c.is_empty
    assert c.plain_modulus == PLAIN_MODULUS
    assert c.stream == 0
    result = session.decrypt(c)
    assert result.dtype == np.uint64
    np.testing.assert_array_equal(result, x)


def test_decrypt_signed(session, rng):
    x = rng.integers(-1000, 1000, 100)
    c = session.encrypt(x)
    signed = session.decrypt(c, signed=True)
    assert signed.dtype == np.int64
    np.testing.assert_array_equal(signed, centered(padded(x)))
    np.testing.assert_array_equal(session.decrypt_signed(c), signed)
    np.testing.assert_array_equal(session.decrypt(c), expected(padded(x)))


def test_encrypt_scalar_fills_slots(session):
    np.testing.assert_array_equal(
        session.decrypt(session.encrypt(7)), np.full(RING_SIZE, 7)
    )
    np.testing.assert_array_equal(
        session.decrypt(session.encrypt(-1)),
        np.full(RING_SIZE, PLAIN_MODULUS - 1),
    )


def test_encrypt_short_and_empty(session):
    np.testing.assert_array_equal(
        session.decrypt(session.encrypt([1, 2, 3])), padded([1, 2, 3])
    )
    np.testing.assert_array_equal(
        session.decrypt(session.encrypt([])), np.zeros(RING_SIZE)
    )


def test_encrypt_reduces_values(session):
    values = [-1, PLAIN_MODULUS, 2**70 + 5, -(2**80), 2**64 - 1]
    np.testing.assert_array_equal(
        session.decrypt(session.encrypt(values)), expected(padded(values))
    )
    big = np.array([2**64 - 1, 2**63], dtype=np.uint64)
    np.testing.assert_array_equal(
        session.decrypt(session.encrypt(big))[:2],
        expected([2**64 - 1, 2**63]),
    )


@pytest.mark.parametrize(
    "dtype", [np.int8, np.uint8, np.int16, np.int32, np.uint32, np.int64, bool]
)
def test_encrypt_dtypes(session, dtype):
    x = np.array([0, 1, 1, 0, 1], dtype=dtype)
    if dtype is not bool:
        x = np.array([0, 1, 2, 100, 127], dtype=dtype)
    np.testing.assert_array_equal(
        session.decrypt(session.encrypt(x)), padded(x.astype(np.int64))
    )


def test_encrypt_rejects_bad_input(session):
    with pytest.raises(TypeError):
        session.encrypt([1.5, 2.0])
    with pytest.raises(TypeError):
        session.encrypt("abc")
    with pytest.raises(ValueError):
        session.encrypt(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(ValueError):
        session.encrypt(np.zeros(RING_SIZE + 1, dtype=np.int64))


def test_encrypt_plaintext(session, rng):
    x = random_values(rng)
    plain = session.encoder().encode(x)
    np.testing.assert_array_equal(session.decrypt(session.encrypt(plain)), x)


def test_noise_budget(session, rng):
    x = rng.integers(0, 100, 100)
    c = session.encrypt(x)
    fresh = session.noise_budget(c)
    assert fresh > 0
    assert 0 < session.noise_budget(c * c) < fresh


# ----------------------------------------------------------------------
# Arithmetic
# ----------------------------------------------------------------------


def test_ciphertext_arithmetic(session, rng):
    x = random_values(rng)
    y = random_values(rng)
    a = session.encrypt(x)
    b = session.encrypt(y)
    xo = x.astype(object)
    yo = y.astype(object)
    np.testing.assert_array_equal(session.decrypt(a + b), expected(xo + yo))
    np.testing.assert_array_equal(session.decrypt(a - b), expected(xo - yo))
    np.testing.assert_array_equal(session.decrypt(a * b), expected(xo * yo))
    np.testing.assert_array_equal(session.decrypt(-a), expected(-xo))
    np.testing.assert_array_equal(session.decrypt(+a), x)


def test_polynomial(session, rng):
    x = rng.integers(0, PLAIN_MODULUS, RING_SIZE).astype(object)
    y = rng.integers(0, PLAIN_MODULUS, RING_SIZE).astype(object)
    a = session.encrypt(x)
    b = session.encrypt(y)
    z = a * a * b + 3 * a - b
    np.testing.assert_array_equal(
        session.decrypt(z), expected(x * x * y + 3 * x - y)
    )
    assert session.noise_budget(z) > 0


@pytest.mark.parametrize("value", [5, -7, 0, PLAIN_MODULUS + 3, 2**90])
def test_scalar_operands(session, rng, value):
    x = random_values(rng).astype(object)
    a = session.encrypt(x)
    d = session.decrypt
    np.testing.assert_array_equal(d(a + value), expected(x + value))
    np.testing.assert_array_equal(d(value + a), expected(x + value))
    np.testing.assert_array_equal(d(a - value), expected(x - value))
    np.testing.assert_array_equal(d(value - a), expected(value - x))
    np.testing.assert_array_equal(d(a * value), expected(x * value))
    np.testing.assert_array_equal(d(value * a), expected(x * value))


def test_numpy_scalar_operands(session, rng):
    x = random_values(rng).astype(object)
    a = session.encrypt(x)
    for value in [np.int64(-9), np.uint32(11), np.int8(3), True]:
        v = int(value)
        np.testing.assert_array_equal(
            session.decrypt(value + a), expected(x + v)
        )
        np.testing.assert_array_equal(
            session.decrypt(value * a), expected(x * v)
        )
        np.testing.assert_array_equal(
            session.decrypt(value - a), expected(v - x)
        )


@pytest.mark.parametrize("kind", ["list", "int64", "uint64", "short"])
def test_vector_operands(session, rng, kind):
    x = random_values(rng).astype(object)
    a = session.encrypt(x)
    if kind == "short":
        values = [3, -4, 2**70]
    elif kind == "list":
        values = [int(v) for v in rng.integers(-(2**40), 2**40, RING_SIZE)]
    elif kind == "int64":
        values = rng.integers(-(2**62), 2**62, RING_SIZE, dtype=np.int64)
    else:
        values = rng.integers(0, 2**64 - 1, RING_SIZE, dtype=np.uint64)
    v = padded([int(value) for value in values])
    d = session.decrypt
    np.testing.assert_array_equal(d(a + values), expected(x + v))
    np.testing.assert_array_equal(d(values + a), expected(x + v))
    np.testing.assert_array_equal(d(a - values), expected(x - v))
    np.testing.assert_array_equal(d(values - a), expected(v - x))
    np.testing.assert_array_equal(d(a * values), expected(x * v))
    np.testing.assert_array_equal(d(values * a), expected(x * v))


def test_numpy_array_on_the_left(session, rng):
    x = random_values(rng).astype(object)
    a = session.encrypt(x)
    values = rng.integers(0, 1000, RING_SIZE)
    result = values + a
    assert isinstance(result, bfv.SessionCiphertext)
    np.testing.assert_array_equal(
        session.decrypt(result), expected(x + values.astype(object))
    )
    result = values * a
    assert isinstance(result, bfv.SessionCiphertext)
    np.testing.assert_array_equal(
        session.decrypt(result), expected(x * values.astype(object))
    )


def test_inplace_operators(session, rng):
    x = random_values(rng).astype(object)
    y = random_values(rng).astype(object)
    a = session.encrypt(x)
    b = session.encrypt(y)
    original = a
    a += b
    a -= 5
    a *= [2, 3, 4]
    a += np.arange(10)
    a *= b
    a -= -1
    assert a is original
    v = (((x + y - 5) * padded([2, 3, 4])) + padded(range(10))) * y + 1
    np.testing.assert_array_equal(session.decrypt(a), expected(v))
    # b is unchanged.
    np.testing.assert_array_equal(session.decrypt(b), expected(y))


def test_self_operations(session, rng):
    x = random_values(rng).astype(object)
    a = session.encrypt(x)
    np.testing.assert_array_equal(session.decrypt(a * a), expected(x * x))
    a += a
    np.testing.assert_array_equal(session.decrypt(a), expected(2 * x))


def test_unsupported_operands(session):
    a = session.encrypt([1, 2, 3])
    plain = session.encoder().encode([1])
    for value in [1.5, "x", None, plain, np.array([1.5]), object()]:
        with pytest.raises(TypeError):
            a + value
        with pytest.raises(TypeError):
            value * a
        with pytest.raises(TypeError):
            a -= value
    with pytest.raises(ValueError):
        a + np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError):
        a * np.zeros(RING_SIZE + 1, dtype=np.int64)


# ----------------------------------------------------------------------
# Rotations
# ----------------------------------------------------------------------


@pytest.mark.parametrize("shift", [1, 3, -2, 0, RING_SIZE // 2 + 1])
def test_rotate(session, rng, shift):
    x = random_values(rng).astype(object)
    a = session.encrypt(x)
    np.testing.assert_array_equal(
        session.decrypt(a.rotate(shift)), expected(rotate_rows(x, shift))
    )


def test_rotate_generates_new_keys(rng):
    s = bfv.Session(make_params(depth=1, poly_modulus_degree=4096))
    x = rng.integers(0, PLAIN_MODULUS, 4096).astype(object)
    a = s.encrypt(x)
    for shift in [5, -7, 5]:
        rows = x.reshape(2, -1)
        rotated = np.roll(rows, -shift, axis=1).reshape(-1)
        np.testing.assert_array_equal(
            s.decrypt(a.rotate(shift)), expected(rotated)
        )


def test_rotate_columns(session, rng):
    x = random_values(rng).astype(object)
    a = session.encrypt(x)
    swapped = x.reshape(2, -1)[::-1].reshape(-1)
    np.testing.assert_array_equal(
        session.decrypt(a.rotate_columns()), expected(swapped)
    )


def test_rotate_hoisted(session, rng):
    x = random_values(rng).astype(object)
    a = session.encrypt(x)
    shifts = [1, 3, -2]
    results = a.rotate_hoisted(shifts)
    assert isinstance(results, list)
    assert len(results) == len(shifts)
    for shift, result in zip(shifts, results):
        np.testing.assert_array_equal(
            session.decrypt(result), expected(rotate_rows(x, shift))
        )


def test_sum_slots_and_inner_product(session, rng):
    x = rng.integers(0, 100, RING_SIZE).astype(object)
    y = rng.integers(0, 100, RING_SIZE).astype(object)
    a = session.encrypt(x)
    b = session.encrypt(y)
    total = sum(x) % PLAIN_MODULUS
    np.testing.assert_array_equal(
        session.decrypt(a.sum_slots()), np.full(RING_SIZE, total)
    )
    dot = sum(x * y) % PLAIN_MODULUS
    np.testing.assert_array_equal(
        session.decrypt(a.inner_product(b)), np.full(RING_SIZE, dot)
    )
    np.testing.assert_array_equal(
        session.decrypt(a.inner_product(y)), np.full(RING_SIZE, dot)
    )
    np.testing.assert_array_equal(
        session.decrypt(a.inner_product([1, 2])),
        np.full(RING_SIZE, (x[0] + 2 * x[1]) % PLAIN_MODULUS),
    )
    np.testing.assert_array_equal(
        session.decrypt(a.inner_product(-1)),
        np.full(RING_SIZE, (-sum(x)) % PLAIN_MODULUS),
    )
    with pytest.raises(TypeError):
        a.inner_product([0.5])


def test_add_rotations_and_galois_key(rng):
    s = bfv.Session(make_params(depth=1, poly_modulus_degree=4096))
    s.add_rotations([4, -4])
    key = s.galois_key()
    assert isinstance(key, bfv.GaloisKey)
    x = rng.integers(0, PLAIN_MODULUS, 4096).astype(object)
    a = s.encrypt(x)
    rows = x.reshape(2, -1)
    np.testing.assert_array_equal(
        s.decrypt(a.rotate(-4)),
        expected(np.roll(rows, 4, axis=1).reshape(-1)),
    )


def test_concurrent_rotations(rng):
    """Threads generate rotation keys and rotate at the same time."""
    s = bfv.Session(make_params(depth=1, poly_modulus_degree=4096))
    x = rng.integers(0, PLAIN_MODULUS, 4096).astype(object)
    a = s.encrypt(x)
    shifts = [1, 2, 3, 5, 7, 11, 2, 1]

    def work(shift):
        return shift, s.decrypt(a.rotate(shift))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, shifts))
    for shift, result in results:
        rows = x.reshape(2, -1)
        np.testing.assert_array_equal(
            result, expected(np.roll(rows, -shift, axis=1).reshape(-1))
        )


# ----------------------------------------------------------------------
# Logic gates on bits
# ----------------------------------------------------------------------


def test_logic_gates(session, rng):
    p = rng.integers(0, 2, RING_SIZE)
    q = rng.integers(0, 2, RING_SIZE)
    a = session.encrypt(p)
    b = session.encrypt(q)
    gates = {
        bfv.AND: p & q,
        bfv.OR: p | q,
        bfv.XOR: p ^ q,
        bfv.NAND: 1 - (p & q),
        bfv.NOR: 1 - (p | q),
        bfv.XNOR: 1 - (p ^ q),
    }
    for gate, truth in gates.items():
        np.testing.assert_array_equal(
            session.decrypt(gate(a, b)), truth, err_msg=gate.__name__
        )
    np.testing.assert_array_equal(session.decrypt(bfv.NOT(a)), 1 - p)
    nested = bfv.XOR(bfv.AND(a, b), bfv.NOT(b))
    np.testing.assert_array_equal(session.decrypt(nested), (p & q) ^ (1 - q))


def test_logic_gates_reject(session):
    a = session.encrypt([1, 0])
    with pytest.raises(TypeError):
        bfv.AND(a, 1)
    with pytest.raises(hg.StateError):
        bfv.NOT(bfv.SessionCiphertext())


# ----------------------------------------------------------------------
# Lower level objects
# ----------------------------------------------------------------------


def test_accessors(session):
    assert isinstance(session.context(), bfv.Context)
    assert session.context().poly_modulus_degree == RING_SIZE
    assert session.context().plain_modulus == PLAIN_MODULUS
    assert isinstance(session.secret_key(), bfv.SecretKey)
    assert isinstance(session.public_key(), bfv.PublicKey)
    assert isinstance(session.relin_key(), bfv.RelinKey)
    assert isinstance(session.galois_key(), bfv.GaloisKey)
    assert isinstance(session.encoder(), bfv.Encoder)
    assert isinstance(session.encryptor(), bfv.Encryptor)
    assert isinstance(session.decryptor(), bfv.Decryptor)
    assert isinstance(session.operators(), bfv.ArithmeticOperator)
    assert session.encoder().slot_count == RING_SIZE


def test_lower_level_interop(session, rng):
    x = random_values(rng).astype(object)
    y = random_values(rng).astype(object)
    a = session.encrypt(x)
    raw = a.ciphertext
    assert isinstance(raw, bfv.Ciphertext)
    assert raw.ring_size == RING_SIZE
    # The lower level objects of the session work on its ciphertexts.
    encoder = session.encoder()
    plain = session.decryptor().decrypt(raw)
    np.testing.assert_array_equal(encoder.decode(plain), expected(x))
    low = session.encryptor().encrypt(encoder.encode(y))
    total = session.operators().add(raw, low)
    np.testing.assert_array_equal(
        session.decrypt(session.wrap(total)), expected(x + y)
    )


def test_accessors_keep_session_alive(rng):
    s = bfv.Session(make_params(depth=1, poly_modulus_degree=4096))
    encoder = s.encoder()
    decryptor = s.decryptor()
    c = s.encrypt([1, 2, 3])
    raw = c.ciphertext
    del s, c
    import gc

    gc.collect()
    np.testing.assert_array_equal(
        encoder.decode(decryptor.decrypt(raw))[:3], [1, 2, 3]
    )


def test_ciphertext_outlives_session(rng):
    s = bfv.Session(make_params(depth=1, poly_modulus_degree=4096))
    data = s.to_bytes(hg.SessionRole.OWNER)
    c = s.encrypt([4, 5, 6])
    del s
    d = c * c + 1
    owner = bfv.Session.from_bytes(data)
    result = owner.decrypt(owner.ciphertext_from_bytes(d.to_bytes()))
    np.testing.assert_array_equal(result[:4], [17, 26, 37, 1])


def test_wrap_context_mismatch(session):
    other = bfv.Session(make_params(depth=1, poly_modulus_degree=4096))
    raw = other.encrypt([1]).ciphertext
    with pytest.raises(hg.ContextMismatchError):
        session.wrap(raw)
    with pytest.raises(hg.Error):
        session.ciphertext_from_bytes(raw.to_bytes())


def test_mixing_sessions(session):
    other = bfv.Session(make_params(depth=1, poly_modulus_degree=4096))
    a = session.encrypt([1])
    b = other.encrypt([1])
    with pytest.raises(hg.ContextMismatchError):
        a + b
    with pytest.raises(hg.ContextMismatchError):
        bfv.AND(a, b)
    with pytest.raises(hg.ContextMismatchError):
        other.decrypt(a)


def test_empty_ciphertext(session):
    empty = bfv.SessionCiphertext()
    assert empty.is_empty
    assert repr(empty) == "<heongpu.bfv.SessionCiphertext empty>"
    a = session.encrypt([1])
    with pytest.raises(hg.StateError):
        empty + 1
    with pytest.raises(hg.StateError):
        a + empty
    with pytest.raises(hg.StateError):
        empty.rotate(1)
    with pytest.raises(hg.StateError):
        empty.to_bytes()
    with pytest.raises(hg.StateError):
        empty.plain_modulus
    with pytest.raises(hg.StateError):
        session.decrypt(empty)


def test_repr(session):
    a = session.encrypt([1])
    assert repr(a) == (
        f"<heongpu.bfv.SessionCiphertext ring_size={RING_SIZE} "
        f"coeff_modulus_count={a.ciphertext.coeff_modulus_count} "
        f"plain_modulus={PLAIN_MODULUS}>"
    )


def test_copy(session, rng):
    x = random_values(rng).astype(object)
    a = session.encrypt(x)
    for b in [copy.copy(a), copy.deepcopy(a), a.copy()]:
        b += 1
        np.testing.assert_array_equal(session.decrypt(b), expected(x + 1))
    np.testing.assert_array_equal(session.decrypt(a), expected(x))
    view = copy.copy(session)
    assert view.role == hg.SessionRole.OWNER
    np.testing.assert_array_equal(view.decrypt(a), expected(x))


def test_no_pickling(session):
    with pytest.raises(TypeError, match="to_bytes"):
        pickle.dumps(session.encrypt([1]))
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(session)
    with pytest.raises(TypeError):
        copy.deepcopy(session)


# ----------------------------------------------------------------------
# Roles and serialization
# ----------------------------------------------------------------------


def test_session_save_load(session, tmp_path, rng):
    x = random_values(rng)
    path = tmp_path / "owner.bin"
    session.save(path, hg.SessionRole.OWNER)
    loaded = bfv.Session.load(path)
    assert loaded.role == hg.SessionRole.OWNER
    assert loaded.slot_count == RING_SIZE
    assert loaded.plain_modulus == PLAIN_MODULUS
    # The loaded owner has the same keys.
    np.testing.assert_array_equal(
        loaded.decrypt(
            loaded.ciphertext_from_bytes(session.encrypt(x).to_bytes())
        ),
        x,
    )
    buffer = io.BytesIO()
    session.save(buffer, hg.SessionRole.EVALUATOR)
    buffer.seek(0)
    loaded = bfv.Session.load(buffer, expected_role=hg.SessionRole.EVALUATOR)
    assert loaded.role == hg.SessionRole.EVALUATOR
    loaded = bfv.Session.load(str(path), hg.SessionRole.OWNER)
    assert loaded.role == hg.SessionRole.OWNER


def test_role_sizes(session):
    owner = session.to_bytes(hg.SessionRole.OWNER)
    evaluator = session.to_bytes(hg.SessionRole.EVALUATOR)
    encryptor = session.to_bytes(hg.SessionRole.ENCRYPTOR)
    assert len(encryptor) < len(evaluator) < len(owner)


def test_expected_role_mismatch(evaluator_bytes):
    with pytest.raises(hg.StateError, match="evaluator"):
        bfv.Session.from_bytes(
            evaluator_bytes, expected_role=hg.SessionRole.OWNER
        )


def test_corrupted_session_bytes(session, evaluator_bytes):
    with pytest.raises(hg.SerializationError):
        bfv.Session.from_bytes(b"not a session")
    # A truncated header.
    with pytest.raises(hg.SerializationError):
        bfv.Session.from_bytes(evaluator_bytes[:12])
    # Another object than a session.
    with pytest.raises(hg.SerializationError):
        bfv.Session.from_bytes(session.encrypt([1]).to_bytes())


def test_evaluator_role(session, evaluator, rng):
    assert evaluator.role == hg.SessionRole.EVALUATOR
    x = random_values(rng).astype(object)
    c = evaluator.encrypt(x)
    with pytest.raises(hg.StateError):
        evaluator.decrypt(c)
    with pytest.raises(hg.StateError):
        evaluator.decrypt(c, signed=True)
    with pytest.raises(hg.StateError):
        evaluator.noise_budget(c)
    with pytest.raises(hg.StateError):
        evaluator.secret_key()
    with pytest.raises(hg.StateError):
        evaluator.decryptor()
    with pytest.raises(hg.StateError):
        evaluator.to_bytes(hg.SessionRole.OWNER)
    # The evaluator multiplies and rotates by the shifts it was given.
    r = (c * c).rotate(3) + c.rotate(-2)
    with pytest.raises(hg.KeyNotFoundError):
        c.rotate(5)
    with pytest.raises(hg.KeyNotFoundError):
        evaluator.add_rotations([6])
    owner = session
    result = owner.decrypt(owner.ciphertext_from_bytes(r.to_bytes()))
    np.testing.assert_array_equal(
        result, expected(rotate_rows(x * x, 3) + rotate_rows(x, -2))
    )
    # An evaluator can pass on the keys of the evaluator or encryptor.
    again = bfv.Session.from_bytes(evaluator.to_bytes(hg.SessionRole.EVALUATOR))
    assert again.role == hg.SessionRole.EVALUATOR


def test_encryptor_role(session, encryptor, rng):
    assert encryptor.role == hg.SessionRole.ENCRYPTOR
    x = random_values(rng).astype(object)
    c = encryptor.encrypt(x)
    d = (c + 5) * 3 - [1, 2]
    with pytest.raises(hg.KeyNotFoundError):
        c * c
    with pytest.raises(hg.KeyNotFoundError):
        c.rotate(1)
    with pytest.raises(hg.KeyNotFoundError):
        encryptor.relin_key()
    with pytest.raises(hg.StateError):
        encryptor.decrypt(c)
    with pytest.raises(hg.StateError):
        encryptor.to_bytes(hg.SessionRole.EVALUATOR)
    result = session.decrypt(session.ciphertext_from_bytes(d.to_bytes()))
    np.testing.assert_array_equal(
        result, expected((x + 5) * 3 - padded([1, 2]))
    )


def test_client_server_flow(session, evaluator_bytes, rng, tmp_path):
    """The client keeps the secret key; the server gets the evaluation keys
    and ciphertexts as bytes, computes, and sends the result back."""
    x = rng.integers(0, 1000, RING_SIZE).astype(object)
    y = rng.integers(0, 1000, RING_SIZE).astype(object)
    request = [session.encrypt(x).to_bytes(), session.encrypt(y).to_bytes()]

    server = bfv.Session.from_bytes(
        evaluator_bytes, expected_role=hg.SessionRole.EVALUATOR
    )
    a, b = (server.ciphertext_from_bytes(data) for data in request)
    response = (a * b + a.rotate(1)).inner_product([1] * 16)
    path = tmp_path / "response.bin"
    response.save(path)

    result = session.decrypt(session.load_ciphertext(path))
    product = x * y + rotate_rows(x, 1)
    np.testing.assert_array_equal(
        result, np.full(RING_SIZE, sum(product[:16]) % PLAIN_MODULUS)
    )
    with open(path, "rb") as file:
        again = session.load_ciphertext(file)
    np.testing.assert_array_equal(session.decrypt(again), result)


def test_load_lower_level_ciphertext(session, rng):
    x = random_values(rng)
    raw = session.encryptor().encrypt(session.encoder().encode(x))
    loaded = session.ciphertext_from_bytes(raw.to_bytes())
    np.testing.assert_array_equal(session.decrypt(loaded), x)
    buffer = io.BytesIO(raw.to_bytes())
    np.testing.assert_array_equal(
        session.decrypt(session.load_ciphertext(buffer)), x
    )
    # HEonGPU reports some malformed ciphertexts with a plain RuntimeError,
    # the base class of SerializationError.
    with pytest.raises(RuntimeError):
        session.ciphertext_from_bytes(b"garbage")


# ----------------------------------------------------------------------
# Streams
# ----------------------------------------------------------------------


def test_on_stream(session, rng):
    x = random_values(rng).astype(object)
    y = random_values(rng).astype(object)
    with hg.Stream() as stream:
        view = session.on_stream(stream)
        assert view.stream == stream.handle
        assert view.role == hg.SessionRole.OWNER
        assert session.stream == 0
        a = view.encrypt(x)
        b = view.encrypt(y)
        assert a.stream == stream.handle
        c = (a * b + 7).rotate(1)
        assert c.stream == stream.handle
        result = view.decrypt(c)
        view.synchronize()
        np.testing.assert_array_equal(
            result, expected(rotate_rows(x * y + 7, 1))
        )
        # The default session reads the ciphertexts of the view.
        np.testing.assert_array_equal(session.decrypt(a), expected(x))


def test_on_stream_handles(session):
    view = session.on_stream(None)
    assert view.stream == 0
    assert (
        bfv.Session(
            make_params(depth=1, poly_modulus_degree=4096), rotations=None
        ).role
        == hg.SessionRole.OWNER
    )
    data = session.to_bytes(hg.SessionRole.ENCRYPTOR)
    assert bfv.Session.from_bytes(data, expected_role=None).role == (
        hg.SessionRole.ENCRYPTOR
    )
    with hg.Stream() as stream:
        view = session.on_stream(stream.handle)
        assert view.stream == stream.handle
        np.testing.assert_array_equal(
            view.decrypt(view.encrypt([1, 2]))[:2], [1, 2]
        )
    with pytest.raises(TypeError):
        session.on_stream("not a stream")


def test_threads_with_streams(session, rng):
    values = [rng.integers(0, 1000, RING_SIZE).astype(object) for _ in range(4)]

    def work(x):
        with hg.Stream() as stream:
            view = session.on_stream(stream)
            c = view.encrypt(x)
            return view.decrypt(c * c - c)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, values))
    for x, result in zip(values, results):
        np.testing.assert_array_equal(result, expected(x * x - x))


# ----------------------------------------------------------------------
# Larger parameters
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_deep_circuit_large_ring(rng):
    params = bfv.Params(
        multiplicative_depth=6,
        plain_modulus=PLAIN_MODULUS,
        poly_modulus_degree=32768,
    )
    s = bfv.Session(params)
    assert s.slot_count == 32768
    x = rng.integers(0, PLAIN_MODULUS, s.slot_count).astype(object)
    c = s.encrypt(x)
    power = x.copy()
    for _ in range(6):
        c = c * c
        power = power * power % PLAIN_MODULUS
    assert s.noise_budget(c) > 0
    np.testing.assert_array_equal(s.decrypt(c), expected(power))
    np.testing.assert_array_equal(
        s.decrypt(c.sum_slots()),
        np.full(s.slot_count, sum(power) % PLAIN_MODULUS),
    )
