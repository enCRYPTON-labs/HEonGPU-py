# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

import copy
import gc
import io
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.testing import assert_allclose

import heongpu as hg
from heongpu import ckks

TOLERANCE = 1e-5
Role = hg.SessionRole


@pytest.fixture(scope="module")
def session():
    return ckks.Session(ckks.Params(multiplicative_depth=4), rotations=[1])


@pytest.fixture(scope="module")
def other_session():
    return ckks.Session(multiplicative_depth=2)


@pytest.fixture(scope="module")
def xs(session):
    return 0.5 * np.sin(0.01 * np.arange(session.slot_count))


@pytest.fixture(scope="module")
def ys(session):
    return 0.5 * np.sin(0.01 * np.arange(session.slot_count) + 1.0)


@pytest.fixture(scope="module")
def x(session, xs):
    return session.encrypt(xs)


@pytest.fixture(scope="module")
def y(session, ys):
    return session.encrypt(ys)


def check(session, cipher, expected, tolerance=TOLERANCE):
    result = session.decrypt(cipher)
    expected = np.asarray(expected, dtype=np.float64)
    assert_allclose(result[: expected.size], expected, atol=tolerance)
    return result


# ----------------------------------------------------------------------
# Construction and properties
# ----------------------------------------------------------------------


def test_session_properties(session):
    assert session.role == Role.OWNER
    assert session.slot_count == 8192
    assert session.scale == 2.0**40
    assert session.rescale_mode == hg.RescaleMode.AUTOMATIC
    assert session.stream == 0
    assert not session.has_bootstrapping
    with pytest.raises(hg.StateError):
        session.bootstrapping_levels()
    text = repr(session)
    assert "Session" in text and "OWNER" in text and "AUTOMATIC" in text


def test_session_from_parameter_set():
    parameters = ckks.Params(
        multiplicative_depth=2, rescale_mode=hg.RescaleMode.MANUAL
    ).generate()
    session = ckks.Session(parameters)
    assert session.rescale_mode == hg.RescaleMode.MANUAL
    values = np.linspace(-1, 1, 16)
    check(session, session.encrypt(values), values)


def test_session_from_keywords():
    session = ckks.Session(
        multiplicative_depth=2, scale_bit_size=45, rotations=2
    )
    assert session.scale == 2.0**45
    values = np.arange(8) / 8
    check(session, session.encrypt(values).rotate(2), values[2:])


def test_session_invalid_arguments():
    with pytest.raises(hg.ParameterError):
        ckks.Session(multiplicative_depth=100, poly_modulus_degree=8192)
    with pytest.raises(TypeError):
        ckks.Session("not parameters")
    with pytest.raises(TypeError):
        ckks.Session()


# ----------------------------------------------------------------------
# Encryption and decryption
# ----------------------------------------------------------------------


def test_encrypt_decrypt(session, xs, x):
    assert x.level == x.levels_left == 4
    assert x.depth == 0
    assert x.scale == session.scale
    assert x.stream == 0
    result = check(session, x, xs)
    assert result.dtype == np.float64
    assert result.shape == (session.slot_count,)
    assert "SessionCiphertext" in repr(x)


def test_encrypt_sequences_and_scalars(session):
    check(session, session.encrypt([0.25, -0.5, 1]), [0.25, -0.5, 1, 0, 0])
    check(session, session.encrypt((1, 2, 3)), [1, 2, 3, 0])
    check(session, session.encrypt(0.75), np.full(session.slot_count, 0.75))
    check(session, session.encrypt(np.float32(2)), np.full(16, 2.0))
    check(session, session.encrypt(np.arange(4, dtype=np.int32)), range(4))
    with pytest.raises(ValueError):
        session.encrypt(np.zeros((2, 2)))
    with pytest.raises(TypeError):
        session.encrypt("text")


def test_encrypt_complex(session, rng):
    n = session.slot_count
    values = rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)
    cipher = session.encrypt(values)
    result = session.decrypt_complex(cipher)
    assert result.dtype == np.complex128
    assert_allclose(result, values, atol=TOLERANCE)
    assert_allclose(session.decrypt(cipher), values.real, atol=TOLERANCE)
    assert_allclose(
        session.decrypt_complex(session.encrypt(0.5 - 0.25j)),
        np.full(n, 0.5 - 0.25j),
        atol=TOLERANCE,
    )
    # Products and conjugation of complex slots.
    other = values[::-1].copy()
    product = cipher * session.encrypt(other)
    assert_allclose(
        session.decrypt_complex(product), values * other, atol=TOLERANCE
    )
    assert_allclose(
        session.decrypt_complex(cipher.conjugate()),
        np.conj(values),
        atol=TOLERANCE,
    )


def test_encrypt_plaintext_and_wrap(session, xs):
    plain = session.encoder().encode(xs)
    cipher = session.encrypt(plain)
    check(session, cipher, xs)

    raw = session.encryptor().encrypt(plain)
    wrapped = session.wrap(raw)
    check(session, wrapped, xs)
    # wrap copies the ciphertext.
    assert wrapped.ciphertext is not raw


def test_wrap_other_parameters(session, other_session, xs):
    cipher = other_session.encrypt(xs[: other_session.slot_count])
    with pytest.raises(hg.ContextMismatchError):
        session.wrap(cipher.ciphertext)


def test_underlying_ciphertext(session, x):
    inner = x.ciphertext
    assert isinstance(inner, ckks.Ciphertext)
    assert inner.level == x.level
    assert inner.size == 2
    plain = session.decryptor().decrypt(inner)
    assert_allclose(
        session.encoder().decode(plain), session.decrypt(x), atol=1e-9
    )


# ----------------------------------------------------------------------
# Arithmetic
# ----------------------------------------------------------------------

OPERAND_KINDS = ["ciphertext", "int", "float", "numpy_scalar", "list", "array"]


def operand(kind, session, values):
    """Returns the operand of the given kind and its plaintext value."""
    if kind == "ciphertext":
        return session.encrypt(values), values
    if kind == "int":
        return 3, np.full(values.size, 3.0)
    if kind == "float":
        return 0.25, np.full(values.size, 0.25)
    if kind == "numpy_scalar":
        return np.float64(-0.5), np.full(values.size, -0.5)
    if kind == "list":
        return list(values[:64]), np.concatenate(
            [values[:64], np.zeros(values.size - 64)]
        )
    return values, values


@pytest.mark.parametrize("kind", OPERAND_KINDS)
def test_binary_operators(session, xs, ys, x, kind):
    other, expected = operand(kind, session, ys)
    check(session, x + other, xs + expected)
    check(session, x - other, xs - expected)
    check(session, x * other, xs * expected)
    check(session, other + x, expected + xs)
    check(session, other - x, expected - xs)
    check(session, other * x, expected * xs)


@pytest.mark.parametrize("kind", OPERAND_KINDS)
def test_inplace_operators(session, xs, ys, kind):
    other, expected = operand(kind, session, ys)
    cipher = session.encrypt(xs)
    original = cipher
    cipher += other
    cipher -= 0.125
    cipher *= other
    assert cipher is original
    check(session, cipher, (xs + expected - 0.125) * expected)


def test_numpy_array_on_the_left(session, xs, ys, x):
    result = ys + x
    assert isinstance(result, ckks.SessionCiphertext)
    check(session, result, ys + xs)
    result = ys * x
    assert isinstance(result, ckks.SessionCiphertext)
    check(session, result, ys * xs)
    check(session, ys - x, ys - xs)


def test_unary_division_and_power(session, xs, x):
    check(session, -x, -xs)
    positive = +x
    assert positive is not x
    check(session, positive, xs)
    check(session, x / 4, xs / 4)
    check(session, x**1, xs)
    check(session, x**2, xs**2)
    check(session, x**3, xs**3)
    check(session, x**5, xs**5)
    with pytest.raises(ZeroDivisionError):
        x / 0
    with pytest.raises(ValueError):
        x**0
    with pytest.raises(TypeError):
        x / x


def test_circuit(session, xs, ys, x, y):
    z = x * x * y + 0.5 * x - y
    check(session, z, xs * xs * ys + 0.5 * xs - ys)
    assert z.levels_left == x.levels_left - 2


def test_unsupported_operands(x):
    with pytest.raises(TypeError):
        x + "text"
    with pytest.raises(TypeError):
        x * None
    with pytest.raises(TypeError):
        x + object()
    with pytest.raises(TypeError):
        x + (1 + 2j)
    with pytest.raises(TypeError):
        x * np.array([1j, 2j])
    with pytest.raises(ValueError):
        x + np.zeros((2, 2))


def test_too_many_values(session, x):
    with pytest.raises(ValueError):
        x + np.zeros(session.slot_count + 1)


def test_ciphertexts_of_different_sessions(session, other_session, xs, x):
    foreign = other_session.encrypt(xs[: other_session.slot_count])
    with pytest.raises(hg.ContextMismatchError):
        x + foreign
    with pytest.raises(hg.ContextMismatchError):
        x * foreign
    with pytest.raises(hg.ContextMismatchError):
        session.decrypt(foreign)


def test_empty_ciphertext(session, x):
    empty = ckks.SessionCiphertext()
    with pytest.raises(hg.StateError):
        x + empty
    with pytest.raises(hg.StateError):
        empty * 2.0
    with pytest.raises(hg.StateError):
        session.decrypt(empty)
    with pytest.raises(hg.StateError):
        empty.levels_left


def test_copy(session, xs, x):
    duplicate = copy.copy(x)
    deep = copy.deepcopy(x)
    other = x.copy()
    for cipher in (duplicate, deep, other):
        assert cipher is not x
        assert cipher.ciphertext is not x.ciphertext
        check(session, cipher, xs)
    other += 1.0
    check(session, x, xs)


def test_pickle_is_refused(session, x):
    with pytest.raises(TypeError, match="to_bytes"):
        pickle.dumps(x)
    with pytest.raises(TypeError, match="to_bytes"):
        pickle.dumps(session)


# ----------------------------------------------------------------------
# Rescale modes and levels
# ----------------------------------------------------------------------


def test_levels_left(session, x):
    product = x * x
    assert product.levels_left == x.levels_left - 1
    assert (product * x).levels_left == x.levels_left - 2


def test_manual_rescale_mode(xs, ys):
    session = ckks.Session(
        multiplicative_depth=3, rescale_mode=hg.RescaleMode.MANUAL
    )
    n = session.slot_count
    xs, ys = xs[:n], ys[:n]
    x = session.encrypt(xs)
    y = session.encrypt(ys)
    z = x * y
    assert z.ciphertext.rescale_required
    assert not z.ciphertext.relinearization_required
    # The product waits for its rescale.
    assert z.levels_left == x.levels_left - 1
    check(session, z, xs * ys)
    check(session, z + 0.5, xs * ys + 0.5)
    if not hasattr(ckks, "ArithmeticOperator"):
        return
    operators = session.operators()
    assert isinstance(operators, ckks.ArithmeticOperator)
    operators.rescale_inplace(z.ciphertext)
    assert not z.ciphertext.rescale_required
    check(session, z * ys, xs * ys * ys, tolerance=1e-4)


# ----------------------------------------------------------------------
# Rotations and sums
# ----------------------------------------------------------------------


def test_rotations(session, xs, x):
    check(session, x.rotate(1), np.roll(xs, -1))
    check(session, x.rotate(-3), np.roll(xs, 3))
    check(session, x.rotate(0), xs)
    check(session, x.rotate(session.slot_count + 1), np.roll(xs, -1))


def test_add_rotations(session, xs, x):
    session.add_rotations([5, 6])
    session.add_rotations(7)
    for shift in (5, 6, 7):
        check(session, x.rotate(shift), np.roll(xs, -shift))
    # The Galois key of the session holds the new steps.
    galois_key = session.galois_key()
    assert isinstance(galois_key, ckks.GaloisKey)
    assert galois_key.is_on_device


def test_rotate_hoisted(session, xs, x):
    shifts = [1, -2, 7, 0]
    rotated = x.rotate_hoisted(shifts)
    assert isinstance(rotated, list) and len(rotated) == len(shifts)
    for shift, cipher in zip(shifts, rotated):
        check(session, cipher, np.roll(xs, -shift))
    assert x.rotate_hoisted(np.array([3])) and len(x.rotate_hoisted([])) == 0


def test_sum_slots_and_inner_product(session, xs, ys, x, y):
    sums = session.decrypt(x.sum_slots())
    assert_allclose(sums, np.full(session.slot_count, xs.sum()), atol=1e-2)
    dot = xs @ ys
    assert_allclose(session.decrypt(x.inner_product(y))[5], dot, atol=1e-2)
    assert_allclose(session.decrypt(x.inner_product(ys))[5], dot, atol=1e-2)
    assert_allclose(
        session.decrypt(x.inner_product(list(ys)))[0], dot, atol=1e-2
    )
    with pytest.raises(TypeError):
        x.inner_product(ys * 1j)


# ----------------------------------------------------------------------
# Polynomials
# ----------------------------------------------------------------------


def test_evaluate_polynomial(session, xs, x):
    coefficients = [0.5, -1.0, 0.25, 2.0]
    result = session.evaluate_polynomial(x, coefficients)
    expected = np.polynomial.polynomial.polyval(xs, coefficients)
    check(session, result, expected)
    check(
        session,
        session.evaluate_polynomial(x, np.array(coefficients)),
        expected,
    )


def test_evaluate_chebyshev_callable(session, xs, x):
    result = session.evaluate_chebyshev(x, np.exp, -1.0, 1.0, degree=7)
    check(session, result, np.exp(xs), tolerance=1e-4)
    calls = []

    def function(value):
        calls.append(value)
        return value * value

    result = session.evaluate_chebyshev(x, function, -1, 1, 4)
    assert len(calls) > 4
    check(session, result, xs * xs, tolerance=1e-4)
    # More operations on the result.
    check(session, result * x + 1.0, xs**3 + 1.0, tolerance=1e-4)


def test_evaluate_chebyshev_coefficients(session, xs, x):
    # T0 + 0.5 T1 + 0.25 T2 = 1 + 0.5 t + 0.25 (2 t^2 - 1) on [-1, 1].
    result = session.evaluate_chebyshev(x, [1.0, 0.5, 0.25], -1.0, 1.0)
    check(session, result, 0.75 + 0.5 * xs + 0.5 * xs * xs, tolerance=1e-4)


def test_evaluate_chebyshev_errors(session, x):
    def failing(value):
        raise RuntimeError("from the callable")

    with pytest.raises(RuntimeError, match="from the callable"):
        session.evaluate_chebyshev(x, failing, -1, 1, degree=5)
    with pytest.raises(TypeError):
        session.evaluate_chebyshev(x, lambda value: None, -1, 1, degree=5)
    with pytest.raises(TypeError, match="degree"):
        session.evaluate_chebyshev(x, np.sin, -1, 1)
    with pytest.raises(ValueError, match="degree"):
        session.evaluate_chebyshev(x, [1.0, 2.0], -1, 1, degree=3)


# ----------------------------------------------------------------------
# Logic
# ----------------------------------------------------------------------


def test_logic_gates(session):
    n = session.slot_count
    a = np.arange(n) % 2
    b = (np.arange(n) // 2) % 2
    ea = session.encrypt(a)
    eb = session.encrypt(b)
    check(session, ckks.NOT(ea), 1 - a)
    check(session, ckks.AND(ea, eb), a & b)
    check(session, ckks.OR(ea, eb), a | b)
    check(session, ckks.XOR(ea, eb), a ^ b)
    check(session, ckks.NAND(ea, eb), 1 - (a & b))
    check(session, ckks.NOR(ea, eb), 1 - (a | b))
    check(session, ckks.XNOR(ea, eb), 1 - (a ^ b))
    assert ckks.XOR(ea, eb).levels_left == ea.levels_left - 1
    with pytest.raises(TypeError):
        ckks.AND(ea, 1)


def test_gate_bootstrapping_needs_a_gate_preset(session, x):
    with pytest.raises(hg.StateError):
        session.AND_bootstrapping(x, x)
    with pytest.raises(hg.StateError):
        x.bootstrap()


# ----------------------------------------------------------------------
# Roles and serialization
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def owner():
    return ckks.Session(multiplicative_depth=2, rotations=[1])


def test_roles_save_and_load(owner, tmp_path, xs):
    n = owner.slot_count
    xs = xs[:n]
    paths = {}
    for role in (Role.OWNER, Role.EVALUATOR, Role.ENCRYPTOR):
        paths[role] = tmp_path / f"{role.name.lower()}.bin"
        owner.save(paths[role], role)

    evaluator = ckks.Session.load(paths[Role.EVALUATOR], Role.EVALUATOR)
    encryptor = ckks.Session.load(str(paths[Role.ENCRYPTOR]))
    with open(paths[Role.OWNER], "rb") as file:
        loaded_owner = ckks.Session.load(file, expected_role=Role.OWNER)
    assert evaluator.role == Role.EVALUATOR
    assert encryptor.role == Role.ENCRYPTOR
    assert loaded_owner.role == Role.OWNER
    for loaded in (evaluator, encryptor, loaded_owner):
        assert loaded.rescale_mode == hg.RescaleMode.AUTOMATIC
        assert loaded.slot_count == n
        assert loaded.scale == owner.scale

    # The encryptor encrypts for the owner, the evaluator computes.
    cipher = encryptor.encrypt(xs)
    data = cipher.to_bytes()
    on_server = evaluator.ciphertext_from_bytes(data)
    result = (on_server * on_server).rotate(1) + 0.5
    back = owner.ciphertext_from_bytes(result.to_bytes())
    check(owner, back, np.roll(xs * xs, -1) + 0.5)
    check(loaded_owner, loaded_owner.ciphertext_from_bytes(data), xs)

    # The evaluator cannot decrypt.
    with pytest.raises(hg.StateError):
        evaluator.decrypt(on_server)
    with pytest.raises(hg.StateError):
        evaluator.secret_key()
    with pytest.raises(hg.StateError):
        evaluator.decryptor()
    with pytest.raises(hg.StateError):
        evaluator.save(io.BytesIO(), Role.OWNER)
    # nor generate new rotation keys.
    with pytest.raises(hg.KeyNotFoundError):
        on_server.rotate(3)
    with pytest.raises(hg.KeyNotFoundError):
        evaluator.add_rotations([3])

    # The encryptor cannot compute products or decrypt.
    other = encryptor.encrypt(xs)
    with pytest.raises(hg.KeyNotFoundError):
        other * other
    with pytest.raises(hg.KeyNotFoundError):
        encryptor.relin_key()
    with pytest.raises(hg.StateError):
        encryptor.decrypt(other)
    with pytest.raises(hg.StateError):
        encryptor.save(io.BytesIO(), Role.EVALUATOR)

    # The role is checked before any key is read.
    with pytest.raises(hg.StateError):
        ckks.Session.load(paths[Role.OWNER], expected_role=Role.EVALUATOR)


def test_client_server_bytes(owner, xs):
    n = owner.slot_count
    xs = xs[:n]
    # Client: the keys of the evaluator and an encrypted input.
    keys = owner.to_bytes(Role.EVALUATOR)
    request = owner.encrypt(xs).to_bytes()
    assert isinstance(keys, bytes) and isinstance(request, bytes)

    # Server
    server = ckks.Session.from_bytes(keys, expected_role=Role.EVALUATOR)
    with pytest.raises(hg.StateError):
        ckks.Session.from_bytes(keys, Role.OWNER)
    value = server.ciphertext_from_bytes(request)
    response = (value * value - value).to_bytes()

    # Client
    check(owner, owner.ciphertext_from_bytes(response), xs * xs - xs)


def test_load_ciphertext(owner, tmp_path, xs):
    xs = xs[: owner.slot_count]
    cipher = owner.encrypt(xs)
    path = tmp_path / "cipher.bin"
    cipher.save(path)
    check(owner, owner.load_ciphertext(path), xs)
    buffer = io.BytesIO()
    cipher.save(buffer)
    buffer.seek(0)
    check(owner, owner.load_ciphertext(buffer), xs)
    # Ciphertexts of the lower level API.
    raw = cipher.ciphertext.to_bytes()
    check(owner, owner.ciphertext_from_bytes(raw), xs)
    assert ckks.Ciphertext.from_bytes(cipher.to_bytes()).level == cipher.level


def test_load_errors(owner, session, xs):
    with pytest.raises(hg.SerializationError):
        ckks.Session.from_bytes(b"not a session")
    # A truncated ciphertext raises SerializationError or RuntimeError.
    with pytest.raises(RuntimeError):
        owner.ciphertext_from_bytes(b"not a ciphertext")
    foreign = session.encrypt(xs).to_bytes()
    with pytest.raises(hg.ContextMismatchError):
        owner.ciphertext_from_bytes(foreign)
    with pytest.raises(TypeError):
        owner.load_ciphertext(io.StringIO("text"))


def test_manual_rescale_mode_is_saved():
    session = ckks.Session(
        multiplicative_depth=2, rescale_mode=hg.RescaleMode.MANUAL
    )
    loaded = ckks.Session.from_bytes(session.to_bytes(Role.EVALUATOR))
    assert loaded.rescale_mode == hg.RescaleMode.MANUAL


# ----------------------------------------------------------------------
# Lifetime and accessors
# ----------------------------------------------------------------------


def test_ciphertext_outlives_session(xs):
    session = ckks.Session(multiplicative_depth=2)
    xs = xs[: session.slot_count]
    cipher = session.encrypt(xs)
    context = session.context
    secret_key = session.secret_key()
    encoder = session.encoder()
    del session
    gc.collect()
    result = cipher * cipher + 1.0
    decryptor = ckks.Decryptor(context, secret_key)
    plain = decryptor.decrypt(result.ciphertext)
    assert_allclose(encoder.decode(plain), xs * xs + 1.0, atol=TOLERANCE)


def test_accessors(session):
    context = session.context
    assert isinstance(context, ckks.Context)
    assert context.slot_count == session.slot_count
    assert context.rescale_mode == hg.RescaleMode.AUTOMATIC
    assert isinstance(session.secret_key(), ckks.SecretKey)
    assert isinstance(session.public_key(), ckks.PublicKey)
    assert isinstance(session.relin_key(), ckks.RelinKey)
    assert isinstance(session.galois_key(), ckks.GaloisKey)
    assert isinstance(session.encoder(), ckks.Encoder)
    assert isinstance(session.encryptor(), ckks.Encryptor)
    assert isinstance(session.decryptor(), ckks.Decryptor)
    # References into the session, not copies.
    assert session.encoder() is session.encoder()
    assert session.secret_key().ring_size == 2 * session.slot_count
    if hasattr(ckks, "ArithmeticOperator"):
        assert isinstance(session.operators(), ckks.ArithmeticOperator)


# ----------------------------------------------------------------------
# Streams
# ----------------------------------------------------------------------


# The streams below are blocking streams: HEonGPU 2.1.0 copies an operand
# that the automatic rescale mode brings to another level on the stream of
# that operand, so on a non-blocking stream an operand of the default stream
# is not ordered with the operation (see the xfail test below).


def test_on_stream(session, xs, ys, x):
    with hg.Stream(non_blocking=False) as stream:
        view = session.on_stream(stream)
        assert view.stream == stream.handle
        assert session.stream == 0
        assert view.role == Role.OWNER
        a = view.encrypt(xs)
        assert a.stream == stream.handle
        result = (a * x).rotate(1) + ys
        assert result.stream == stream.handle
        view.synchronize()
        check(view, result, np.roll(xs * xs, -1) + ys)
        # Ciphertexts of the view belong to the same session; the
        # difference runs on the stream of its left operand, so wait for
        # it before the default stream reads it.
        difference = result - x
        assert difference.stream == stream.handle
        view.synchronize()
        check(session, difference, np.roll(xs * xs, -1) + ys - xs)

    # Integer handles work too.
    view = session.on_stream(0)
    assert view.stream == 0


def test_threads_with_stream_views(session, xs, x):
    reference = session.decrypt((x * x).rotate(1) + x)
    streams = [hg.Stream(non_blocking=False) for _ in range(3)]

    def work(index):
        view = session.on_stream(streams[index])
        results = []
        for round in range(2):
            a = view.encrypt(xs)
            b = (a * x).rotate(1) + a
            shift = 11 + index + 3 * round
            rotated = view.decrypt(a.rotate(shift))
            results.append((view.decrypt(b), rotated[0], xs[shift]))
        return results

    with ThreadPoolExecutor(len(streams)) as pool:
        outcomes = list(pool.map(work, range(len(streams))))
    for results in outcomes:
        for decrypted, rotated, expected in results:
            assert_allclose(decrypted, reference, atol=TOLERANCE)
            assert abs(rotated - expected) < TOLERANCE
    for stream in streams:
        stream.close()


@pytest.mark.xfail(
    strict=False,
    reason="HEonGPU 2.1.0 aligns the level of an operand of the default "
    "stream on that stream, unordered with a non-blocking stream",
)
def test_non_blocking_stream_with_default_stream_operand(session, xs, ys, x):
    with hg.Stream(non_blocking=True) as stream:
        view = session.on_stream(stream)
        for _ in range(3):
            result = (view.encrypt(xs) * x).rotate(1) + ys
            view.synchronize()
            check(view, result, np.roll(xs * xs, -1) + ys)
            difference = result - x
            view.synchronize()
            check(view, difference, np.roll(xs * xs, -1) + ys - xs)


# ----------------------------------------------------------------------
# Bootstrapping
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_slim_bootstrapping():
    session = ckks.Session(ckks.BootstrappingPreset.slim(2))
    assert session.has_bootstrapping
    assert session.rescale_mode == hg.RescaleMode.AUTOMATIC
    levels = session.bootstrapping_levels()
    assert isinstance(levels, ckks.BootstrappingLevels)
    assert levels.usable_levels == 2
    n = session.slot_count
    xs = 0.5 * np.sin(0.01 * np.arange(n))
    x = session.encrypt(xs)
    assert x.level == levels.output_level
    y = x * x
    assert y.levels_left == 1
    b = y.bootstrap()
    assert b.levels_left == 2
    check(session, b, xs * xs, tolerance=2.0**-15)

    low = b * b * b * b
    assert low.levels_left == -1
    with pytest.raises(hg.LevelMismatchError):
        low.bootstrap()
    with pytest.raises(hg.StateError):
        session.AND_bootstrapping(x, x)

    # The evaluator gets the bootstrapping keys. One session at a time is
    # kept on the GPU.
    evaluator_keys = session.to_bytes(Role.EVALUATOR)
    owner_keys = session.to_bytes(Role.OWNER)
    encryptor_keys = session.to_bytes(Role.ENCRYPTOR)
    assert len(encryptor_keys) < len(evaluator_keys) < len(owner_keys)
    request = y.to_bytes()
    del session, x, y, b, low
    gc.collect()

    evaluator = ckks.Session.from_bytes(evaluator_keys, Role.EVALUATOR)
    assert evaluator.has_bootstrapping
    remote = evaluator.ciphertext_from_bytes(request)
    response = remote.bootstrap().to_bytes()
    del evaluator, remote
    gc.collect()

    encryptor = ckks.Session.from_bytes(encryptor_keys, Role.ENCRYPTOR)
    assert not encryptor.has_bootstrapping
    del encryptor

    owner = ckks.Session.from_bytes(owner_keys, Role.OWNER)
    assert owner.has_bootstrapping
    result = owner.ciphertext_from_bytes(response)
    assert result.levels_left == 2
    check(owner, result, xs * xs, tolerance=2.0**-15)


@pytest.mark.slow
def test_bit_bootstrapping():
    session = ckks.Session(ckks.BootstrappingPreset.bit(2))
    n = session.slot_count
    bits = ((np.arange(n) // 3) % 2).astype(np.float64)
    x = session.encrypt(bits)
    b = (x * x).bootstrap()
    assert b.levels_left == 2
    check(session, b, bits, tolerance=2.0**-16)
    with pytest.raises(hg.StateError):
        session.AND_bootstrapping(x, x)


@pytest.mark.slow
def test_gate_bootstrapping():
    session = ckks.Session(ckks.BootstrappingPreset.gate(2))
    n = session.slot_count
    a = np.arange(n) % 2
    b = (np.arange(n) // 2) % 2
    x = session.encrypt(a)
    y = session.encrypt(b)
    cases = {
        "AND": (session.AND_bootstrapping, a & b),
        "OR": (session.OR_bootstrapping, a | b),
        "XOR": (session.XOR_bootstrapping, a ^ b),
        "NAND": (session.NAND_bootstrapping, 1 - (a & b)),
        "NOR": (session.NOR_bootstrapping, 1 - (a | b)),
        "XNOR": (session.XNOR_bootstrapping, 1 - (a ^ b)),
    }
    for name, (gate, expected) in cases.items():
        result = gate(x, y)
        assert result.levels_left == 2, name
        check(session, result, expected, tolerance=2.0**-16)
    with pytest.raises(hg.StateError):
        x.bootstrap()
    other = ckks.Session(multiplicative_depth=2)
    z = other.encrypt(np.zeros(other.slot_count))
    with pytest.raises(hg.ContextMismatchError):
        session.AND_bootstrapping(x, z)


def test_session_consistency_helpers():
    session = ckks.Session(ckks.Params(multiplicative_depth=2))
    assert copy.copy(session).slot_count == session.slot_count
    assert session.on_stream(None).stream == 0
    assert ckks.SessionCiphertext().is_empty
    assert not session.encrypt([1.0]).is_empty


def test_precision_stats_rejects_empty_values():
    with pytest.raises(ValueError):
        ckks.get_precision_stats([], [])
