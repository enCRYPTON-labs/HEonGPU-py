# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

import concurrent.futures
import io

import numpy as np
import pytest

import heongpu as hg
from heongpu import ckks

TOLERANCE = 1e-6


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup(ckks_setup):
    ckks_setup.operator = ckks.ArithmeticOperator(
        ckks_setup.context, ckks_setup.encoder
    )
    return ckks_setup


@pytest.fixture(scope="module")
def auto(ckks_auto_setup):
    ckks_auto_setup.operator = ckks.ArithmeticOperator(
        ckks_auto_setup.context, ckks_auto_setup.encoder
    )
    return ckks_auto_setup


@pytest.fixture(scope="module")
def logic(ckks_setup):
    return ckks.LogicOperator(
        ckks_setup.context, ckks_setup.encoder, ckks_setup.scale
    )


class SmallSetup:
    """A small manual-rescale context (N = 4096, no security level) for the
    encoding transforms."""

    def __init__(self):
        params = ckks.Params(
            security_level=hg.SecurityLevel.NONE,
            poly_modulus_degree=4096,
            coeff_modulus_bit_sizes=[50] + [40] * 8,
            special_modulus_bit_sizes=[50, 50],
        )
        self.scale = 2.0**40
        self.context = params.make_context()
        self.context.generate()
        self.keygen = ckks.KeyGenerator(self.context)
        self.secret_key = self.keygen.secret_key()
        public_key = self.keygen.public_key(self.secret_key)
        self.encoder = ckks.Encoder(self.context)
        self.encryptor = ckks.Encryptor(self.context, public_key)
        self.decryptor = ckks.Decryptor(self.context, self.secret_key)
        self.operator = ckks.ArithmeticOperator(self.context, self.encoder)


@pytest.fixture(scope="module")
def small():
    return SmallSetup()


def values(setup, rng, low=-1.0, high=1.0):
    return rng.uniform(low, high, setup.slot_count)


def complex_values(setup, rng):
    return values(setup, rng) + 1j * values(setup, rng)


# ----------------------------------------------------------------------
# Ciphertext arithmetic (manual rescale mode)
# ----------------------------------------------------------------------


def test_add_sub_negate(setup, rng):
    x, y = values(setup, rng), values(setup, rng)
    a, b = setup.encrypt(x), setup.encrypt(y)
    op = setup.operator
    np.testing.assert_allclose(setup.decrypt(op.add(a, b)), x + y, atol=1e-6)
    np.testing.assert_allclose(setup.decrypt(op.sub(a, b)), x - y, atol=1e-6)
    np.testing.assert_allclose(setup.decrypt(op.negate(a)), -x, atol=1e-6)
    # The inputs are not changed.
    np.testing.assert_allclose(setup.decrypt(a), x, atol=TOLERANCE)


def test_inplace_arithmetic(setup, rng):
    x, y = values(setup, rng), values(setup, rng)
    a, b = setup.encrypt(x), setup.encrypt(y)
    op = setup.operator
    assert op.add_inplace(a, b) is None
    np.testing.assert_allclose(setup.decrypt(a), x + y, atol=1e-6)
    op.sub_inplace(a, b)
    np.testing.assert_allclose(setup.decrypt(a), x, atol=1e-6)
    op.negate_inplace(a)
    np.testing.assert_allclose(setup.decrypt(a), -x, atol=1e-6)
    op.multiply_inplace(a, b)
    op.relinearize_inplace(a, setup.relin_key)
    op.rescale_inplace(a)
    np.testing.assert_allclose(setup.decrypt(a), -x * y, atol=1e-6)


def test_multiply_relinearize_rescale(setup, rng):
    x, y = values(setup, rng), values(setup, rng)
    a, b = setup.encrypt(x), setup.encrypt(y)
    op = setup.operator
    product = op.multiply(a, b)
    assert product.size == 3
    assert product.relinearization_required
    assert product.scale == pytest.approx(setup.scale**2)
    relinearized = op.relinearize(product, setup.relin_key)
    assert relinearized.size == 2
    assert product.size == 3
    rescaled = op.rescale(relinearized)
    assert rescaled.level == a.level - 1
    assert rescaled.depth == 1
    assert relinearized.level == a.level
    np.testing.assert_allclose(setup.decrypt(rescaled), x * y, atol=1e-6)


def test_multiplication_chain(setup, rng):
    x = values(setup, rng, 0.5, 1.0)
    cipher = setup.encrypt(x)
    op = setup.operator
    expected = x.copy()
    for _ in range(3):
        cipher = op.multiply(cipher, cipher)
        op.relinearize_inplace(cipher, setup.relin_key)
        op.rescale_inplace(cipher)
        expected = expected * expected
    assert cipher.level == setup.context.ciphertext_modulus_count - 4
    np.testing.assert_allclose(setup.decrypt(cipher), expected, atol=1e-5)


# ----------------------------------------------------------------------
# Plaintexts and constants
# ----------------------------------------------------------------------


def test_plaintext_operations(setup, rng):
    x, w = values(setup, rng), values(setup, rng)
    cipher = setup.encrypt(x)
    plain = setup.encoder.encode(w, setup.scale)
    op = setup.operator
    np.testing.assert_allclose(
        setup.decrypt(op.add_plain(cipher, plain)), x + w, atol=1e-6
    )
    np.testing.assert_allclose(
        setup.decrypt(op.sub_plain(cipher, plain)), x - w, atol=1e-6
    )
    product = op.multiply_plain(cipher, plain)
    assert product.rescale_required
    op.rescale_inplace(product)
    np.testing.assert_allclose(setup.decrypt(product), x * w, atol=1e-6)

    target = setup.encrypt(x)
    op.add_plain_inplace(target, plain)
    op.sub_plain_inplace(target, plain)
    op.multiply_plain_inplace(target, plain)
    op.rescale_inplace(target)
    np.testing.assert_allclose(setup.decrypt(target), x * w, atol=1e-6)


@pytest.mark.parametrize("constant", [0.75, -2, np.float64(1.5), np.int64(3)])
def test_real_constants(setup, rng, constant):
    x = values(setup, rng)
    cipher = setup.encrypt(x)
    op = setup.operator
    c = float(constant)
    np.testing.assert_allclose(
        setup.decrypt(op.add_plain(cipher, constant)), x + c, atol=1e-6
    )
    np.testing.assert_allclose(
        setup.decrypt(op.sub_plain(cipher, constant)), x - c, atol=1e-6
    )
    product = op.multiply_plain(cipher, constant)
    assert product.rescale_required
    assert product.scale == pytest.approx(cipher.scale**2)
    np.testing.assert_allclose(
        setup.decrypt(op.rescale(product)), x * c, atol=1e-6
    )


def test_real_constants_inplace(setup, rng):
    x = values(setup, rng)
    cipher = setup.encrypt(x)
    op = setup.operator
    op.add_plain_inplace(cipher, 0.5)
    op.sub_plain_inplace(cipher, 0.25)
    op.multiply_plain_inplace(cipher, 2.0, scale=2.0**30)
    assert cipher.scale == pytest.approx(setup.scale * 2.0**30)
    op.rescale_inplace(cipher)
    # The scale is about 2^30 after the rescaling.
    np.testing.assert_allclose(
        setup.decrypt(cipher), 2.0 * (x + 0.25), atol=1e-4
    )


def test_complex_constants(setup, rng):
    z = complex_values(setup, rng)
    cipher = setup.encryptor.encrypt(setup.encoder.encode(z, setup.scale))
    op = setup.operator
    c = 0.5 - 0.25j
    np.testing.assert_allclose(
        setup.decrypt_complex(op.add_plain(cipher, c)), z + c, atol=1e-6
    )
    np.testing.assert_allclose(
        setup.decrypt_complex(op.sub_plain(cipher, c)), z - c, atol=1e-6
    )
    # A Gaussian integer keeps the scale.
    product = op.multiply_plain(cipher, 2 + 1j)
    assert product.scale == cipher.scale
    np.testing.assert_allclose(
        setup.decrypt_complex(product), (2 + 1j) * z, atol=1e-6
    )
    # Another constant is encoded at the prime of the level.
    product = op.multiply_plain(cipher, c)
    assert product.scale > cipher.scale * 2.0**30
    np.testing.assert_allclose(setup.decrypt_complex(product), c * z, atol=1e-6)

    target = setup.encryptor.encrypt(setup.encoder.encode(z, setup.scale))
    op.add_plain_inplace(target, c)
    op.sub_plain_inplace(target, 2 * c)
    op.multiply_plain_inplace(target, 1j)
    np.testing.assert_allclose(
        setup.decrypt_complex(target), 1j * (z - c), atol=1e-6
    )


def test_constants_reject_arrays(setup, rng):
    cipher = setup.encrypt(values(setup, rng))
    with pytest.raises(TypeError):
        setup.operator.add_plain(cipher, np.ones(4))
    with pytest.raises(TypeError):
        setup.operator.multiply_plain(cipher, [1.0, 2.0])
    with pytest.raises(TypeError):
        setup.operator.multiply_plain(cipher, "2")


def test_imaginary_unit_and_scale_up(setup, rng):
    z = complex_values(setup, rng)
    cipher = setup.encryptor.encrypt(setup.encoder.encode(z, setup.scale))
    op = setup.operator
    np.testing.assert_allclose(
        setup.decrypt_complex(op.mult_i(cipher)), 1j * z, atol=1e-6
    )
    np.testing.assert_allclose(
        setup.decrypt_complex(op.div_i(cipher)), -1j * z, atol=1e-6
    )
    scaled = op.scale_up(cipher, 8.0)
    assert scaled.scale == pytest.approx(8.0 * cipher.scale)
    assert scaled.level == cipher.level
    np.testing.assert_allclose(setup.decrypt_complex(scaled), z, atol=1e-6)


# ----------------------------------------------------------------------
# Rotations, conjugation, Galois automorphisms and key switching
# ----------------------------------------------------------------------


@pytest.mark.parametrize("shift", [1, 3, 16, -1, -5])
def test_rotate(setup, rng, shift):
    x = values(setup, rng)
    cipher = setup.encrypt(x)
    rotated = setup.operator.rotate(cipher, setup.galois_key, shift)
    np.testing.assert_allclose(
        setup.decrypt(rotated), np.roll(x, -shift), atol=1e-6
    )
    setup.operator.rotate_inplace(cipher, setup.galois_key, shift)
    np.testing.assert_allclose(
        setup.decrypt(cipher), np.roll(x, -shift), atol=1e-6
    )


def test_rotate_with_loaded_galois_key(setup, rng):
    shifts = [3, -7, 100]
    key = setup.keygen.galois_key(setup.secret_key, shifts)
    loaded = ckks.GaloisKey.from_bytes(key.to_bytes())
    buffer = io.BytesIO()
    key.save(buffer)
    buffer.seek(0)
    read = ckks.GaloisKey.load(buffer)
    x = values(setup, rng)
    cipher = setup.encrypt(x)
    for galois_key in (key, loaded, read):
        for shift in shifts:
            rotated = setup.operator.rotate(cipher, galois_key, shift)
            np.testing.assert_allclose(
                setup.decrypt(rotated), np.roll(x, -shift), atol=1e-6
            )


def test_rotate_missing_key(setup, rng):
    key = setup.keygen.galois_key(setup.secret_key, [3])
    cipher = setup.encrypt(values(setup, rng))
    with pytest.raises(hg.KeyNotFoundError):
        setup.operator.rotate(cipher, key, 5)


def test_rotate_hoisted(setup, rng):
    x = values(setup, rng)
    cipher = setup.encrypt(x)
    shifts = [0, 1, 2, 8, -4]
    results = setup.operator.rotate_hoisted(
        cipher, setup.galois_key, np.array(shifts)
    )
    assert isinstance(results, list)
    assert len(results) == len(shifts)
    for shift, rotated in zip(shifts, results):
        np.testing.assert_allclose(
            setup.decrypt(rotated), np.roll(x, -shift), atol=1e-6
        )


def test_conjugate(setup, rng):
    z = complex_values(setup, rng)
    cipher = setup.encryptor.encrypt(setup.encoder.encode(z, setup.scale))
    result = setup.operator.conjugate(cipher, setup.galois_key)
    np.testing.assert_allclose(
        setup.decrypt_complex(result), np.conj(z), atol=1e-6
    )


def test_apply_galois(setup, rng):
    n = setup.context.poly_modulus_degree
    element = pow(5, 3, 2 * n)  # the rotation by 3
    key = ckks.GaloisKey(setup.context, galois_elements=[element])
    setup.keygen.generate_galois_key(key, setup.secret_key)
    x = values(setup, rng)
    cipher = setup.encrypt(x)
    result = setup.operator.apply_galois(cipher, key, element)
    np.testing.assert_allclose(setup.decrypt(result), np.roll(x, -3), atol=1e-6)
    setup.operator.apply_galois_inplace(cipher, key, element)
    np.testing.assert_allclose(setup.decrypt(cipher), np.roll(x, -3), atol=1e-6)


def test_keyswitch(setup, rng):
    new_secret_key = setup.keygen.secret_key()
    switch_key = ckks.SwitchKey(setup.context)
    setup.keygen.generate_switch_key(
        switch_key, new_secret_key, setup.secret_key
    )
    loaded = ckks.SwitchKey.from_bytes(switch_key.to_bytes())
    x = values(setup, rng)
    cipher = setup.encrypt(x)
    new_decryptor = ckks.Decryptor(setup.context, new_secret_key)
    for key in (switch_key, loaded):
        switched = setup.operator.keyswitch(cipher, key)
        result = setup.encoder.decode(new_decryptor.decrypt(switched))
        np.testing.assert_allclose(result, x, atol=1e-6)
    # The old secret key no longer decrypts it.
    assert np.abs(setup.decrypt(switched) - x).max() > 1.0


def test_sum_slots_and_inner_product(setup, rng):
    x, y = values(setup, rng), values(setup, rng)
    a, b = setup.encrypt(x), setup.encrypt(y)
    op = setup.operator
    total = op.sum_slots(a, setup.galois_key)
    np.testing.assert_allclose(setup.decrypt(total), x.sum(), atol=1e-4)
    inner = op.inner_product(a, b, setup.relin_key, setup.galois_key)
    np.testing.assert_allclose(setup.decrypt(inner), x @ y, atol=1e-4)
    plain = setup.encoder.encode(y, setup.scale)
    inner = op.inner_product(a, plain, setup.galois_key)
    np.testing.assert_allclose(setup.decrypt(inner), x @ y, atol=1e-4)


# ----------------------------------------------------------------------
# Polynomials
# ----------------------------------------------------------------------


def test_evaluate_polynomial(setup, rng):
    x = values(setup, rng)
    cipher = setup.encrypt(x)
    coefficients = [0.5, 0.25, 0.0, -0.02]
    result = setup.operator.evaluate_polynomial(
        cipher, coefficients, setup.relin_key
    )
    depth = ckks.ArithmeticOperator.polynomial_depth(3)
    assert depth == 2
    assert result.level == cipher.level - depth
    assert result.scale == pytest.approx(cipher.scale, rel=1e-9)
    expected = np.polynomial.polynomial.polyval(x, coefficients)
    np.testing.assert_allclose(setup.decrypt(result), expected, atol=1e-6)


def test_evaluate_chebyshev(setup, rng):
    x = values(setup, rng)
    cipher = setup.encrypt(x)
    coefficients = np.array([0.1, -0.4, 0.3, 0.2, -0.15, 0.05, 0.1, -0.02])
    result = setup.operator.evaluate_chebyshev(
        cipher, coefficients, -1.0, 1.0, setup.relin_key
    )
    assert result.level == cipher.level - 3
    expected = np.polynomial.chebyshev.chebval(x, coefficients)
    np.testing.assert_allclose(setup.decrypt(result), expected, atol=1e-6)


def test_evaluate_chebyshev_function(setup, rng):
    op = setup.operator
    for function, a, b in ((np.exp, -1.0, 1.0), (lambda v: 1.0 / v, 1.0, 4.0)):
        x = values(setup, rng, a, b)
        cipher = setup.encrypt(x)
        result = op.evaluate_chebyshev(
            cipher, function, a, b, 7, setup.relin_key
        )
        depth = ckks.ArithmeticOperator.polynomial_depth(7, True, a, b)
        assert result.level == cipher.level - depth
        # The interpolant, evaluated on the CPU.
        coefficients = ckks.chebyshev_coefficients(function, a, b, 7)
        expected = np.polynomial.chebyshev.chebval(
            (2.0 * x - a - b) / (b - a), coefficients
        )
        np.testing.assert_allclose(setup.decrypt(result), expected, atol=1e-6)
        np.testing.assert_allclose(
            setup.decrypt(result), function(x), atol=1e-3
        )


def test_chebyshev_helpers():
    coefficients = ckks.chebyshev_coefficients(np.exp, -1.0, 1.0, 7)
    assert coefficients.dtype == np.float64
    assert coefficients.shape == (8,)
    nodes = ckks.chebyshev_nodes(8, -1.0, 1.0)
    assert nodes.shape == (8,)
    assert np.all(np.abs(nodes) < 1.0)
    np.testing.assert_allclose(
        np.polynomial.chebyshev.chebval(nodes, coefficients),
        np.exp(nodes),
        atol=1e-9,
    )


def test_evaluate_chebyshev_function_errors(setup, rng):
    cipher = setup.encrypt(values(setup, rng))

    def failing(v):
        raise ZeroDivisionError("from the callable")

    with pytest.raises(ZeroDivisionError, match="from the callable"):
        setup.operator.evaluate_chebyshev(
            cipher, failing, -1.0, 1.0, 7, setup.relin_key
        )
    with pytest.raises(TypeError):
        setup.operator.evaluate_chebyshev(
            cipher, lambda v: "text", -1.0, 1.0, 7, setup.relin_key
        )
    with pytest.raises(hg.ParameterError):
        setup.operator.evaluate_chebyshev(
            cipher, np.sin, 1.0, 1.0, 7, setup.relin_key
        )


def test_polynomial_errors(setup, rng):
    cipher = setup.encrypt(values(setup, rng))
    with pytest.raises(hg.ParameterError):
        setup.operator.evaluate_polynomial(cipher, [1.0], setup.relin_key)
    # Degree 64 needs more levels than the ciphertext has.
    assert ckks.ArithmeticOperator.polynomial_depth(64) > cipher.level
    with pytest.raises(hg.LevelMismatchError):
        setup.operator.evaluate_polynomial(
            cipher, np.full(65, 0.01), setup.relin_key
        )


def test_chebyshev_callable_in_threads(setup, rng):
    # The callable is called from C++ with the GIL released.
    x = values(setup, rng)
    ciphers = [setup.encrypt(x) for _ in range(3)]

    def evaluate(cipher):
        result = setup.operator.evaluate_chebyshev(
            cipher, np.sin, -1.0, 1.0, 7, setup.relin_key
        )
        return setup.decrypt(result)

    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        results = list(pool.map(evaluate, ciphers))
    for result in results:
        np.testing.assert_allclose(result, np.sin(x), atol=1e-5)


# ----------------------------------------------------------------------
# Levels
# ----------------------------------------------------------------------


def test_mod_drop(setup, rng):
    x, w = values(setup, rng), values(setup, rng)
    cipher = setup.encrypt(x)
    op = setup.operator
    dropped = op.mod_drop(cipher)
    assert dropped.level == cipher.level - 1
    assert dropped.scale == cipher.scale
    np.testing.assert_allclose(setup.decrypt(dropped), x, atol=1e-6)
    op.mod_drop_inplace(cipher)
    assert cipher.level == dropped.level

    plain = setup.encoder.encode(w, setup.scale)
    dropped_plain = op.mod_drop(plain)
    assert dropped_plain.depth == 1
    assert plain.depth == 0
    op.mod_drop_inplace(plain)
    assert plain.depth == 1
    np.testing.assert_allclose(
        setup.decrypt(op.add_plain(cipher, dropped_plain)), x + w, atol=1e-6
    )


def test_level_mismatch(setup, rng):
    a = setup.encrypt(values(setup, rng))
    b = setup.operator.mod_drop(setup.encrypt(values(setup, rng)))
    with pytest.raises(hg.LevelMismatchError, match="mod_drop"):
        setup.operator.add(a, b)
    with pytest.raises(hg.LevelMismatchError):
        setup.operator.multiply(a, b)
    with pytest.raises(ValueError):
        setup.operator.sub(a, b)
    plain = setup.encoder.encode(values(setup, rng), setup.scale)
    with pytest.raises(hg.LevelMismatchError):
        setup.operator.add_plain(b, plain)


def test_context_mismatch(setup, small, rng):
    cipher = setup.encrypt(values(setup, rng))
    with pytest.raises(hg.ContextMismatchError):
        small.operator.add(cipher, cipher)


def test_unrelinearized_product_errors(setup, rng):
    cipher = setup.encrypt(values(setup, rng))
    product = setup.operator.multiply(cipher, cipher)
    with pytest.raises(ValueError):
        setup.operator.add_plain(product, 1.0)


# ----------------------------------------------------------------------
# Automatic rescale mode
# ----------------------------------------------------------------------


def test_automatic_products(auto, rng):
    x, y = values(auto, rng), values(auto, rng)
    a, b = auto.encrypt(x), auto.encrypt(y)
    op = auto.operator
    top = auto.context.top_level
    assert a.level == top
    square = op.multiply(a, a)
    op.relinearize_inplace(square, auto.relin_key)
    assert square.rescale_required
    # The square is rescaled and b brought down to its level by itself.
    product = op.multiply(square, b)
    op.relinearize_inplace(product, auto.relin_key)
    assert product.level == top - 1
    total = op.add(product, a)
    assert total.level == top - 2
    assert total.scale == auto.context.get_level_scale(top - 2)
    np.testing.assert_allclose(auto.decrypt(total), x * x * y + x, atol=1e-5)


def test_automatic_constants_and_rotation(auto, rng):
    x = values(auto, rng)
    a = auto.encrypt(x)
    op = auto.operator
    # No scale: the constant gets the scale of the level.
    halved = op.multiply_plain(a, 0.5)
    rotated = op.rotate(halved, auto.galois_key, 2)
    assert not rotated.rescale_required
    assert rotated.level == a.level - 1
    result = op.add_plain(rotated, 0.25)
    np.testing.assert_allclose(
        auto.decrypt(result), 0.5 * np.roll(x, -2) + 0.25, atol=1e-6
    )
    plain = auto.encoder.encode(x, depth=1)
    np.testing.assert_allclose(
        auto.decrypt(op.multiply_plain(rotated, plain)),
        0.5 * np.roll(x, -2) * x,
        atol=1e-6,
    )


def test_automatic_mod_drop_and_polynomial(auto, rng):
    x = values(auto, rng)
    a = auto.encrypt(x)
    op = auto.operator
    dropped = op.mod_drop(a)
    assert dropped.scale == auto.context.get_level_scale(a.level - 1)
    np.testing.assert_allclose(auto.decrypt(op.sub(dropped, a)), 0.0, atol=1e-6)
    result = op.evaluate_polynomial(a, [1.0, 1.0, 0.5], auto.relin_key)
    assert result.scale == auto.context.get_level_scale(result.level)
    np.testing.assert_allclose(
        auto.decrypt(result), 1.0 + x + 0.5 * x * x, atol=1e-5
    )


def test_automatic_scale_mismatch(auto, rng):
    a = auto.encrypt(values(auto, rng))
    other = auto.encryptor.encrypt(
        auto.encoder.encode(values(auto, rng), 1.001 * 2.0**40)
    )
    with pytest.raises(hg.ParameterError):
        auto.operator.add(a, other)
    unrelinearized = auto.operator.multiply(a, a)
    with pytest.raises(hg.StateError):
        auto.operator.multiply(unrelinearized, a)


# ----------------------------------------------------------------------
# Streams
# ----------------------------------------------------------------------


def test_operations_on_streams(setup, rng):
    x, y = values(setup, rng), values(setup, rng)
    a, b = setup.encrypt(x), setup.encrypt(y)
    op = setup.operator
    hg.synchronize()
    with hg.Stream() as stream:
        total = op.add(a, b, stream)
        product = op.multiply(a, b, options=hg.ExecutionOptions(stream=stream))
        op.relinearize_inplace(product, setup.relin_key, options=stream)
        op.rescale_inplace(product, options=stream)
        rotated = op.rotate(total, setup.galois_key, 1, options=stream)
        hoisted = op.rotate_hoisted(a, setup.galois_key, [2], options=stream)
        stream.synchronize()
    np.testing.assert_allclose(setup.decrypt(total), x + y, atol=1e-6)
    np.testing.assert_allclose(setup.decrypt(product), x * y, atol=1e-6)
    np.testing.assert_allclose(
        setup.decrypt(rotated), np.roll(x + y, -1), atol=1e-6
    )
    np.testing.assert_allclose(
        setup.decrypt(hoisted[0]), np.roll(x, -2), atol=1e-6
    )


def evaluate_polynomial_on(setup, stream, x):
    a = setup.encrypt(x)
    hg.synchronize()
    result = setup.operator.evaluate_polynomial(
        a, [0.5, 1.0, 1.0], setup.relin_key, options=stream
    )
    stream.synchronize()
    return setup.decrypt(result)


def test_polynomial_on_blocking_stream(setup, rng):
    x = values(setup, rng)
    stream = hg.Stream(non_blocking=False)
    result = evaluate_polynomial_on(setup, stream, x)
    np.testing.assert_allclose(result, 0.5 + x + x * x, atol=1e-6)


def test_polynomial_on_non_blocking_stream(setup, rng):
    x = values(setup, rng)
    stream = hg.Stream(non_blocking=True)
    result = evaluate_polynomial_on(setup, stream, x)
    np.testing.assert_allclose(result, 0.5 + x + x * x, atol=1e-6)


def test_copying_operations_on_stream(setup, rng):
    # relinearize and rescale copy their input on its stream first.
    x, y = values(setup, rng), values(setup, rng)
    a, b = setup.encrypt(x), setup.encrypt(y)
    op = setup.operator
    product = op.multiply(a, b)
    hg.synchronize()
    with hg.Stream() as stream:
        relinearized = op.relinearize(product, setup.relin_key, stream)
        rescaled = op.rescale(relinearized, options=stream)
        stream.synchronize()
        assert rescaled.stream == stream.handle
    np.testing.assert_allclose(setup.decrypt(rescaled), x * y, atol=1e-6)


def test_options_type_error(setup, rng):
    a = setup.encrypt(values(setup, rng))
    with pytest.raises(TypeError):
        setup.operator.add(a, a, options="stream")


# ----------------------------------------------------------------------
# Encoding transforms
# ----------------------------------------------------------------------


def test_encoding_transform_config():
    config = ckks.EncodingTransformConfig()
    assert (config.cts_piece, config.stc_piece) == (3, 3)
    assert (config.cts_start_level, config.stc_start_level) == (-1, -1)
    config = ckks.EncodingTransformConfig(2, 4, 0, 3)
    config.stc_piece = 3
    assert "stc_piece=3" in repr(config)
    context = ckks.EncodingTransformContext()
    assert not context.generated
    assert context.key_indices == []


def test_coeff_to_slot_round_trip(small):
    op = small.operator
    transform = op.generate_encoding_transform_context(
        ckks.EncodingTransformConfig(3, 3, 0, 3)
    )
    assert transform.generated
    assert (transform.cts_piece, transform.stc_piece) == (3, 3)
    assert (transform.cts_level, transform.stc_level) == (0, 3)
    galois_key = small.keygen.galois_key(
        small.secret_key, transform.key_indices
    )

    n = small.context.poly_modulus_degree
    message = np.zeros(n)
    message[:8] = [1.0, -0.5, 0.25, -1.75, 2.5, -3.25, 0.125, -0.875]
    plain = small.encoder.encode(
        message, small.scale, encoding=hg.Encoding.COEFFICIENT
    )
    cipher = small.encryptor.encrypt(plain)

    halves = op.coeff_to_slot(cipher, galois_key, transform)
    assert len(halves) == 2
    assert all(half.encoding == hg.Encoding.SLOT for half in halves)

    back = op.slot_to_coeff(halves[0], halves[1], galois_key, transform)
    assert back.encoding == hg.Encoding.COEFFICIENT
    result = small.encoder.decode(small.decryptor.decrypt(back))
    np.testing.assert_allclose(result, message, atol=5e-2)


def test_encoding_transform_errors(small, auto, rng):
    with pytest.raises(IndexError):
        small.operator.generate_encoding_transform_context(
            ckks.EncodingTransformConfig(cts_piece=6)
        )
    with pytest.raises(hg.StateError):
        auto.operator.generate_encoding_transform_context()
    galois_key = small.keygen.galois_key(small.secret_key)
    cipher = small.encryptor.encrypt(
        small.encoder.encode(np.zeros(4), small.scale)
    )
    with pytest.raises(ValueError):
        small.operator.coeff_to_slot(
            cipher, galois_key, ckks.EncodingTransformContext()
        )


def test_bootstrapping_key_indices_before_params(setup):
    with pytest.raises(ValueError):
        setup.operator.bootstrapping_key_indices()


# ----------------------------------------------------------------------
# Logic gates without bootstrapping
# ----------------------------------------------------------------------


GATES = {
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "XOR": lambda a, b: a ^ b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR": lambda a, b: 1 - (a | b),
    "XNOR": lambda a, b: 1 - (a ^ b),
}


def bits(setup, rng):
    return rng.integers(0, 2, setup.slot_count)


@pytest.mark.parametrize("gate", sorted(GATES))
def test_logic_gates_ciphertexts(setup, logic, rng, gate):
    a, b = bits(setup, rng), bits(setup, rng)
    x, y = setup.encrypt(a.astype(float)), setup.encrypt(b.astype(float))
    expected = GATES[gate](a, b)
    result = getattr(logic, gate)(x, y, setup.relin_key)
    np.testing.assert_allclose(setup.decrypt(result), expected, atol=1e-5)
    target = setup.encrypt(a.astype(float))
    assert getattr(logic, gate + "_inplace")(target, y, setup.relin_key) is None
    np.testing.assert_allclose(setup.decrypt(target), expected, atol=1e-5)


@pytest.mark.parametrize("gate", sorted(GATES))
def test_logic_gates_plaintexts(setup, logic, rng, gate):
    a, b = bits(setup, rng), bits(setup, rng)
    x = setup.encrypt(a.astype(float))
    plain = setup.encoder.encode(b.astype(float), setup.scale)
    expected = GATES[gate](a, b)
    result = getattr(logic, gate)(x, plain)
    np.testing.assert_allclose(setup.decrypt(result), expected, atol=1e-5)
    getattr(logic, gate + "_inplace")(x, plain)
    np.testing.assert_allclose(setup.decrypt(x), expected, atol=1e-5)


def test_logic_not_and_shared_operations(setup, logic, rng):
    a = bits(setup, rng)
    x = setup.encrypt(a.astype(float))
    np.testing.assert_allclose(setup.decrypt(logic.NOT(x)), 1 - a, atol=1e-5)
    logic.NOT_inplace(x)
    np.testing.assert_allclose(setup.decrypt(x), 1 - a, atol=1e-5)
    rotated = logic.rotate(x, setup.galois_key, 2)
    np.testing.assert_allclose(
        setup.decrypt(rotated), np.roll(1 - a, -2), atol=1e-5
    )
    dropped = logic.mod_drop(x)
    assert dropped.level == x.level - 1
    logic.mod_drop_inplace(x)
    assert x.level == dropped.level
    with pytest.raises(ValueError):
        logic.bootstrapping_key_indices()


def test_logic_gates_need_manual_mode(auto, rng):
    logic = ckks.LogicOperator(auto.context, auto.encoder, 2.0**40)
    x = auto.encrypt(bits(auto, rng).astype(float))
    with pytest.raises(hg.StateError):
        logic.NOT(x)
    with pytest.raises(hg.StateError):
        logic.AND(x, x, auto.relin_key)


# ----------------------------------------------------------------------
# Precision statistics
# ----------------------------------------------------------------------


def test_precision_stats(rng):
    want = rng.uniform(-1, 1, 1024)
    test = want + rng.uniform(-1e-6, 1e-6, 1024)
    stats = ckks.get_precision_stats(want, test)
    assert isinstance(stats, ckks.PrecisionStats)
    assert 19.0 < stats.min_precision.real <= stats.mean_precision.real
    assert stats.max_delta.real <= 1e-6
    assert stats.max_delta.imag == 0.0
    assert len(stats.real_dist) > 0
    assert isinstance(stats.real_dist[0], ckks.DistEntry)
    assert stats.cdf_resol == 500
    assert "Stats" in repr(stats.mean_delta)
    assert str(stats)
    assert "PrecisionStats" in repr(stats)

    complex_want = want + 1j * want[::-1]
    complex_test = test + 1j * test[::-1]
    stats = ckks.get_precision_stats(complex_want, complex_test)
    assert stats.min_precision.imag > 19.0
    stats = ckks.get_precision_stats(list(want), complex_test)
    assert stats.min_precision.real > 19.0
    stats = ckks.get_precision_stats(complex_want, test)
    assert stats.max_delta.imag > 0.5

    with pytest.raises(ValueError):
        ckks.get_precision_stats(want, test[:-1])
