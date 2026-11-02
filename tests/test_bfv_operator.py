# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

import numpy as np
import pytest

import heongpu as hg
from heongpu import bfv

# A prime with t = 1 (mod 2N), which allows batching.
PLAIN_MODULUS = 786433


class BFVOperatorSetup:
    """A BFV context with its keys, operators and helpers. The values are
    kept as NumPy object arrays of Python ints, so that the expected results
    are computed exactly."""

    def __init__(self, depth=3, poly_modulus_degree=8192):
        params = bfv.Params(
            multiplicative_depth=depth,
            plain_modulus=PLAIN_MODULUS,
            poly_modulus_degree=poly_modulus_degree,
        )
        self.context = params.make_context()
        self.context.generate()
        self.t = self.context.plain_modulus
        self.n = self.context.poly_modulus_degree
        self.row_size = self.n // 2
        keygen = bfv.KeyGenerator(self.context)
        self.keygen = keygen
        self.secret_key = keygen.secret_key()
        self.public_key = keygen.public_key(self.secret_key)
        self.relin_key = keygen.relin_key(self.secret_key)
        self.galois_key = keygen.galois_key(self.secret_key)
        # The steps +-1, ..., +-2^(log2(N) - 2), which sum_slots needs.
        self.full_galois_key = bfv.GaloisKey(
            self.context, max_shift=self.n.bit_length() - 3
        )
        keygen.generate_galois_key(self.full_galois_key, self.secret_key)
        self.encoder = bfv.Encoder(self.context)
        self.encryptor = bfv.Encryptor(self.context, self.public_key)
        self.decryptor = bfv.Decryptor(self.context, self.secret_key)
        self.operator = bfv.ArithmeticOperator(self.context, self.encoder)
        self.logic = bfv.LogicOperator(self.context, self.encoder)

    def random(self, rng):
        return rng.integers(0, self.t, self.n).astype(object)

    def bits(self, rng):
        return rng.integers(0, 2, self.n).astype(object)

    def encode(self, values):
        return self.encoder.encode(values)

    def encrypt(self, values):
        return self.encryptor.encrypt(self.encoder.encode(values))

    def decrypt(self, cipher):
        return self.encoder.decode(self.decryptor.decrypt(cipher)).astype(
            object
        )

    def check(self, cipher, expected):
        np.testing.assert_array_equal(
            self.decrypt(cipher), np.mod(expected, self.t)
        )

    def rows(self, values):
        return np.asarray(values, dtype=object).reshape(2, self.row_size)

    def rotated(self, values, shift):
        """Slot i of each row gets slot i + shift."""
        return np.roll(self.rows(values), -shift, axis=1).reshape(-1)


@pytest.fixture(scope="module")
def setup():
    return BFVOperatorSetup()


@pytest.fixture(scope="module")
def other_context():
    context = bfv.Params(
        poly_modulus_degree=8192,
        plain_modulus=PLAIN_MODULUS,
        coeff_modulus_bit_sizes=[50, 50],
        special_modulus_bit_sizes=[51],
    ).make_context()
    context.generate()
    return context


# ----------------------------------------------------------------------
# Addition, subtraction, negation
# ----------------------------------------------------------------------


def test_add_sub_negate(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    ca, cb = setup.encrypt(a), setup.encrypt(b)
    op = setup.operator
    setup.check(op.add(ca, cb), a + b)
    setup.check(op.sub(ca, cb), a - b)
    setup.check(op.negate(ca), -a)
    # The inputs are unchanged.
    setup.check(ca, a)
    setup.check(cb, b)


def test_inplace_add_sub_negate(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    op = setup.operator
    cipher = setup.encrypt(a)
    assert op.add_inplace(cipher, setup.encrypt(b)) is None
    setup.check(cipher, a + b)
    op.sub_inplace(cipher, setup.encrypt(b))
    setup.check(cipher, a)
    op.negate_inplace(cipher)
    setup.check(cipher, -a)
    op.add_inplace(cipher, cipher)
    setup.check(cipher, -2 * a)


def test_plain_add_sub(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    ca, pb = setup.encrypt(a), setup.encode(b)
    op = setup.operator
    setup.check(op.add_plain(ca, pb), a + b)
    setup.check(op.sub_plain(ca, pb), a - b)
    op.add_plain_inplace(ca, pb)
    op.add_plain_inplace(ca, pb)
    setup.check(ca, a + 2 * b)
    op.sub_plain_inplace(ca, pb)
    setup.check(ca, a + b)


def test_signed_arithmetic(setup, rng):
    t = setup.t
    a = rng.integers(-500, 500, setup.n)
    b = rng.integers(-500, 500, setup.n)
    assert np.all(np.abs(a * b - a) < t // 2)
    op = setup.operator
    product = op.relinearize(
        op.multiply(setup.encrypt(a), setup.encrypt(b)), setup.relin_key
    )
    result = setup.encoder.decode(
        setup.decryptor.decrypt(op.sub(product, setup.encrypt(a))), signed=True
    )
    np.testing.assert_array_equal(result, a * b - a)


def test_plain_modulus_wrap(setup):
    t = setup.t
    op = setup.operator
    cipher = op.add(setup.encrypt([t - 1, t - 2, 1]), setup.encrypt([2, 5, -3]))
    np.testing.assert_array_equal(setup.decrypt(cipher)[:3], [1, 3, t - 2])


# ----------------------------------------------------------------------
# Multiplication and relinearization
# ----------------------------------------------------------------------


def test_multiply_relinearize(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    op = setup.operator
    product = op.multiply(setup.encrypt(a), setup.encrypt(b))
    assert product.size == 3
    assert product.relinearization_required
    # A product must be relinearized before it is decrypted.
    with pytest.raises(hg.StateError):
        setup.decryptor.decrypt(product)
    with pytest.raises(hg.StateError):
        setup.decryptor.remainder_noise_budget(product)
    relinearized = op.relinearize(product, setup.relin_key)
    assert not relinearized.relinearization_required
    assert product.relinearization_required
    setup.check(relinearized, a * b)
    op.relinearize_inplace(product, setup.relin_key)
    assert not product.relinearization_required
    setup.check(product, a * b)


def test_multiply_inplace_chain(setup, rng):
    a, b, c = setup.random(rng), setup.random(rng), setup.random(rng)
    op = setup.operator
    decryptor = setup.decryptor
    cipher = setup.encrypt(a)
    budget = decryptor.remainder_noise_budget(cipher)
    for factor in (b, c, a):
        op.multiply_inplace(cipher, setup.encrypt(factor))
        op.relinearize_inplace(cipher, setup.relin_key)
        new_budget = decryptor.remainder_noise_budget(cipher)
        assert 0 < new_budget < budget
        budget = new_budget
    setup.check(cipher, a * b * c * a)


def test_multiply_plain(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    op = setup.operator
    ca, pb = setup.encrypt(a), setup.encode(b)
    setup.check(op.multiply_plain(ca, pb), a * b)
    op.multiply_plain_inplace(ca, pb)
    setup.check(ca, a * b)
    # A scalar plaintext multiplies every slot.
    setup.check(op.multiply_plain(setup.encrypt(a), setup.encode(-3)), -3 * a)


def test_square(setup, rng):
    a = setup.random(rng)
    op = setup.operator
    cipher = setup.encrypt(a)
    op.multiply_inplace(cipher, cipher)
    op.relinearize_inplace(cipher, setup.relin_key)
    setup.check(cipher, a * a)


def test_relinearization_errors(setup, rng):
    op = setup.operator
    cipher = setup.encrypt(setup.random(rng))
    with pytest.raises(ValueError):
        op.relinearize_inplace(cipher, setup.relin_key)
    product = op.multiply(cipher, cipher)
    with pytest.raises(ValueError):
        op.multiply(product, cipher)
    with pytest.raises(ValueError):
        op.multiply_plain(product, setup.encode([1]))
    with pytest.raises(hg.StateError):
        op.rotate_rows(product, setup.galois_key, 1)


# ----------------------------------------------------------------------
# Rotations, Galois automorphisms and key switching
# ----------------------------------------------------------------------


@pytest.mark.parametrize("shift", [1, 3, 7, -1, -6, 100])
def test_rotate_rows(setup, rng, shift):
    a = setup.random(rng)
    cipher = setup.encrypt(a)
    setup.check(
        setup.operator.rotate_rows(cipher, setup.galois_key, shift),
        setup.rotated(a, shift),
    )
    setup.operator.rotate_rows_inplace(cipher, setup.galois_key, shift)
    setup.check(cipher, setup.rotated(a, shift))


def test_rotate_rows_trivial(setup, rng):
    a = setup.random(rng)
    cipher = setup.encrypt(a)
    setup.check(setup.operator.rotate_rows(cipher, setup.galois_key, 0), a)
    setup.check(
        setup.operator.rotate_rows(cipher, setup.galois_key, setup.row_size), a
    )


def test_rotate_rows_hoisted(setup, rng):
    a = setup.random(rng)
    shifts = [1, 2, 5, -4, 0]
    results = setup.operator.rotate_rows_hoisted(
        setup.encrypt(a), setup.galois_key, np.array(shifts)
    )
    assert isinstance(results, list) and len(results) == len(shifts)
    for shift, cipher in zip(shifts, results):
        setup.check(cipher, setup.rotated(a, shift))


def test_rotate_columns(setup, rng):
    a = setup.random(rng)
    result = setup.operator.rotate_columns(setup.encrypt(a), setup.galois_key)
    setup.check(result, setup.rows(a)[::-1].reshape(-1))


def test_rotation_with_custom_keys(setup, rng):
    key = setup.keygen.galois_key(setup.secret_key, [5])
    a = setup.random(rng)
    setup.check(
        setup.operator.rotate_rows(setup.encrypt(a), key, 5),
        setup.rotated(a, 5),
    )
    with pytest.raises(hg.KeyNotFoundError):
        setup.operator.rotate_rows(setup.encrypt(a), key, 3)


def test_apply_galois(setup, rng):
    a = setup.random(rng)
    element = pow(3, 4, 2 * setup.n)
    op = setup.operator
    cipher = setup.encrypt(a)
    setup.check(
        op.apply_galois(cipher, setup.galois_key, element),
        setup.rotated(a, 4),
    )
    op.apply_galois_inplace(cipher, setup.galois_key, element)
    setup.check(cipher, setup.rotated(a, 4))


def test_apply_galois_custom_element(setup, rng):
    a = setup.random(rng)
    element = pow(3, 6, 2 * setup.n)
    key = bfv.GaloisKey(setup.context, galois_elements=[element])
    setup.keygen.generate_galois_key(key, setup.secret_key)
    setup.check(
        setup.operator.apply_galois(setup.encrypt(a), key, element),
        setup.rotated(a, 6),
    )


def test_keyswitch(setup, rng):
    new_key = setup.keygen.secret_key()
    switch_key = bfv.SwitchKey(setup.context)
    setup.keygen.generate_switch_key(switch_key, new_key, setup.secret_key)
    a = setup.random(rng)
    switched = setup.operator.keyswitch(setup.encrypt(a), switch_key)
    decryptor = bfv.Decryptor(setup.context, new_key)
    np.testing.assert_array_equal(
        setup.encoder.decode(decryptor.decrypt(switched)).astype(object), a
    )


# ----------------------------------------------------------------------
# Slot sums, powers of X and the NTT domain
# ----------------------------------------------------------------------


def test_sum_slots(setup, rng):
    a = setup.random(rng)
    result = setup.operator.sum_slots(setup.encrypt(a), setup.full_galois_key)
    setup.check(result, np.full(setup.n, sum(a), dtype=object))


def test_inner_product(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    expected = np.full(setup.n, int(np.dot(a, b)), dtype=object)
    op = setup.operator
    result = op.inner_product(
        setup.encrypt(a),
        setup.encrypt(b),
        setup.relin_key,
        setup.full_galois_key,
    )
    setup.check(result, expected)
    result = op.inner_product(
        setup.encrypt(a), setup.encode(b), setup.full_galois_key
    )
    setup.check(result, expected)


@pytest.mark.parametrize("index", [0, 1, 5, 1000])
def test_multiply_power_of_x(setup, rng, index):
    cipher = setup.encrypt(setup.random(rng))
    before = setup.decryptor.decrypt(cipher).get_data().astype(object)
    after = setup.operator.multiply_power_of_X(cipher, index)
    coefficients = setup.decryptor.decrypt(after).get_data().astype(object)
    # X^index * m(X) modulo X^N + 1: the coefficients that wrap around change
    # sign.
    expected = np.concatenate(
        [-before[setup.n - index :], before[: setup.n - index]]
    )
    np.testing.assert_array_equal(coefficients, np.mod(expected, setup.t))


def test_ntt_domain(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    op = setup.operator
    plain = setup.encode(b)
    plain_ntt = op.transform_to_ntt(plain)
    assert plain_ntt.in_ntt_domain
    assert not plain.in_ntt_domain
    assert plain_ntt.size == setup.n * setup.context.ciphertext_modulus_count
    cipher = setup.encrypt(a)
    cipher_ntt = op.transform_to_ntt(cipher)
    assert cipher_ntt.in_ntt_domain and not cipher.in_ntt_domain
    product = op.multiply_plain(cipher_ntt, plain_ntt)
    assert product.in_ntt_domain
    back = op.transform_from_ntt(product)
    assert not back.in_ntt_domain
    setup.check(back, a * b)
    op.transform_to_ntt_inplace(cipher)
    op.transform_to_ntt_inplace(plain)
    assert cipher.in_ntt_domain and plain.in_ntt_domain
    op.multiply_plain_inplace(cipher, plain)
    op.transform_from_ntt_inplace(cipher)
    setup.check(cipher, a * b)


def test_ntt_domain_mismatch(setup, rng):
    op = setup.operator
    cipher_ntt = op.transform_to_ntt(setup.encrypt(setup.random(rng)))
    with pytest.raises(RuntimeError):
        op.multiply_plain(cipher_ntt, setup.encode([1]))
    with pytest.raises(RuntimeError):
        op.add_plain(cipher_ntt, setup.encode([1]))


# ----------------------------------------------------------------------
# Errors, options and streams
# ----------------------------------------------------------------------


def test_context_mismatch(setup, other_context, rng):
    other_cipher = bfv.Encryptor(
        other_context,
        bfv.KeyGenerator(other_context).public_key(
            bfv.KeyGenerator(other_context).secret_key()
        ),
    ).encrypt(bfv.Encoder(other_context).encode([1]))
    cipher = setup.encrypt([1])
    op = setup.operator
    with pytest.raises(hg.ContextMismatchError):
        op.add(cipher, other_cipher)
    with pytest.raises(hg.ContextMismatchError):
        op.add_plain(cipher, bfv.Encoder(other_context).encode([1]))
    other_key = bfv.KeyGenerator(other_context).relin_key(
        bfv.KeyGenerator(other_context).secret_key()
    )
    with pytest.raises(hg.ContextMismatchError):
        op.relinearize(op.multiply(cipher, cipher), other_key)
    with pytest.raises(hg.ContextMismatchError):
        setup.logic.AND(cipher, other_cipher, setup.relin_key)


def test_operator_type_errors(setup):
    with pytest.raises(TypeError):
        setup.operator.add(setup.encrypt([1]), [1, 2])
    with pytest.raises(TypeError):
        setup.operator.rotate_rows_hoisted(
            setup.encrypt([1]), setup.galois_key, ["a"]
        )


def test_host_output_option(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    host = hg.ExecutionOptions(storage=hg.StorageType.HOST)
    result = setup.operator.add(
        setup.encrypt(a), setup.encrypt(b), options=host
    )
    assert not result.is_on_device
    setup.check(result, a + b)


def test_host_inputs(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    ca, cb = setup.encrypt(a), setup.encrypt(b)
    ca.store_in_host()
    cb.store_in_host()
    result = setup.operator.multiply(ca, cb)
    assert not ca.is_on_device and not cb.is_on_device
    assert result.is_on_device
    setup.check(setup.operator.relinearize(result, setup.relin_key), a * b)


def test_stream_execution(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    op = setup.operator
    with hg.Stream() as stream:
        options = hg.ExecutionOptions(stream=stream)
        ca = setup.encryptor.encrypt(
            setup.encoder.encode(a, options=stream), stream
        )
        cb = setup.encryptor.encrypt(
            setup.encoder.encode(b, options=stream), stream
        )
        product = op.multiply(ca, cb, options=stream)
        op.relinearize_inplace(product, setup.relin_key, options=options)
        rotated = op.rotate_rows(product, setup.galois_key, 2, options=stream)
        assert rotated.stream == stream.handle
        anded = setup.logic.XOR(ca, cb, setup.relin_key, options=stream)
        stream.synchronize()
    setup.check(rotated, setup.rotated(a * b, 2))
    assert not anded.relinearization_required


# ----------------------------------------------------------------------
# Logic gates
# ----------------------------------------------------------------------

GATES = {
    "AND": lambda x, y: x & y,
    "OR": lambda x, y: x | y,
    "XOR": lambda x, y: x ^ y,
    "NAND": lambda x, y: 1 - (x & y),
    "NOR": lambda x, y: 1 - (x | y),
    "XNOR": lambda x, y: 1 - (x ^ y),
}


@pytest.mark.parametrize("gate", sorted(GATES))
def test_logic_gate_ciphertexts(setup, rng, gate):
    x, y = setup.bits(rng), setup.bits(rng)
    cx, cy = setup.encrypt(x), setup.encrypt(y)
    expected = GATES[gate](x, y)
    method = getattr(setup.logic, gate)
    result = method(cx, cy, setup.relin_key)
    assert not result.relinearization_required
    setup.check(result, expected)
    setup.check(cx, x)
    getattr(setup.logic, gate + "_inplace")(cx, cy, setup.relin_key)
    setup.check(cx, expected)


@pytest.mark.parametrize("gate", sorted(GATES))
def test_logic_gate_plaintext(setup, rng, gate):
    x, y = setup.bits(rng), setup.bits(rng)
    cx, py = setup.encrypt(x), setup.encode(y)
    expected = GATES[gate](x, y)
    setup.check(getattr(setup.logic, gate)(cx, py), expected)
    getattr(setup.logic, gate + "_inplace")(cx, py)
    setup.check(cx, expected)


def test_logic_not(setup, rng):
    x = setup.bits(rng)
    cipher = setup.encrypt(x)
    setup.check(setup.logic.NOT(cipher), 1 - x)
    setup.check(cipher, x)
    setup.logic.NOT_inplace(cipher)
    setup.check(cipher, 1 - x)


def test_logic_circuit(setup, rng):
    # A full adder: sum = x XOR y XOR c, carry = (x AND y) OR (c AND (x XOR y)).
    x, y, c = setup.bits(rng), setup.bits(rng), setup.bits(rng)
    cx, cy, cc = setup.encrypt(x), setup.encrypt(y), setup.encrypt(c)
    logic, rk = setup.logic, setup.relin_key
    half = logic.XOR(cx, cy, rk)
    total = logic.XOR(half, cc, rk)
    carry = logic.OR(logic.AND(cx, cy, rk), logic.AND(cc, half, rk), rk)
    setup.check(total, x ^ y ^ c)
    setup.check(carry, (x & y) | (c & (x ^ y)))


def test_logic_gate_needs_relin_key(setup):
    cipher = setup.encrypt([1])
    with pytest.raises(TypeError):
        setup.logic.AND(cipher, cipher)


@pytest.mark.slow
def test_deep_circuit_large_ring(rng):
    large = BFVOperatorSetup(depth=6, poly_modulus_degree=32768)
    a = large.random(rng)
    op = large.operator
    cipher = large.encrypt(a)
    expected = a.copy()
    for _ in range(6):
        op.multiply_inplace(cipher, cipher)
        op.relinearize_inplace(cipher, large.relin_key)
        expected = np.mod(expected * expected, large.t)
    assert large.decryptor.remainder_noise_budget(cipher) > 0
    large.check(cipher, expected)
    rotated = op.rotate_rows(cipher, large.galois_key, 11)
    large.check(rotated, large.rotated(expected, 11))


def test_relinearized_serialization(setup, rng):
    a, b = setup.random(rng), setup.random(rng)
    op = setup.operator
    product = op.relinearize(
        op.multiply(setup.encrypt(a), setup.encrypt(b)), setup.relin_key
    )
    for loaded in (
        bfv.Ciphertext.from_bytes(product.to_bytes()),
        product.copy(),
    ):
        assert not loaded.relinearization_required
        setup.check(loaded, a * b)
        setup.check(op.add(loaded, setup.encrypt(a)), a * b + a)
