# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""CKKS bootstrapping: regular, slim, bit and gate.

The presets have N = 2^16 (about 11 GB of GPU memory for regular
bootstrapping). These tests use the same moduli and configurations with
N = 2^12 and no security level, which need about 400 MB each. Set
HEONGPU_PY_TEST_FULL_PRESETS=1 to also run a full preset.
"""

import os

import numpy as np
import pytest

import heongpu as hg
from heongpu import ckks

pytestmark = pytest.mark.slow

SMALL_RING = 4096


class BootstrappingSetup:
    """A context, keys and operator for a bootstrapping preset."""

    def __init__(self, preset, automatic=False, security_level=None):
        self.preset = preset
        self.levels = preset.levels()
        if security_level is None:
            security_level = hg.SecurityLevel.NONE
        self.context = ckks.Context(security_level)
        preset.apply(self.context)
        self.context.generate()
        if automatic:
            self.context.set_rescale_mode(
                hg.RescaleMode.AUTOMATIC,
                preset.scale,
                self.levels.output_level,
            )
        self.keygen = ckks.KeyGenerator(self.context)
        self.secret_key = ckks.SecretKey(
            self.context, preset.secret_key_hamming_weight
        )
        self.keygen.generate_secret_key(self.secret_key)
        public_key = self.keygen.public_key(self.secret_key)
        self.encoder = ckks.Encoder(self.context)
        self.encryptor = ckks.Encryptor(self.context, public_key)
        self.decryptor = ckks.Decryptor(self.context, self.secret_key)
        self.slot_count = self.encoder.slot_count
        self.scale = preset.scale
        self.automatic = automatic

        if preset.logic_type == hg.LogicBootstrappingType.NONE:
            self.operator = ckks.ArithmeticOperator(self.context, self.encoder)
        else:
            self.operator = ckks.LogicOperator(
                self.context, self.encoder, preset.scale
            )
        self.operator.generate_bootstrapping_params(preset)
        self.keys = self.keygen.bootstrapping_keys(
            self.secret_key,
            self.operator.bootstrapping_key_indices(),
            preset.sparse_secret_key_hamming_weight,
        )

    def encrypt(self, values):
        scale = None if self.automatic else self.scale
        return self.encryptor.encrypt(self.encoder.encode(values, scale))

    def decrypt(self, cipher):
        return self.encoder.decode(self.decryptor.decrypt(cipher))

    def decrypt_complex(self, cipher):
        return self.encoder.decode_complex(self.decryptor.decrypt(cipher))


def small_preset(name, usable_levels=None):
    factory = getattr(ckks.BootstrappingPreset, name)
    preset = factory() if usable_levels is None else factory(usable_levels)
    preset.poly_modulus_degree = SMALL_RING
    return preset


@pytest.fixture(scope="module")
def regular():
    return BootstrappingSetup(small_preset("regular"))


@pytest.fixture(scope="module")
def slim():
    return BootstrappingSetup(small_preset("slim"))


@pytest.fixture(scope="module")
def bit():
    return BootstrappingSetup(small_preset("bit"))


@pytest.fixture(scope="module")
def gate():
    return BootstrappingSetup(small_preset("gate"))


def message(setup):
    return 0.5 * np.sin(0.01 * np.arange(setup.slot_count))


def bits(setup):
    index = np.arange(setup.slot_count)
    return (index % 2).astype(float), ((index // 2) % 2).astype(float)


# ----------------------------------------------------------------------
# Presets and parameters
# ----------------------------------------------------------------------


def test_preset_levels():
    cases = {
        "regular": (8, 0, 9, 9),
        "slim": (8, 3, 12, 9),
        "bit": (9, 3, 11, 8),
        "gate": (10, 3, 10, 7),
    }
    for name, (eval_mod, input_level, output_level, usable) in cases.items():
        preset = getattr(ckks.BootstrappingPreset, name)()
        levels = preset.levels()
        assert levels.eval_mod == eval_mod
        assert levels.input_level == input_level
        assert levels.output_level == output_level
        assert levels.usable_levels == usable
        assert levels.depth == 7 + eval_mod
    assert ckks.BootstrappingPreset.slim(3).levels().usable_levels == 3


def test_key_indices_and_keys(slim):
    indices = slim.operator.bootstrapping_key_indices()
    assert len(indices) > 0
    assert all(isinstance(index, int) for index in indices)
    assert slim.keys.dense_to_sparse is not None
    assert slim.keys.sparse_to_dense is not None


def test_wrong_preset_type(regular, bit):
    with pytest.raises(hg.ParameterError):
        regular.operator.generate_bootstrapping_params(
            ckks.BootstrappingPreset.bit()
        )
    with pytest.raises(hg.ParameterError):
        bit.operator.generate_bootstrapping_params(
            ckks.BootstrappingPreset.regular()
        )


# ----------------------------------------------------------------------
# Arithmetic bootstrapping
# ----------------------------------------------------------------------


def test_regular_bootstrapping(regular):
    x = message(regular)
    z = x + 0.25j * np.cos(0.01 * np.arange(regular.slot_count))
    cipher = regular.encryptor.encrypt(regular.encoder.encode(z, regular.scale))
    # The input is at the top level; bootstrapping brings it down.
    result = regular.operator.regular_bootstrapping(cipher, regular.keys)
    assert result.level == regular.levels.output_level
    values = regular.decrypt_complex(result)
    np.testing.assert_allclose(values, z, atol=2.0**-12)
    stats = ckks.get_precision_stats(z, values)
    assert stats.min_precision.real > 12.0
    assert stats.min_precision.imag > 12.0


def test_regular_bootstrapping_then_compute(regular):
    x = message(regular)
    cipher = regular.encrypt(x)
    op = regular.operator
    result = op.regular_bootstrapping(cipher, regular.keys)
    square = op.multiply(result, result)
    op.relinearize_inplace(square, regular.keys.relin_key)
    op.rescale_inplace(square)
    assert square.level == regular.levels.output_level - 1
    np.testing.assert_allclose(regular.decrypt(square), x * x, atol=2.0**-11)


def test_slim_bootstrapping(slim):
    x = message(slim)
    cipher = slim.encrypt(x)
    result = slim.operator.slim_bootstrapping(cipher, slim.keys)
    assert result.level == slim.levels.output_level
    values = slim.decrypt(result)
    np.testing.assert_allclose(values, x, atol=2.0**-18)
    stats = ckks.get_precision_stats(x, values)
    assert stats.mean_precision.real > 18.0


def test_slim_bootstrapping_input_too_low(slim):
    cipher = slim.encrypt(message(slim))
    while cipher.level >= slim.levels.input_level:
        slim.operator.mod_drop_inplace(cipher)
    with pytest.raises(hg.LevelMismatchError):
        slim.operator.slim_bootstrapping(cipher, slim.keys)


def test_slim_bootstrapping_with_config(slim):
    """generate_bootstrapping_params(scale, config, type) with the
    configuration of the preset gives the same keys and results."""
    preset = slim.preset
    operator = ckks.ArithmeticOperator(slim.context, slim.encoder)
    operator.generate_bootstrapping_params(
        preset.scale, preset.config, hg.ArithmeticBootstrappingType.SLIM
    )
    assert operator.bootstrapping_key_indices() == (
        slim.operator.bootstrapping_key_indices()
    )
    x = message(slim)
    result = operator.slim_bootstrapping(slim.encrypt(x), slim.keys)
    np.testing.assert_allclose(slim.decrypt(result), x, atol=2.0**-18)


def test_bootstrapping_with_loaded_keys(slim):
    keys = slim.keygen.bootstrapping_keys(
        slim.secret_key,
        slim.operator.bootstrapping_key_indices(),
        slim.preset.sparse_secret_key_hamming_weight,
    )
    keys.galois_key = ckks.GaloisKey.from_bytes(keys.galois_key.to_bytes())
    keys.relin_key = ckks.RelinKey.from_bytes(keys.relin_key.to_bytes())
    keys.dense_to_sparse = ckks.SwitchKey.from_bytes(
        keys.dense_to_sparse.to_bytes()
    )
    keys.sparse_to_dense = ckks.SwitchKey.from_bytes(
        keys.sparse_to_dense.to_bytes()
    )
    x = message(slim)
    result = slim.operator.slim_bootstrapping(slim.encrypt(x), keys)
    np.testing.assert_allclose(slim.decrypt(result), x, atol=2.0**-18)


def test_bootstrapping_on_stream(slim):
    x = message(slim)
    cipher = slim.encrypt(x)
    hg.synchronize()
    with hg.Stream(non_blocking=False) as stream:
        result = slim.operator.slim_bootstrapping(
            cipher, slim.keys, options=stream
        )
        stream.synchronize()
    np.testing.assert_allclose(slim.decrypt(result), x, atol=2.0**-18)


@pytest.mark.parametrize("name", ["regular", "slim"])
def test_automatic_mode_bootstrapping(name):
    setup = BootstrappingSetup(small_preset(name), automatic=True)
    op = setup.operator
    top = setup.levels.output_level
    assert setup.context.top_level == top
    x = message(setup) + 0.25
    cipher = setup.encrypt(x)
    assert cipher.level == top
    # The product waits for its rescaling, which bootstrapping does.
    square = op.multiply(cipher, cipher)
    op.relinearize_inplace(square, setup.keys.relin_key)
    if name == "regular":
        result = op.regular_bootstrapping(square, setup.keys)
        tolerance = 2.0**-12
    else:
        result = op.slim_bootstrapping(square, setup.keys)
        tolerance = 2.0**-18
    assert result.level == top
    assert result.scale == setup.context.get_level_scale(top)
    np.testing.assert_allclose(setup.decrypt(result), x * x, atol=tolerance)
    # The result is a ciphertext of the automatic mode.
    following = op.multiply(result, result)
    op.relinearize_inplace(following, setup.keys.relin_key)
    following = op.add(following, result)
    np.testing.assert_allclose(
        setup.decrypt(following), x**4 + x * x, atol=2 * tolerance
    )


def test_automatic_mode_needs_output_level():
    preset = small_preset("slim")
    context = ckks.Context(hg.SecurityLevel.NONE)
    preset.apply(context)
    context.generate()
    context.set_rescale_mode(hg.RescaleMode.AUTOMATIC, preset.scale)
    operator = ckks.ArithmeticOperator(context, ckks.Encoder(context))
    with pytest.raises(hg.ParameterError):
        operator.generate_bootstrapping_params(preset)


# ----------------------------------------------------------------------
# Logic bootstrapping
# ----------------------------------------------------------------------


def test_bit_bootstrapping(bit):
    a, _ = bits(bit)
    cipher = bit.encrypt(a)
    result = bit.operator.bit_bootstrapping(cipher, bit.keys)
    assert result.level == bit.levels.output_level
    np.testing.assert_allclose(bit.decrypt(result), a, atol=2.0**-16)


def test_bit_bootstrapping_cleans_noise(bit):
    a, _ = bits(bit)
    # Bits with an error of up to 0.05 come back with a much smaller error.
    noise = 0.05 * np.cos(np.arange(bit.slot_count))
    result = bit.operator.bit_bootstrapping(bit.encrypt(a + noise), bit.keys)
    error = np.abs(bit.decrypt(result) - a)
    assert error.max() < 0.25 * np.abs(noise).max()


GATES = {
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "XOR": lambda a, b: a ^ b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR": lambda a, b: 1 - (a | b),
    "XNOR": lambda a, b: 1 - (a ^ b),
}


@pytest.mark.parametrize("name", sorted(GATES))
def test_gate_bootstrapping(gate, name):
    a, b = bits(gate)
    x, y = gate.encrypt(a), gate.encrypt(b)
    result = getattr(gate.operator, name + "_bootstrapping")(x, y, gate.keys)
    assert result.level == gate.levels.output_level
    expected = GATES[name](a.astype(int), b.astype(int))
    np.testing.assert_allclose(gate.decrypt(result), expected, atol=2.0**-16)


def test_logic_bootstrapping_with_config(bit):
    preset = bit.preset
    operator = ckks.LogicOperator(bit.context, bit.encoder, preset.scale)
    with pytest.raises(ValueError):
        operator.bootstrapping_key_indices()
    operator.generate_bootstrapping_params(
        preset.scale, preset.config, hg.LogicBootstrappingType.BIT
    )
    assert operator.bootstrapping_key_indices() == (
        bit.operator.bootstrapping_key_indices()
    )
    a, _ = bits(bit)
    result = operator.bit_bootstrapping(bit.encrypt(a), bit.keys)
    np.testing.assert_allclose(bit.decrypt(result), a, atol=2.0**-16)


# ----------------------------------------------------------------------
# A full preset (N = 2^16, 128-bit security)
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("HEONGPU_PY_TEST_FULL_PRESETS") != "1",
    reason="set HEONGPU_PY_TEST_FULL_PRESETS=1 (needs several GB of GPU "
    "memory)",
)
def test_full_slim_preset():
    setup = BootstrappingSetup(
        ckks.BootstrappingPreset.slim(), security_level=hg.SecurityLevel.SEC128
    )
    assert setup.slot_count == 2**15
    x = message(setup)
    result = setup.operator.slim_bootstrapping(setup.encrypt(x), setup.keys)
    assert result.level == setup.levels.output_level
    np.testing.assert_allclose(setup.decrypt(result), x, atol=2.0**-18)
