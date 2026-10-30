# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

import os

import numpy as np
import pytest

import heongpu as hg
from heongpu import ckks


def pytest_collection_modifyitems(config, items):
    if hg.device_count() > 0:
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "nogpu" not in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def memory_pool():
    """Starts with a small device pool, so that several test processes can
    share a GPU. HEONGPU_PY_TEST_MAX_DEVICE_FRACTION limits its growth."""
    if hg.device_count() == 0:
        yield
        return
    max_fraction = float(
        os.environ.get("HEONGPU_PY_TEST_MAX_DEVICE_FRACTION", "0.8")
    )
    hg.initialize(
        hg.MemoryPoolConfig(
            initial_device_bytes=512 * 1024 * 1024,
            max_device_fraction=max_fraction,
        )
    )
    yield


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(2026)


class CKKSSetup:
    """A manual-rescale CKKS context with its keys and helpers."""

    def __init__(self, depth=4, rescale_mode=None, rotations=None):
        self.params = ckks.Params(
            multiplicative_depth=depth,
            scale_bit_size=40,
            poly_modulus_degree=16384,
            rescale_mode=rescale_mode,
        )
        self.scale = self.params.scale()
        self.context = self.params.make_context()
        self.context.generate()
        keygen = ckks.KeyGenerator(self.context)
        self.keygen = keygen
        self.secret_key = keygen.secret_key()
        self.public_key = keygen.public_key(self.secret_key)
        self.relin_key = keygen.relin_key(self.secret_key)
        self.galois_key = keygen.galois_key(self.secret_key, rotations)
        self.encoder = ckks.Encoder(self.context)
        self.encryptor = ckks.Encryptor(self.context, self.public_key)
        self.decryptor = ckks.Decryptor(self.context, self.secret_key)
        self.slot_count = self.encoder.slot_count

    def encrypt(self, values, scale=None):
        if scale is None and self.context.rescale_mode == hg.RescaleMode.MANUAL:
            scale = self.scale
        return self.encryptor.encrypt(self.encoder.encode(values, scale))

    def decrypt(self, cipher):
        return self.encoder.decode(self.decryptor.decrypt(cipher))

    def decrypt_complex(self, cipher):
        return self.encoder.decode_complex(self.decryptor.decrypt(cipher))


@pytest.fixture(scope="session")
def ckks_setup():
    return CKKSSetup()


@pytest.fixture(scope="session")
def ckks_auto_setup():
    return CKKSSetup(rescale_mode=hg.RescaleMode.AUTOMATIC)
