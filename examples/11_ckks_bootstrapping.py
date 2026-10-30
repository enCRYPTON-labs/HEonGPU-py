# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""CKKS bootstrapping with a Session.

Every product uses a level; bootstrapping gives a ciphertext new levels, so
a computation can be as deep as needed. A Session created from a
BootstrappingPreset has the bootstrapping keys, and SessionCiphertext.
bootstrap() refreshes a ciphertext.

The presets use N = 2^16 for 128-bit security. Their keys are large: with
3 levels between two bootstrappings, as here, a session needs a device
memory pool of about 12 GiB; the presets with their default number of
levels (9) need more than 13 GiB. Fewer levels need less memory and more
bootstrappings.

Two kinds of arithmetic bootstrapping:

- slim: for real values; cheaper. Its input must still have
  bootstrapping_levels().input_level levels (levels_left >= 0).
- regular: for complex values; it takes its input at any level.
"""

import gc
import time

import numpy as np

import heongpu as hg
from heongpu import ckks

USABLE_LEVELS = 3  # levels between two bootstrappings


def slim_bootstrapping():
    print("== Slim bootstrapping: 12 steps of the logistic map")
    start = time.time()
    session = ckks.Session(ckks.BootstrappingPreset.slim(USABLE_LEVELS))
    levels = session.bootstrapping_levels()
    print(f"session created in {time.time() - start:.1f} s:", session)
    print(
        f"  levels: input {levels.input_level}, output {levels.output_level},"
        f" usable {levels.usable_levels}; slots {session.slot_count}"
    )

    # x <- r x (1 - x) for r = 3.7 (a chaotic map; the CKKS error grows, so
    # compare with a tolerance). Each step uses 2 levels: bootstrap when
    # fewer are left.
    xs = np.linspace(0.1, 0.9, session.slot_count)
    x = session.encrypt(xs)
    expected = xs.copy()
    bootstrappings = 0
    for step in range(12):
        if x.levels_left < 2:
            start = time.time()
            x = x.bootstrap()
            bootstrappings += 1
            elapsed = time.time() - start
        x = 3.7 * x * (1.0 - x)
        expected = 3.7 * expected * (1.0 - expected)
    error = np.abs(session.decrypt(x) - expected).max()
    print(
        f"  {bootstrappings} bootstrappings (the last took {elapsed:.2f} s),"
        f" max error {error:.1e}"
    )
    assert error < 1e-2
    return session


def regular_bootstrapping():
    print("== Regular bootstrapping: complex values")
    session = ckks.Session(ckks.BootstrappingPreset.regular(USABLE_LEVELS))
    n = session.slot_count
    zs = 0.5 * np.exp(2j * np.pi * np.arange(n) / n)  # on a circle
    z = session.encrypt(zs)
    while z.levels_left > 0:
        z = z * 1.0
    print("  before:", z)
    z = z.bootstrap()
    print("  after: ", z)
    error = np.abs(session.decrypt_complex(z) - zs).max()
    print(f"  max error {error:.1e}")
    assert error < 1e-3


def main():
    session = slim_bootstrapping()
    usage = hg.memory_pool_usage()
    print(
        f"  device pool: {usage['device_used'] / 2**30:.1f} GiB in use,"
        f" {(usage['device_used'] + usage['device_free']) / 2**30:.1f} GiB"
        " reserved"
    )
    # Free the keys of the first session before creating the second one.
    del session
    gc.collect()
    regular_bootstrapping()
    print("OK")


if __name__ == "__main__":
    main()
