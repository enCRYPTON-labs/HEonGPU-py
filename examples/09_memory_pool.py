# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""The memory pool.

HEonGPU allocates GPU (device) and pinned host memory from pools. By
default the device pool starts at half of the free GPU memory and may grow
to 95% of it. heongpu.initialize() sets other sizes; call it before the
first context is generated (otherwise the first context sets up the pool
with the defaults). The sizes are fractions of the memory that is free when
the pool is created, or byte counts; byte counts take precedence.
"""

import gc

import numpy as np

import heongpu as hg
from heongpu import ckks


def show_usage(label):
    usage = hg.memory_pool_usage()
    print(
        f"{label:<34} device used {usage['device_used'] / 2**20:8.1f} MiB,"
        f" free {usage['device_free'] / 2**20:8.1f} MiB"
    )


def main():
    print("defaults:", hg.MemoryPoolConfig.defaults())

    # A pool that starts at 256 MiB and grows up to 30% of the GPU memory,
    # with at most 1 GiB of pinned host memory.
    config = hg.MemoryPoolConfig(
        initial_device_bytes=256 << 20,
        max_device_fraction=0.3,
        max_host_bytes=1 << 30,
    )
    hg.initialize(config)
    show_usage("after initialize()")

    session = ckks.Session(multiplicative_depth=4, rotations=[1, 2, 3])
    show_usage("after creating a session")
    ciphertexts = [
        session.encrypt(np.full(session.slot_count, float(i)))
        for i in range(20)
    ]
    show_usage("with 20 ciphertexts")

    # Ciphertexts and keys can move to host memory and back.
    for c in ciphertexts:
        c.ciphertext.store_in_host()
    show_usage("ciphertexts moved to the host")
    for c in ciphertexts:
        c.ciphertext.store_in_device()
    assert abs(session.decrypt(ciphertexts[7])[0] - 7.0) < 1e-6

    # Freed objects return their memory to the pool, not to the driver.
    del ciphertexts, c
    gc.collect()
    show_usage("ciphertexts deleted")

    hg.print_memory_pool_status()

    # shutdown() releases the pools; it refuses while HEonGPU objects exist
    # (every Python reference to them must be gone).
    try:
        hg.shutdown()
    except hg.StateError as error:
        print(
            "shutdown() with a live session -> StateError:",
            str(error).splitlines()[0][:60],
            "...",
        )
    del session
    gc.collect()
    hg.shutdown()
    print("shutdown() done")

    # After shutdown() the pool can be set up again with other sizes.
    hg.initialize(
        hg.MemoryPoolConfig(
            initial_device_bytes=128 << 20, max_device_fraction=0.2
        )
    )
    session = ckks.Session(multiplicative_depth=1)
    x = session.encrypt([1.0, 2.0])
    print("new pool works:", session.decrypt(x * x)[:2])
    show_usage("second pool")
    print("OK")


if __name__ == "__main__":
    main()
