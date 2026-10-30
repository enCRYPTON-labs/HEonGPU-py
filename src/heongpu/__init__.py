# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""Python bindings of HEonGPU, a GPU library for fully homomorphic encryption.

The schemes are in submodules:

- :mod:`heongpu.ckks`: approximate arithmetic on real and complex numbers,
- :mod:`heongpu.bfv`: exact arithmetic on integers,
- :mod:`heongpu.tfhe`: Boolean gates with bootstrapping.

The top-level module holds what the schemes share: the enumerations, the
exceptions, CUDA streams, execution options and the memory pool.
"""

from ._heongpu import (
    ArithmeticBootstrappingType,
    ContextMismatchError,
    CudaError,
    Encoding,
    Error,
    ExecutionOptions,
    KeyNotFoundError,
    LevelMismatchError,
    LogicBootstrappingType,
    MemoryPoolConfig,
    ParameterError,
    RescaleMode,
    SchemeType,
    SecurityLevel,
    SerializationError,
    SessionRole,
    StateError,
    StorageType,
    Stream,
    SystemServiceError,
    __version__,
    device_count,
    get_device,
    heongpu_version,
    initialize,
    memory_pool_usage,
    print_memory_pool_status,
    set_device,
    shutdown,
    synchronize,
)
from . import bfv, ckks, tfhe

__all__ = [
    "ArithmeticBootstrappingType",
    "ContextMismatchError",
    "CudaError",
    "Encoding",
    "Error",
    "ExecutionOptions",
    "KeyNotFoundError",
    "LevelMismatchError",
    "LogicBootstrappingType",
    "MemoryPoolConfig",
    "ParameterError",
    "RescaleMode",
    "SchemeType",
    "SecurityLevel",
    "SerializationError",
    "SessionRole",
    "StateError",
    "StorageType",
    "Stream",
    "SystemServiceError",
    "__version__",
    "bfv",
    "ckks",
    "device_count",
    "get_device",
    "heongpu_version",
    "initialize",
    "memory_pool_usage",
    "print_memory_pool_status",
    "set_device",
    "shutdown",
    "synchronize",
    "tfhe",
]
