# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

from . import bfv as bfv
from . import ckks as ckks
from . import tfhe as tfhe
from ._heongpu import ArithmeticBootstrappingType as ArithmeticBootstrappingType
from ._heongpu import Encoding as Encoding
from ._heongpu import ExecutionOptions as ExecutionOptions
from ._heongpu import LogicBootstrappingType as LogicBootstrappingType
from ._heongpu import MemoryPoolConfig as MemoryPoolConfig
from ._heongpu import RescaleMode as RescaleMode
from ._heongpu import SchemeType as SchemeType
from ._heongpu import SecurityLevel as SecurityLevel
from ._heongpu import SessionRole as SessionRole
from ._heongpu import StorageType as StorageType
from ._heongpu import Stream as Stream
from ._heongpu import device_count as device_count
from ._heongpu import get_device as get_device
from ._heongpu import initialize as initialize
from ._heongpu import memory_pool_usage as memory_pool_usage
from ._heongpu import print_memory_pool_status as print_memory_pool_status
from ._heongpu import set_device as set_device
from ._heongpu import shutdown as shutdown
from ._heongpu import synchronize as synchronize

__version__: str
heongpu_version: str

class Error(Exception):
    """Common base of the exceptions raised by HEonGPU."""

class ParameterError(Error, ValueError):
    """A parameter or an argument has an invalid value."""

class StateError(Error, ValueError):
    """An object is used before it is ready or after it can no longer
    change."""

class ContextMismatchError(Error, ValueError):
    """An object created with one context is used with a different
    context."""

class LevelMismatchError(Error, ValueError):
    """The operands of an operation are not at the levels it requires."""

class KeyNotFoundError(Error, ValueError):
    """A key needed by an operation is missing."""

class SerializationError(Error, RuntimeError):
    """Serialized data is malformed, truncated or written for a different
    scheme."""

class SystemServiceError(Error, RuntimeError):
    """A system service HEonGPU depends on failed."""

class CudaError(Error, RuntimeError):
    """A CUDA runtime call failed."""
