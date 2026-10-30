# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

import pytest

import heongpu as hg


@pytest.mark.nogpu
def test_version():
    assert isinstance(hg.__version__, str)
    assert hg.heongpu_version.startswith("2.")


@pytest.mark.nogpu
def test_exception_hierarchy():
    for error in (
        hg.ParameterError,
        hg.StateError,
        hg.ContextMismatchError,
        hg.LevelMismatchError,
        hg.KeyNotFoundError,
    ):
        assert issubclass(error, hg.Error)
        assert issubclass(error, ValueError)
    for error in (hg.SerializationError, hg.SystemServiceError, hg.CudaError):
        assert issubclass(error, hg.Error)
        assert issubclass(error, RuntimeError)


@pytest.mark.nogpu
def test_enums():
    assert hg.SecurityLevel.SEC128 != hg.SecurityLevel.SEC192
    assert hg.RescaleMode.AUTOMATIC.name == "AUTOMATIC"
    assert {role.name for role in hg.SessionRole} == {
        "OWNER",
        "EVALUATOR",
        "ENCRYPTOR",
    }


@pytest.mark.nogpu
def test_memory_pool_config():
    config = hg.MemoryPoolConfig(max_device_fraction=0.5)
    assert config.max_device_fraction == pytest.approx(0.5)
    assert config.initial_device_bytes is None
    config.initial_device_bytes = 1 << 20
    assert config.initial_device_bytes == 1 << 20
    assert "max_device_fraction=0.5" in repr(config)
    defaults = hg.MemoryPoolConfig.defaults()
    assert defaults.max_device_fraction is not None


@pytest.mark.nogpu
def test_execution_options_from_int():
    options = hg.ExecutionOptions(stream=0, storage=hg.StorageType.HOST)
    assert options.stream == 0
    assert options.storage == hg.StorageType.HOST
    assert options.keep_initial_condition


def test_stream_protocol():
    class ForeignStream:
        def __init__(self, handle):
            self.handle_value = handle

        def __cuda_stream__(self):
            return (0, self.handle_value)

    with hg.Stream() as stream:
        assert stream.handle != 0
        assert stream.__cuda_stream__() == (0, stream.handle)
        assert hg.ExecutionOptions(stream=stream).stream == stream.handle
        foreign = ForeignStream(stream.handle)
        assert hg.ExecutionOptions(stream=foreign).stream == stream.handle
        stream.synchronize()
    assert stream.closed
    with pytest.raises(hg.StateError):
        stream.synchronize()


def test_stream_rejects_other_objects():
    with pytest.raises(TypeError):
        hg.ExecutionOptions(stream="not a stream")


def test_devices():
    assert hg.device_count() >= 1
    device = hg.get_device()
    hg.set_device(device)
    hg.synchronize()


def test_memory_pool_usage():
    usage = hg.memory_pool_usage()
    assert set(usage) == {"device_used", "device_free", "host_used", "host_free"}
