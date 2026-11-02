# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""CUDA streams: heongpu.Stream, ExecutionOptions, Session.on_stream and
streams of other libraries.

Every operation of the lower level API takes options=, which is a stream or
an ExecutionOptions. Work on different streams can overlap on the GPU, and
the GIL is released during GPU work, so Python threads can drive one stream
each. Sessions run on a stream through a view from Session.on_stream().

Ordering rules:

- Operations are asynchronous: synchronize the stream (or the session view)
  before reading a result on another stream or on the host.
- A non-blocking stream (the default of heongpu.Stream) does not wait for
  the default stream: call heongpu.synchronize() after preparing inputs on
  the default stream.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import heongpu as hg
from heongpu import ckks


class ForeignStream:
    """Stands for the stream of another library. Any object with
    __cuda_stream__ (returning (0, handle)), cuda_stream, ptr or handle is
    accepted as a stream, for example cupy.cuda.Stream or
    torch.cuda.Stream; so is an int handle."""

    def __init__(self, handle):
        self._handle = handle

    def __cuda_stream__(self):
        return (0, self._handle)


def lower_level_api():
    print("== Lower level API")
    params = ckks.Params(multiplicative_depth=2, scale_bit_size=40)
    context = params.make_context()
    context.generate()
    keygen = ckks.KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.public_key(secret_key)
    relin_key = keygen.relin_key(secret_key)
    encoder = ckks.Encoder(context)
    encryptor = ckks.Encryptor(context, public_key)
    decryptor = ckks.Decryptor(context, secret_key)
    operator = ckks.ArithmeticOperator(context, encoder)
    scale = params.scale()
    xs = np.linspace(-1, 1, encoder.slot_count)

    # A stream as a context manager: leaving the block waits for the stream
    # and retires it (heongpu.shutdown() destroys the CUDA stream).
    with hg.Stream() as stream:
        plain = encoder.encode(xs, scale, options=stream)
        x = encryptor.encrypt(plain, options=stream)
        print("ciphertext stream == stream.handle:", x.stream == stream.handle)
        square = operator.multiply(x, x, options=stream)
        operator.relinearize_inplace(square, relin_key, options=stream)
        operator.rescale_inplace(square, options=stream)
        result = encoder.decode(
            decryptor.decrypt(square, options=stream), options=stream
        )
        stream.synchronize()
    np.testing.assert_allclose(result, xs * xs, atol=1e-6)

    # ExecutionOptions also chooses where the output is stored: here the
    # ciphertext is written to host memory.
    options = hg.ExecutionOptions(stream=None, storage=hg.StorageType.HOST)
    on_host = encryptor.encrypt(encoder.encode(xs, scale), options=options)
    print("stored on the device:", on_host.is_on_device)
    np.testing.assert_allclose(
        encoder.decode(decryptor.decrypt(on_host)), xs, atol=1e-6
    )

    # Streams of other libraries: the stream protocol and int handles.
    # Objects remember the stream they were created on and use it when they
    # are freed, so keep a foreign stream alive while they exist.
    owner = hg.Stream()
    for foreign in (ForeignStream(owner.handle), owner.handle):
        # The inputs were made on the default stream; a non-blocking stream
        # does not wait for it.
        hg.synchronize()
        total = operator.add(x, x, options=foreign)
        print(
            "operation on",
            type(foreign).__name__,
            "-> stream",
            hex(total.stream),
        )
    owner.synchronize()
    np.testing.assert_allclose(
        encoder.decode(decryptor.decrypt(total)), 2 * xs, atol=1e-6
    )
    # With CuPy, for example:
    #     with cupy.cuda.Stream(non_blocking=True) as s:
    #         y = operator.add(x, x, options=s)
    #         s.synchronize()
    owner.close()


def sessions_on_streams():
    print("== Sessions on streams, one thread per stream")
    session = ckks.Session(multiplicative_depth=3, rotations=[1])
    xs = 0.5 * np.sin(np.arange(session.slot_count) / 50.0)
    x = session.encrypt(xs)
    reference = session.decrypt((x * x).rotate(1) + x)

    # A view of the session shares its keys and runs on its own stream. The
    # views also use x, a ciphertext of the default stream, which these
    # non-blocking streams do not wait for.
    hg.synchronize()
    streams = [hg.Stream() for _ in range(4)]

    def work(index):
        view = session.on_stream(streams[index])
        a = view.encrypt(xs)  # a ciphertext of the view's stream
        b = (a * x).rotate(1) + a  # x is a ciphertext of the default stream
        values = view.decrypt(b)
        view.synchronize()
        return view.stream, values

    with ThreadPoolExecutor(len(streams)) as pool:
        results = list(pool.map(work, range(len(streams))))
    for handle, values in results:
        np.testing.assert_allclose(values, reference, atol=1e-5)
    print("4 threads, streams", [hex(handle) for handle, _ in results])
    for stream in streams:
        stream.close()


def main():
    lower_level_api()
    sessions_on_streams()
    print("OK")


if __name__ == "__main__":
    main()
