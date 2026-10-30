# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""Serialization: client and server sessions, bytes, files and pickle.

A session is saved for a role:

- OWNER: all keys, including the secret key (the client keeps it),
- EVALUATOR: the public and evaluation keys (a server computes, but cannot
  decrypt),
- ENCRYPTOR: the public key only (a data source encrypts, but can neither
  compute products nor decrypt).

The data is the HEonGPU format, so it can also be read by C++ programs that
use HEonGPU. In this example all parties run in one process; in practice the
bytes travel over a network or through files.
"""

import io
import pickle
import tempfile
from pathlib import Path

import numpy as np

import heongpu as hg
from heongpu import bfv, ckks

Role = hg.SessionRole


def client_server():
    print("== Client and server (CKKS session)")
    # The client creates the session and keeps the secret key.
    client = ckks.Session(multiplicative_depth=3, rotations=[1])
    server_keys = client.to_bytes(Role.EVALUATOR)
    source_keys = client.to_bytes(Role.ENCRYPTOR)
    print(
        f"evaluator keys: {len(server_keys) / 2**20:.1f} MiB,"
        f" encryptor keys: {len(source_keys) / 2**20:.1f} MiB"
    )

    # A data source encrypts with the public key only.
    source = ckks.Session.from_bytes(source_keys, expected_role=Role.ENCRYPTOR)
    values = np.linspace(0.0, 1.0, source.slot_count)
    request = source.encrypt(values).to_bytes()

    # The server computes on the ciphertext without the secret key.
    # expected_role makes from_bytes() refuse a session with other keys.
    server = ckks.Session.from_bytes(server_keys, expected_role=Role.EVALUATOR)
    print("server:", server)
    x = server.ciphertext_from_bytes(request)
    response = ((x * x).rotate(1) + 0.5).to_bytes()

    # The server cannot decrypt, nor create new rotation keys.
    try:
        server.decrypt(x)
    except hg.StateError as error:
        print("server.decrypt  -> StateError:", error)
    try:
        x.rotate(3)
    except hg.KeyNotFoundError as error:
        print("x.rotate(3)     -> KeyNotFoundError:", str(error)[:50], "...")

    # The client decrypts the response.
    result = client.decrypt(client.ciphertext_from_bytes(response))
    np.testing.assert_allclose(result, np.roll(values**2, -1) + 0.5, atol=1e-5)
    print("client result:", result[:4])


def files():
    print("== Files (BFV session)")
    owner = bfv.Session(bfv.Params(multiplicative_depth=1, plain_modulus=65537))
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        # save() and load() take a path or a binary file object.
        owner.save(directory / "evaluator.bin", Role.EVALUATOR)
        with open(directory / "owner.bin", "wb") as file:
            owner.save(file, Role.OWNER)

        evaluator = bfv.Session.load(directory / "evaluator.bin")
        x = owner.encrypt([1, 2, 3])
        x.save(directory / "x.bin")

        y = evaluator.load_ciphertext(directory / "x.bin")
        (y * y + 1).save(directory / "y.bin")

        with open(directory / "owner.bin", "rb") as file:
            restored = bfv.Session.load(file, expected_role=Role.OWNER)
        result = restored.decrypt(restored.load_ciphertext(directory / "y.bin"))
        print("restored owner decrypts (x*x + 1):", result[:3])
        assert list(result[:3]) == [2, 5, 10]

        # A file of another role is refused before any key is read.
        try:
            bfv.Session.load(directory / "evaluator.bin", Role.OWNER)
        except hg.StateError as error:
            print("load(..., OWNER) -> StateError:", error)


def lower_level_objects():
    print("== Keys, plaintexts and ciphertexts of the lower level API")
    params = ckks.Params(multiplicative_depth=2)
    context = params.make_context()
    context.generate()
    keygen = ckks.KeyGenerator(context)
    secret_key = keygen.secret_key()
    public_key = keygen.public_key(secret_key)
    encoder = ckks.Encoder(context)
    scale = params.scale()

    # Every object has to_bytes() / from_bytes() and save() / load().
    context_data = context.to_bytes()
    loaded_context = ckks.Context.from_bytes(context_data)
    print(
        "context:",
        len(context_data),
        "bytes, same context id:",
        loaded_context.context_id == context.context_id,
    )

    buffer = io.BytesIO()
    public_key.save(buffer)
    buffer.seek(0)
    loaded_public_key = ckks.PublicKey.load(buffer)

    # pickle uses the same format.
    loaded_secret_key = pickle.loads(pickle.dumps(secret_key))

    encryptor = ckks.Encryptor(context, loaded_public_key)
    decryptor = ckks.Decryptor(context, loaded_secret_key)
    plain = encoder.encode([0.25, 0.5, 0.75], scale)
    cipher = encryptor.encrypt(pickle.loads(pickle.dumps(plain)))
    restored = ckks.Ciphertext.from_bytes(cipher.to_bytes())
    values = encoder.decode(decryptor.decrypt(restored))[:3]
    print("decrypted with the loaded keys:", values)
    np.testing.assert_allclose(values, [0.25, 0.5, 0.75], atol=1e-6)

    # Objects can be moved to host memory, for example to keep many
    # ciphertexts without using GPU memory, and back.
    restored.store_in_host()
    print("on device after store_in_host():", restored.is_on_device)
    restored.store_in_device()

    # Session ciphertexts are not pickled (they need their session): use
    # to_bytes() and Session.ciphertext_from_bytes(). Session.wrap() turns a
    # Ciphertext of the lower level API into one.
    session = ckks.Session(multiplicative_depth=1)
    try:
        pickle.dumps(session.encrypt(1.0))
    except TypeError as error:
        print(
            "pickle a SessionCiphertext -> TypeError:", str(error)[:50], "..."
        )


def main():
    client_server()
    files()
    lower_level_objects()
    print("OK")


if __name__ == "__main__":
    main()
