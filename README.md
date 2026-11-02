# HEonGPU-py

Python bindings of [HEonGPU](https://github.com/enCRYPTON-labs/HEonGPU), a GPU
library for fully homomorphic encryption. HEonGPU-py gives Python programs the
CUDA implementation of the BFV, CKKS and TFHE schemes, from a one-object
session API down to the individual keys, encoders and operators.

```python
import numpy as np
from heongpu import ckks

session = ckks.Session(ckks.Params(multiplicative_depth=3), rotations=[1])

x = session.encrypt(np.linspace(-1.0, 1.0, session.slot_count))
y = session.encrypt(np.full(session.slot_count, 0.5))

z = (x * y + 1.0).rotate(1)
print(session.decrypt(z)[:4])
```

## Features

- **Schemes**: CKKS (real and complex numbers, bootstrapping, logic
  bootstrapping), BFV (exact integer arithmetic) and TFHE (Boolean gates).
- **Two levels of API**: `Session` (CKKS and BFV) keeps the context, the
  keys and the operators together and gives ciphertexts Python operators (`+`, `-`, `*`);
  the lower level gives every HEonGPU class (`Context`, `KeyGenerator`,
  `Encoder`, `Encryptor`, `Decryptor`, `ArithmeticOperator`, ...).
- **Parameter generation**: `Params` chooses secure parameters from the
  multiplicative depth and the scale; bootstrapping presets give complete
  parameter sets.
- **NumPy in and out**: messages are scalars, sequences or NumPy arrays, and
  results are NumPy arrays.
- **Serialization**: keys, plaintexts and ciphertexts have `to_bytes()`,
  `from_bytes()`, `save()` and `load()` and support `pickle` (except the key
  shares of multiparty computation, and session ciphertexts, which are read
  back by their session). The data is the HEonGPU format, so objects can be
  exchanged with C++ programs. Sessions can be saved for a role (owner,
  evaluator or encryptor) for client and server setups.
- **CUDA streams**: every operation of the lower level API takes `options=`,
  which accepts a `heongpu.Stream`, an `ExecutionOptions`, a raw stream
  handle or a stream of another library (any object with `__cuda_stream__`,
  `cuda_stream`, `ptr` or `handle`, such as CuPy and PyTorch streams).
  `Session.on_stream()` returns a view of a session that runs on a stream.
- **Multiparty computation**: collective key generation, distributed
  decryption and collective bootstrapping for BFV and CKKS.
- **Errors as exceptions**: HEonGPU errors become `heongpu.Error` subclasses
  (`ParameterError`, `StateError`, `ContextMismatchError`,
  `LevelMismatchError`, `KeyNotFoundError`, `SerializationError`, ...), and
  a full memory pool raises `MemoryError`.
- **Typed**: the package ships type stubs (`py.typed`).
- **The GIL is released** during GPU work, so other Python threads keep
  running.

## Installation

```bash
pip install heongpu
```

The wheels on PyPI are for Linux x86_64 (glibc 2.28 or newer) and Python
3.10 or newer. They are built with CUDA 13.0 and contain the CUDA runtime
and the other libraries the module needs, so the only requirement is an
NVIDIA driver of version 580 or newer. They contain the code of the GPU
architectures 7.5 (Turing), 8.0 (Ampere: it also runs on 8.6), 8.9 (Ada),
9.0 (Hopper), 10.0 and 12.0 (Blackwell), and PTX that the driver compiles
for newer GPUs. On other systems pip builds the package from the
source distribution, which needs what a build from source needs.

### Building from source

Requirements:

- Linux with an NVIDIA GPU and driver
- CUDA Toolkit 12.8 or newer
- Python 3.10 or newer, NumPy
- CMake 4.0 or newer (installed by pip if missing), GCC with C++20 support
- The libraries HEonGPU needs: GMP, NTL, OpenSSL and zlib (for example
  `apt install libgmp-dev libntl-dev libssl-dev zlib1g-dev`)

```bash
git clone https://github.com/enCRYPTON-labs/HEonGPU-py.git
cd HEonGPU-py
pip install .

# Or build the source distribution of PyPI instead of using a wheel.
pip install --no-binary heongpu heongpu
```

By default the build uses an installed HEonGPU 2.1 if CMake finds one, and
otherwise downloads and builds HEonGPU v2.1.0 together with the extension
module (this takes several minutes). The CMake option
`HEONGPU_PY_HEONGPU_PROVIDER` selects this explicitly:

| Value    | HEonGPU                                                       |
|----------|---------------------------------------------------------------|
| `AUTO`   | an installed HEonGPU if one is found, otherwise from source   |
| `SYSTEM` | an installed HEonGPU only                                     |
| `FETCH`  | always built from source                                      |

```bash
# Use the HEonGPU installed in /opt/heongpu.
CMAKE_PREFIX_PATH=/opt/heongpu pip install . \
    -Ccmake.define.HEONGPU_PY_HEONGPU_PROVIDER=SYSTEM

# Build for specific GPU architectures (default: the GPU of this machine).
pip install . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="80;89"

# Build HEonGPU from a local source tree.
pip install . -Ccmake.define.HEONGPU_PY_HEONGPU_PROVIDER=FETCH \
    -Ccmake.define.FETCHCONTENT_SOURCE_DIR_HEONGPU=/path/to/HEonGPU
```

The environment variable `HEONGPU_PY_HEONGPU_PROVIDER` works too.

## Usage

### Sessions

A session holds a generated context, the keys, an encoder, an encryptor, a
decryptor and an operator. Its ciphertexts know their session, so arithmetic
needs no further arguments. Sessions relinearize products, and CKKS
sessions in the automatic rescale mode (the default) also rescale them and
bring operands to the same level.

```python
import numpy as np
from heongpu import ckks

params = ckks.Params(multiplicative_depth=4, scale_bit_size=40)
session = ckks.Session(params, rotations=[1, 2])

a = np.random.uniform(-1, 1, session.slot_count)
x = session.encrypt(a)

y = x * x - 0.5 * x + 1.0          # ciphertext-ciphertext and with scalars
y = y * a                          # with a NumPy array
y = y.rotate(2)                    # rotations need their Galois keys
print(y.levels_left)               # 2

expected = np.roll((a * a - 0.5 * a + 1.0) * a, -2)
print(np.max(np.abs(session.decrypt(y) - expected)))   # about 1e-8
```

Every product uses a level. Keep `levels_left` at 0 or above: a ciphertext
that has run out of levels decrypts to wrong values.

BFV works the same with integers:

```python
from heongpu import bfv

session = bfv.Session(bfv.Params(multiplicative_depth=2, plain_modulus=65537))
x = session.encrypt([1, 2, 3, 4])
print(session.decrypt(x * x + 10)[:4])   # [11 14 19 26]
```

### Client and server

A session can be saved for a role. An evaluator has the public and the
evaluation keys but no secret key, so it can compute but not decrypt.

```python
import heongpu as hg
from heongpu import ckks

client = ckks.Session(ckks.Params(multiplicative_depth=3), rotations=[1])
evaluator_data = client.to_bytes(hg.SessionRole.EVALUATOR)

server = ckks.Session.from_bytes(evaluator_data)
x = server.ciphertext_from_bytes(client.encrypt([1.0, 2.0, 3.0]).to_bytes())
result = (x * x).to_bytes()

print(client.decrypt(client.ciphertext_from_bytes(result))[:3])   # about [1. 4. 9.]
```

`save()`, `Session.load()` and `load_ciphertext()` do the same with files
(a path or a binary file object).

### Bootstrapping

```python
from heongpu import ckks

# Slim bootstrapping with 3 levels between two bootstrappings.
session = ckks.Session(ckks.BootstrappingPreset.slim(3))
x = session.encrypt([0.25, 0.5])
while x.levels_left > 0:
    x = x * 1.0
x = x.bootstrap()                  # the levels are restored
print(x.levels_left)               # 3
```

The presets (`regular()`, `slim()`, `bit()` and `gate()`) use N = 2^16 for
128-bit security, and their keys need a lot of GPU memory: a session with
a few levels between two bootstrappings needs a device memory pool of about
12 GiB, and the presets with their default number of levels need more than
13 GiB. A session from a `gate()` preset has `AND_bootstrapping()`,
`OR_bootstrapping()` and the other gates instead of `bootstrap()`.

### The lower level API

```python
import numpy as np
from heongpu import ckks

params = ckks.Params(multiplicative_depth=3, scale_bit_size=40)
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
x = encryptor.encrypt(encoder.encode(np.arange(8) / 8, scale))

y = operator.multiply(x, x)
operator.relinearize_inplace(y, relin_key)
operator.rescale_inplace(y)

print(encoder.decode(decryptor.decrypt(y))[:8])
```

### Streams

```python
import heongpu as hg

stream = hg.Stream()
x = encryptor.encrypt(encoder.encode(values, scale, options=stream), options=stream)
stream.synchronize()

view = session.on_stream(stream)   # a session on the stream
```

`options=` also takes `hg.ExecutionOptions(stream=..., storage=...)`, a CuPy
or PyTorch stream, or an `int` handle. A `heongpu.Stream` is non-blocking by
default: it does not wait for the default stream, so call
`hg.synchronize()` before it uses data made on the default stream, or
create it with `hg.Stream(non_blocking=False)`. Objects remember the stream
they were created on and use it when they are freed: keep a foreign stream
alive while such objects exist. A `heongpu.Stream` takes care of this
itself; its CUDA stream is destroyed by `heongpu.shutdown()`.

### Memory pool

HEonGPU allocates GPU memory from a pool. Configure it before the first
context is generated; fractions are of the memory that is free at that
time, and byte sizes (`initial_device_bytes`, `max_device_bytes`) can be
given instead:

```python
import heongpu as hg

hg.initialize(hg.MemoryPoolConfig(initial_device_fraction=0.2, max_device_fraction=0.8))
print(hg.memory_pool_usage())
```

## Examples and tests

The [examples](examples) directory has runnable scripts for each part of
the API, from sessions to bootstrapping and multiparty computation. The
tests need a CUDA device:

```bash
pip install .[test]
pytest              # without bootstrapping and other long tests
pytest -m slow      # only the long tests
pytest -m ""        # all tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for a development build.

## Known limitations

These behaviors of HEonGPU 2.1.0 affect Python programs; the examples avoid
them.

- **Collective relinearization key**: in multiparty computation (CKKS and
  BFV), the round 2 share of the relinearization key
  (`generate_relin_key_share`) uses uninitialized error polynomials, so the
  collective relinearization key is not always correct and products
  relinearized with it can decrypt to wrong values. Collective public and
  Galois keys, distributed decryption and collective bootstrapping are not
  affected.
- **`ckks.LogicOperator` gates**: OR, XOR, NOR and XNOR of two ciphertexts,
  and OR, XOR and NOR of a ciphertext and a plaintext, raise
  `LevelMismatchError`, since they subtract the rescaled product (one level
  lower) from the sum of the inputs. AND, NAND and NOT work; so do the gates
  of the session API (`ckks.OR(a, b)`, ...) and gate bootstrapping.
- **TFHE gate composition**: a bootstrapped gate output carries a phase
  offset of about -1/16 of the torus, so a gate whose two inputs are both
  gate outputs can give wrong bits. A gate with one gate output and one
  fresh ciphertext as inputs gives correct bits.
- **Non-blocking streams**: in the automatic rescale mode, an operand of the
  default stream that is brought to another level is copied on the default
  stream, which a non-blocking stream does not wait for; and
  `ArithmeticOperator.evaluate_polynomial` copies its input on the input's
  stream without the stream of `options=` waiting for the copy. Use
  blocking streams (`heongpu.Stream(non_blocking=False)`) for session views
  that use ciphertexts of the default stream and for polynomial evaluation
  on a stream.
- **Out of memory while creating a bootstrapping session**: a full memory
  pool usually raises `MemoryError`, but while a session is created from a
  `BootstrappingPreset` it terminates the process. Give the pool enough room
  (see Bootstrapping above) before creating such a session.

## License

HEonGPU-py is licensed under the [Apache License 2.0](LICENSE).
