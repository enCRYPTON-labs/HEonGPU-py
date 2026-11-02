# HEonGPU-py examples

Runnable scripts that show the Python API, from the one-object session API
down to the individual keys, encoders and operators. Each script checks its
results and prints `OK` at the end.

```bash
python examples/01_ckks_session.py
```

The memory column is the device memory pool the script needed in a test on
an RTX 4090 (it ran in a pool limited to that size). By default HEonGPU
reserves half of the free GPU memory for its pool when the first context is
generated and lets the pool grow to 95% of it; see
[09_memory_pool.py](09_memory_pool.py) to choose other sizes.

| Example | What it shows | GPU memory |
|---|---|---|
| [01_ckks_session.py](01_ckks_session.py) | CKKS with `ckks.Session`: encryption, `+ - * ** /` with ciphertexts, constants and arrays, levels, rotations, sums, complex numbers, logic gates on bits | 128 MiB |
| [02_bfv_session.py](02_bfv_session.py) | BFV with `bfv.Session`: exact integer arithmetic modulo t, signed decryption, the noise budget, row and column rotations, logic gates | 128 MiB |
| [03_ckks_basic.py](03_ckks_basic.py) | The lower level CKKS API in the manual rescale mode: context, keys, encoder, `multiply`, `relinearize`, `rescale`, `mod_drop`, rotations, conjugation, key switching | 128 MiB |
| [04_bfv_basic.py](04_bfv_basic.py) | The lower level BFV API: keys, products with relinearization, plaintext operands, rotations, `sum_slots`, key switching, `LogicOperator` | 128 MiB |
| [05_ckks_automatic_rescale.py](05_ckks_automatic_rescale.py) | The lower level CKKS API in the automatic rescale mode: the operators rescale and align levels and scales | 128 MiB |
| [06_ckks_rotations_and_polynomials.py](06_ckks_rotations_and_polynomials.py) | A matrix-vector product with hoisted rotations; polynomials and Chebyshev series, including a Python function (the sigmoid) | 256 MiB |
| [07_serialization_client_server.py](07_serialization_client_server.py) | Sessions saved for the owner, evaluator and encryptor roles; ciphertexts as bytes and files; keys and ciphertexts with `to_bytes`, `save`/`load` and `pickle` | 128 MiB |
| [08_streams.py](08_streams.py) | `heongpu.Stream`, `options=`, `ExecutionOptions` (stream and output storage), streams of other libraries and `int` handles, `Session.on_stream` with one thread per stream | 128 MiB |
| [09_memory_pool.py](09_memory_pool.py) | `MemoryPoolConfig`, `initialize`, `memory_pool_usage`, moving ciphertexts to host memory, `shutdown` and a second pool | 256 MiB (sets its own pool) |
| [10_tfhe_gates.py](10_tfhe_gates.py) | TFHE Boolean gates: truth tables, `MUX`, `out=`, gate chains, a client and a server, streams | 512 MiB |
| [11_ckks_bootstrapping.py](11_ckks_bootstrapping.py) | CKKS slim and regular bootstrapping with a session from a `BootstrappingPreset` (N = 2^16) | 12 GiB |
| [12_ckks_logic_bootstrapping.py](12_ckks_logic_bootstrapping.py) | CKKS bit bootstrapping (cleaning noisy bits) and gate bootstrapping (a 4-bit adder in every slot) | 12 GiB |
| [13_mpc_ckks.py](13_mpc_ckks.py) | CKKS multiparty computation: collective public and Galois keys, distributed decryption, collective bootstrapping | 128 MiB |
| [14_mpc_bfv.py](14_mpc_bfv.py) | BFV multiparty computation: an encrypted vote count, distributed decryption, collective bootstrapping | 128 MiB |

The bootstrapping examples use presets with fewer levels between two
bootstrappings than the default presets (3 for slim and regular, 2 for bit,
1 for gate), since their keys need less memory; the default presets need
more than 13 GiB.
