# Changelog

All notable changes to HEonGPU-py are listed here. The project follows
[Semantic Versioning](https://semver.org/).

## 0.1.0 (unreleased)

The first release of the Python bindings, for HEonGPU 2.1.

- The CKKS, BFV and TFHE schemes: contexts, parameter generation, keys,
  encoders, encryptors, decryptors, key generators and operators.
- The session API of CKKS and BFV, with Python operators on ciphertexts,
  roles (owner, evaluator, encryptor) and client/server serialization.
- CKKS bootstrapping (regular, slim, bit and gate) with presets.
- Multiparty computation for CKKS and BFV.
- Serialization of every key, plaintext and ciphertext to bytes and files in
  the HEonGPU format, and pickle support.
- CUDA streams: `heongpu.Stream`, `ExecutionOptions`, and streams of other
  libraries through `__cuda_stream__`, `cuda_stream`, `ptr` or `handle`.
- The memory pool configuration, HEonGPU errors as Python exceptions, type
  stubs, and released GIL during GPU work.
- The build uses an installed HEonGPU or builds HEonGPU from source.
