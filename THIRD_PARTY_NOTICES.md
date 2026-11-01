# Third-party software

HEonGPU-py is licensed under the [Apache License 2.0](LICENSE). The wheels on
PyPI contain the following third-party software. A build from source uses
the same software, from the system or downloaded by the build.

## Linked into the extension module

| Software | License |
|----------|---------|
| [HEonGPU](https://github.com/enCRYPTON-labs/HEonGPU), with GPU-NTT, GPU-FFT and RNGonGPU | Apache-2.0 |
| [CCCL](https://github.com/NVIDIA/cccl) (Thrust, CUB, libcu++) | Apache-2.0 WITH LLVM-exception |
| [NVTX](https://github.com/NVIDIA/NVTX) | Apache-2.0 WITH LLVM-exception |
| [nanobind](https://github.com/wjakob/nanobind) | BSD-3-Clause |

## Shared libraries in `heongpu.libs` of the wheels

| Software | License |
|----------|---------|
| [RMM](https://github.com/rapidsai/rmm) | Apache-2.0 |
| [rapids-logger](https://github.com/rapidsai/rapids-logger), with [spdlog](https://github.com/gabime/spdlog) | Apache-2.0, MIT |
| CUDA Runtime 13.0 (`libcudart`) | [NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/) (redistributable runtime component) |
| [NTL](https://libntl.org) 11.5.1 | LGPL-2.1-or-later |
| [GMP](https://gmplib.org) 6.1.2 | LGPL-3.0-or-later OR GPL-2.0-or-later |
| [OpenSSL](https://www.openssl.org) 1.1.1k (`libcrypto`) | OpenSSL License and SSLeay License; Apache-2.0 for the changes of the AlmaLinux package |

NTL and GMP are shared libraries under the LGPL: they can be replaced by
other builds of the same versions. Their source code is available from their
websites; NTL is built by [ci/manylinux_before_all.sh](ci/manylinux_before_all.sh)
from the release tarball, and GMP and OpenSSL are the AlmaLinux 8
packages of the `manylinux_2_28` image, whose sources are the AlmaLinux
source packages. zlib and the C and C++ runtime libraries come from the
system.
