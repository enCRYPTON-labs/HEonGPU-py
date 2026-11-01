#!/usr/bin/env bash
# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

# Prepares a manylinux_2_28 container for building the wheels: installs the
# CUDA Toolkit, the system libraries of HEonGPU, and builds and installs
# HEonGPU once into /opt/heongpu, which the builds of all Python versions use.
#
# Environment:
#   HEONGPU_PY_CUDA_VERSION        the CUDA version, for example 13.0
#   HEONGPU_PY_CUDA_ARCHITECTURES  the CMAKE_CUDA_ARCHITECTURES of HEonGPU
#   HEONGPU_PY_HEONGPU_TAG         the HEonGPU tag to build

set -euo pipefail

cuda_version=${HEONGPU_PY_CUDA_VERSION:?}
cuda_architectures=${HEONGPU_PY_CUDA_ARCHITECTURES:?}
heongpu_tag=${HEONGPU_PY_HEONGPU_TAG:?}
cuda_package_version=${cuda_version//./-}

ntl_version=11.5.1
ntl_sha256=210d06c31306cbc6eaf6814453c56c776d9d8e8df36d74eb306f6a523d1c6a8a

# GMP, OpenSSL and zlib.
dnf install -y gmp-devel openssl-devel zlib-devel git

# NTL has no package for EL8: a shared library built from source, without
# tuning for the CPU of the build machine.
work=$(mktemp -d)
curl -sSfL -o "${work}/ntl.tar.gz" "https://libntl.org/ntl-${ntl_version}.tar.gz"
echo "${ntl_sha256}  ${work}/ntl.tar.gz" | sha256sum -c -
tar -xzf "${work}/ntl.tar.gz" -C "${work}"
(
    cd "${work}/ntl-${ntl_version}/src"
    ./configure PREFIX=/usr/local SHARED=on NTL_GMP_LIP=on NTL_GF2X_LIB=off \
        NTL_THREADS=on NATIVE=off TUNE=x86
    make -j "$(nproc)"
    make install
)
rm -rf "${work}"

# The parts of the CUDA Toolkit that HEonGPU needs.
dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y \
    "cuda-nvcc-${cuda_package_version}" \
    "cuda-cudart-devel-${cuda_package_version}" \
    "libcurand-devel-${cuda_package_version}"
dnf clean all

cuda_home=/usr/local/cuda-${cuda_version}
export PATH=${cuda_home}/bin:${PATH}

# The image has CMake 4, but not Ninja.
pipx install ninja

# HEonGPU, built once for all Python versions.
work=$(mktemp -d)
git clone --depth 1 --branch "${heongpu_tag}" --recurse-submodules \
    --shallow-submodules https://github.com/enCRYPTON-labs/HEonGPU.git \
    "${work}/HEonGPU"
cmake -S "${work}/HEonGPU" -B "${work}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
    -DCMAKE_INSTALL_PREFIX=/opt/heongpu
cmake --build "${work}/build" --parallel "$(nproc)"
cmake --install "${work}/build"
rm -rf "${work}"
