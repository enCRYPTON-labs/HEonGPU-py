// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

NB_MODULE(_heongpu, m)
{
    using namespace heongpu_py;

    m.doc() = "Python bindings of HEonGPU, a GPU library for fully "
              "homomorphic encryption.";

    bind_exceptions(m);
    bind_core(m);

    nb::module_ ckks =
        m.def_submodule("ckks", "The CKKS scheme: approximate arithmetic on "
                                "real and complex numbers.");
    bind_ckks_core(ckks);
    bind_ckks_operator(ckks);
    bind_ckks_session(ckks);
    bind_ckks_mpc(ckks);

    nb::module_ bfv = m.def_submodule(
        "bfv", "The BFV scheme: exact arithmetic on integers modulo the "
               "plaintext modulus.");
    bind_bfv_core(bfv);
    bind_bfv_operator(bfv);
    bind_bfv_session(bfv);
    bind_bfv_mpc(bfv);

    nb::module_ tfhe = m.def_submodule(
        "tfhe", "The TFHE scheme: Boolean gates with bootstrapping.");
    bind_tfhe(tfhe);
}
