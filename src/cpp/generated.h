// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_PY_GENERATED_H
#define HEONGPU_PY_GENERATED_H

#include "common.h"

namespace heongpu_py
{
    // Accessors of the "generated" flags of keys and plaintexts, used to check
    // that an object holds data before an operation uses it. The tags are in
    // an unnamed namespace, so every translation unit has its own
    // instantiations.
    namespace
    {
        template <typename Tag, typename Tag::type Member>
        struct GeneratedFlagAccess
        {
            friend typename Tag::type generated_flag(Tag) { return Member; }
        };

#define HEONGPU_PY_GENERATED_FLAG(TAG, CLASS, MEMBER)                          \
    struct TAG                                                                 \
    {                                                                          \
        using type = bool CLASS::*;                                            \
        friend type generated_flag(TAG);                                       \
    };                                                                         \
    template struct GeneratedFlagAccess<TAG, &CLASS::MEMBER>;

        HEONGPU_PY_GENERATED_FLAG(CKKSPlaintextFlag,
                                  heongpu::Plaintext<heongpu::Scheme::CKKS>,
                                  plaintext_generated_)
        HEONGPU_PY_GENERATED_FLAG(CKKSPublicKeyFlag,
                                  heongpu::Publickey<heongpu::Scheme::CKKS>,
                                  public_key_generated_)
        HEONGPU_PY_GENERATED_FLAG(CKKSRelinKeyFlag,
                                  heongpu::Relinkey<heongpu::Scheme::CKKS>,
                                  relin_key_generated_)
        HEONGPU_PY_GENERATED_FLAG(CKKSGaloisKeyFlag,
                                  heongpu::Galoiskey<heongpu::Scheme::CKKS>,
                                  galois_key_generated_)
        HEONGPU_PY_GENERATED_FLAG(CKKSSwitchKeyFlag,
                                  heongpu::Switchkey<heongpu::Scheme::CKKS>,
                                  switch_key_generated_)
        HEONGPU_PY_GENERATED_FLAG(BFVPlaintextFlag,
                                  heongpu::Plaintext<heongpu::Scheme::BFV>,
                                  plaintext_generated_)
        HEONGPU_PY_GENERATED_FLAG(BFVPublicKeyFlag,
                                  heongpu::Publickey<heongpu::Scheme::BFV>,
                                  public_key_generated_)
        HEONGPU_PY_GENERATED_FLAG(BFVRelinKeyFlag,
                                  heongpu::Relinkey<heongpu::Scheme::BFV>,
                                  relin_key_generated_)
        HEONGPU_PY_GENERATED_FLAG(BFVGaloisKeyFlag,
                                  heongpu::Galoiskey<heongpu::Scheme::BFV>,
                                  galois_key_generated_)
        HEONGPU_PY_GENERATED_FLAG(BFVSwitchKeyFlag,
                                  heongpu::Switchkey<heongpu::Scheme::BFV>,
                                  switch_key_generated_)

#undef HEONGPU_PY_GENERATED_FLAG

        template <typename Tag, typename T>
        void require_flag(const T& object, const char* name)
        {
            if (!(object.*generated_flag(Tag{})))
            {
                throw heongpu::StateError(
                    std::string("The ") + name +
                    " is not generated! Generate it, or load it, first.");
            }
        }

        inline void
        require_generated(const heongpu::Plaintext<heongpu::Scheme::CKKS>& p)
        {
            require_flag<CKKSPlaintextFlag>(p, "plaintext");
        }
        inline void
        require_generated(const heongpu::Publickey<heongpu::Scheme::CKKS>& k)
        {
            require_flag<CKKSPublicKeyFlag>(k, "public key");
        }
        inline void
        require_generated(const heongpu::Relinkey<heongpu::Scheme::CKKS>& k)
        {
            require_flag<CKKSRelinKeyFlag>(k, "relinearization key");
        }
        inline void
        require_generated(const heongpu::Galoiskey<heongpu::Scheme::CKKS>& k)
        {
            require_flag<CKKSGaloisKeyFlag>(k, "Galois key");
        }
        inline void
        require_generated(const heongpu::Switchkey<heongpu::Scheme::CKKS>& k)
        {
            require_flag<CKKSSwitchKeyFlag>(k, "switch key");
        }
        inline void
        require_generated(const heongpu::Plaintext<heongpu::Scheme::BFV>& p)
        {
            require_flag<BFVPlaintextFlag>(p, "plaintext");
        }
        inline void
        require_generated(const heongpu::Publickey<heongpu::Scheme::BFV>& k)
        {
            require_flag<BFVPublicKeyFlag>(k, "public key");
        }
        inline void
        require_generated(const heongpu::Relinkey<heongpu::Scheme::BFV>& k)
        {
            require_flag<BFVRelinKeyFlag>(k, "relinearization key");
        }
        inline void
        require_generated(const heongpu::Galoiskey<heongpu::Scheme::BFV>& k)
        {
            require_flag<BFVGaloisKeyFlag>(k, "Galois key");
        }
        inline void
        require_generated(const heongpu::Switchkey<heongpu::Scheme::BFV>& k)
        {
            require_flag<BFVSwitchKeyFlag>(k, "switch key");
        }

        /**
         * @brief Marks a plaintext as generated.
         */
        inline void
        mark_generated(heongpu::Plaintext<heongpu::Scheme::BFV>& plain)
        {
            plain.*generated_flag(BFVPlaintextFlag{}) = true;
        }
    } // namespace

} // namespace heongpu_py

#endif // HEONGPU_PY_GENERATED_H
