// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

#include <unordered_map>

namespace heongpu_py
{
    namespace
    {
        constexpr auto BFV = heongpu::Scheme::BFV;

        using ContextPtr = heongpu::HEContext<BFV>;
        using SecretKey = heongpu::Secretkey<BFV>;
        using PublicKey = heongpu::Publickey<BFV>;
        using RelinKey = heongpu::Relinkey<BFV>;
        using GaloisKey = heongpu::Galoiskey<BFV>;
        using MultipartyPublicKey = heongpu::MultipartyPublickey<BFV>;
        using MultipartyRelinKey = heongpu::MultipartyRelinkey<BFV>;
        using MultipartyGaloisKey = heongpu::MultipartyGaloiskey<BFV>;
        using Plaintext = heongpu::Plaintext<BFV>;
        using Ciphertext = heongpu::Ciphertext<BFV>;
        using Manager = heongpu::HEMultiPartyManager<BFV>;
        using heongpu::RNGSeed;

        /**
         * @brief The multiparty manager together with its context, which is
         * needed to create the objects the manager methods return.
         *
         * HEMultiPartyManager<Scheme::BFV> also draws a private random value
         * when it is created, which both rounds of the relinearization key
         * generation of a participant use; so a participant keeps its
         * manager between the two rounds.
         *
         * The name carries the scheme: nanobind identifies bound types by
         * their names, which are the same for types of equal names in the
         * anonymous namespaces of different files.
         */
        struct BFVMultiPartyManager
        {
            explicit BFVMultiPartyManager(ContextPtr context_)
                : context(context_), manager(std::move(context_))
            {
            }

            ContextPtr context;
            Manager manager;
        };

        nb::bytes to_bytes(const std::vector<unsigned char>& data)
        {
            return nb::bytes(reinterpret_cast<const char*>(data.data()),
                             data.size());
        }

        std::vector<unsigned char> from_bytes(nb::bytes data)
        {
            const auto* begin =
                reinterpret_cast<const unsigned char*>(data.c_str());
            return std::vector<unsigned char>(begin, begin + data.size());
        }

        /**
         * @brief The given seed, or a new random one.
         */
        RNGSeed seed_or_random(const RNGSeed* seed)
        {
            return seed ? *seed : RNGSeed();
        }

        /**
         * @brief The partial decryption and collective bootstrapping read the
         * first two polynomials of a ciphertext only, so a product must be
         * relinearized first.
         */
        void require_relinearized(const Ciphertext& cipher)
        {
            if (cipher.relinearization_required())
            {
                throw heongpu::StateError(
                    "The ciphertext must be relinearized first.");
            }
        }

        /**
         * @brief The partial decryption and collective bootstrapping do not
         * check that their operands belong to the context of the manager, so
         * the shapes are compared here.
         */
        void check_ciphertext(const ContextPtr& context,
                              const Ciphertext& cipher, const char* name)
        {
            if (static_cast<std::size_t>(cipher.ring_size()) !=
                    context->get_poly_modulus_degree() ||
                cipher.coeff_modulus_count() !=
                    context->get_ciphertext_modulus_count())
            {
                throw heongpu::ContextMismatchError(
                    std::string(name) +
                    " does not belong to the context of the manager.");
            }
        }

        void check_secret_key(const ContextPtr& context,
                              const SecretKey& secret_key)
        {
            if (static_cast<std::size_t>(secret_key.ring_size()) !=
                    context->get_poly_modulus_degree() ||
                secret_key.coeff_modulus_count() !=
                    context->get_key_modulus_count())
            {
                throw heongpu::ContextMismatchError(
                    "The secret key does not belong to the context of the "
                    "manager.");
            }
        }

        // Some members the bindings need are private. Pointers to them are
        // formed in explicit instantiations, where access checking does not
        // apply.
        template <typename Tag, typename Tag::type Member> struct MemberAccess
        {
            friend typename Tag::type member_pointer(Tag) { return Member; }
        };

        // HEMultiPartyManager<Scheme::BFV>::decrypt_partial() does not mark
        // its output as generated, so a partial decryption could not be
        // serialized and sent to the party that fuses them.
        struct CiphertextGeneratedTag
        {
            using type = bool Ciphertext::*;
            friend type member_pointer(CiphertextGeneratedTag);
        };

        template struct MemberAccess<CiphertextGeneratedTag,
                                     &Ciphertext::ciphertext_generated_>;

        // HEMultiPartyManager<Scheme::BFV>::assemble_galois_key_share()
        // aggregates the keys of the rotation steps of the first share only
        // (not the keys of explicit Galois elements), and reads the other
        // shares without checking that they have these keys.
        struct GaloisCustomizedTag
        {
            using type = bool GaloisKey::*;
            friend type member_pointer(GaloisCustomizedTag);
        };

        struct GaloisElementsTag
        {
            using type = std::unordered_map<int, int> GaloisKey::*;
            friend type member_pointer(GaloisElementsTag);
        };

        template struct MemberAccess<GaloisCustomizedTag,
                                     &GaloisKey::customized>;
        template struct MemberAccess<GaloisElementsTag, &GaloisKey::galois_elt>;

        void check_galois_shares(const std::vector<MultipartyGaloisKey>& shares)
        {
            const GaloisKey& first = shares[0];
            for (const GaloisKey& share : shares)
            {
                if (share.*member_pointer(GaloisCustomizedTag{}))
                {
                    throw heongpu::ParameterError(
                        "Galois key shares generated for explicit Galois "
                        "elements cannot be combined (HEonGPU aggregates the "
                        "keys of rotation steps only); generate the shares "
                        "with shifts.");
                }
                if (share.*member_pointer(GaloisElementsTag{}) !=
                    first.*member_pointer(GaloisElementsTag{}))
                {
                    throw heongpu::ParameterError(
                        "The Galois key shares must be generated for the "
                        "same rotation steps.");
                }
            }
        }

        /**
         * @brief Host storage is filled by asynchronous copies to pinned
         * memory; waits for them, so that the returned object can be read
         * (and copied) on the host right away.
         */
        void finish(const heongpu::ExecutionOptions& options)
        {
            if (options.storage_ == heongpu::storage_type::host)
            {
                HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
            }
        }

        template <typename T>
        void require_shares(const std::vector<T>& shares, const char* what)
        {
            if (shares.empty())
            {
                throw nb::value_error(
                    (std::string("No ") + what + " to combine.").c_str());
            }
        }

        /**
         * @brief Binds heongpu::RNGSeed once. The type is shared by the
         * schemes; if another part of the module registered it already, the
         * existing type is added to this module under the same name.
         */
        void bind_rng_seed(nb::module_& m)
        {
            nb::handle existing = nb::type<RNGSeed>();
            if (existing.is_valid())
            {
                if (!nb::hasattr(m, "RNGSeed"))
                {
                    m.attr("RNGSeed") = existing;
                }
                return;
            }

            nb::class_<RNGSeed> seed(
                m, "RNGSeed",
                "Seed of the random generator for multiparty computation. All "
                "participants use the same seed to sample the common random "
                "polynomials of the collective keys and of collective "
                "bootstrapping. Without arguments a random seed (a 128-bit "
                "key and a 64-bit nonce) is drawn from the operating system.");
            seed.def(nb::init<>())
                .def(
                    "__init__",
                    [](RNGSeed* self, nb::bytes key, nb::bytes nonce,
                       nb::bytes personalization_string)
                    {
                        new (self) RNGSeed(from_bytes(key), from_bytes(nonce),
                                           from_bytes(personalization_string));
                    },
                    "key"_a, "nonce"_a,
                    "personalization_string"_a = nb::bytes("", 0),
                    "A seed with the given key (at least 16 bytes), nonce and "
                    "personalization string.")
                .def_prop_ro("key", [](const RNGSeed& self)
                             { return to_bytes(self.key_); })
                .def_prop_ro("nonce", [](const RNGSeed& self)
                             { return to_bytes(self.nonce_); })
                .def_prop_ro("personalization_string", [](const RNGSeed& self)
                             { return to_bytes(self.personalization_string_); })
                .def(
                    "__eq__",
                    [](const RNGSeed& self, const RNGSeed& other)
                    {
                        return self.key_ == other.key_ &&
                               self.nonce_ == other.nonce_ &&
                               self.personalization_string_ ==
                                   other.personalization_string_;
                    },
                    nb::is_operator())
                .def("__getstate__",
                     [](const RNGSeed& self)
                     {
                         return nb::make_tuple(
                             to_bytes(self.key_), to_bytes(self.nonce_),
                             to_bytes(self.personalization_string_));
                     })
                .def("__setstate__",
                     [](RNGSeed& self, nb::tuple state)
                     {
                         if (state.size() != 3)
                         {
                             throw nb::value_error("Invalid RNGSeed state.");
                         }
                         new (&self)
                             RNGSeed(from_bytes(nb::cast<nb::bytes>(state[0])),
                                     from_bytes(nb::cast<nb::bytes>(state[1])),
                                     from_bytes(nb::cast<nb::bytes>(state[2])));
                     })
                .def("__repr__",
                     [](const RNGSeed& self)
                     {
                         return "<heongpu.RNGSeed key=" +
                                std::to_string(self.key_.size()) +
                                " bytes nonce=" +
                                std::to_string(self.nonce_.size()) + " bytes>";
                     });
            def_copy<RNGSeed>(seed);
        }

        /**
         * @brief Adds the serialization methods of a multiparty key. The data
         * is the HEonGPU format of the base key (save() and to_bytes() are
         * inherited); the seed is not part of it, so loading needs the
         * context and, optionally, the seed. Pickling is not supported
         * because a multiparty key cannot be created without its context.
         */
        template <typename Key, typename Class>
        void def_multiparty_key_serialization(Class& cls, const char* name)
        {
            cls.def_static(
                   "from_bytes",
                   [](nb::bytes data, ContextPtr context, const RNGSeed* seed)
                   {
                       Key key(std::move(context), seed_or_random(seed));
                       load_from_string(key, bytes_to_string(data));
                       return key;
                   },
                   "data"_a, "context"_a, "seed"_a = nb::none(),
                   "Deserializes a key written by to_bytes() or by the C++ "
                   "save(). The seed is not stored in the data; give it if "
                   "the seed property is needed (by default a random one).")
                .def_static(
                    "load",
                    [](nb::handle file, ContextPtr context, const RNGSeed* seed)
                    {
                        Key key(std::move(context), seed_or_random(seed));
                        load_from_string(key, read_file(file));
                        return key;
                    },
                    "file"_a, "context"_a, "seed"_a = nb::none(),
                    "Reads a key from a path or from a binary file object.")
                .def("__getstate__",
                     [name](const Key&) -> nb::bytes
                     {
                         throw nb::type_error(
                             (std::string(name) +
                              " cannot be pickled; use to_bytes() and "
                              "from_bytes(data, context).")
                                 .c_str());
                     })
                .def("__setstate__",
                     [name](Key&, nb::bytes)
                     {
                         throw nb::type_error(
                             (std::string(name) +
                              " cannot be pickled; use to_bytes() and "
                              "from_bytes(data, context).")
                                 .c_str());
                     });
        }
    } // namespace

    void bind_bfv_mpc(nb::module_& m)
    {
        bind_rng_seed(m);

        // --------------------------------------------------------------
        // Multiparty keys
        // --------------------------------------------------------------

        nb::class_<MultipartyPublicKey, PublicKey> public_key(
            m, "MultipartyPublicKey",
            "The public key share of one participant, generated with a common "
            "seed. MultiPartyManager.assemble_public_key_share() combines the "
            "shares into the collective PublicKey.");
        public_key
            .def(nb::init<ContextPtr, const RNGSeed>(), "context"_a, "seed"_a)
            .def_prop_ro("seed", &MultipartyPublicKey::seed,
                         "The common seed of the key.");
        def_multiparty_key_serialization<MultipartyPublicKey>(
            public_key, "MultipartyPublicKey");
        def_copy<MultipartyPublicKey>(public_key);

        nb::class_<MultipartyRelinKey, RelinKey> relin_key(
            m, "MultipartyRelinKey",
            "A relinearization key share or the common round 1 key of the "
            "two-round collective relinearization key generation.");
        relin_key
            .def(nb::init<ContextPtr, const RNGSeed>(), "context"_a, "seed"_a)
            .def_prop_ro("seed", &MultipartyRelinKey::seed,
                         "The common seed of the key.");
        def_multiparty_key_serialization<MultipartyRelinKey>(
            relin_key, "MultipartyRelinKey");
        def_copy<MultipartyRelinKey>(relin_key);

        nb::class_<MultipartyGaloisKey, GaloisKey> galois_key(
            m, "MultipartyGaloisKey",
            "The Galois key share of one participant, generated with a common "
            "seed. MultiPartyManager.assemble_galois_key_share() combines the "
            "shares into the collective GaloisKey.");
        galois_key
            .def(
                "__init__",
                [](MultipartyGaloisKey* self, ContextPtr context,
                   const RNGSeed& seed, std::optional<std::vector<int>> shifts,
                   std::optional<std::vector<std::uint32_t>> galois_elements)
                {
                    if (shifts && galois_elements)
                    {
                        throw nb::value_error(
                            "Give only one of shifts and galois_elements.");
                    }
                    if (shifts)
                    {
                        new (self) MultipartyGaloisKey(std::move(context),
                                                       *shifts, seed);
                    }
                    else if (galois_elements)
                    {
                        new (self) MultipartyGaloisKey(std::move(context),
                                                       *galois_elements, seed);
                    }
                    else
                    {
                        new (self)
                            MultipartyGaloisKey(std::move(context), seed);
                    }
                },
                "context"_a, "seed"_a, nb::kw_only(), "shifts"_a = nb::none(),
                "galois_elements"_a = nb::none(),
                "Key share for the given row rotation steps, for the given "
                "Galois elements, or by default for the row rotations of "
                "GaloisKey(context). The column rotation is always included.")
            .def_prop_ro("seed", &MultipartyGaloisKey::seed,
                         "The common seed of the key.");
        def_multiparty_key_serialization<MultipartyGaloisKey>(
            galois_key, "MultipartyGaloisKey");
        def_copy<MultipartyGaloisKey>(galois_key);

        // --------------------------------------------------------------
        // Multiparty manager
        // --------------------------------------------------------------

        nb::class_<BFVMultiPartyManager>(
            m, "MultiPartyManager",
            "Multiparty BFV protocols: collective public, relinearization and "
            "Galois key generation from the participants' secret keys, "
            "collective (distributed) decryption from partial decryptions, "
            "and collective bootstrapping. Each participant and the server "
            "use their own manager; a participant must use the same manager "
            "for both rounds of the relinearization key generation.")
            .def(nb::init<ContextPtr>(), "context"_a)
            .def_prop_ro(
                "context",
                [](const BFVMultiPartyManager& self) { return self.context; },
                "The context of the manager.")

            // Public key
            .def(
                "generate_public_key_share",
                [](BFVMultiPartyManager& self, const SecretKey& secret_key,
                   const RNGSeed& seed, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    MultipartyPublicKey key(self.context, seed);
                    self.manager.generate_public_key_share(key, secret_key,
                                                           opts);
                    finish(opts);
                    return key;
                },
                "secret_key"_a, "seed"_a, "options"_a = nb::none(),
                "Generates the public key share of a participant from its "
                "secret key and the common seed.")
            .def(
                "assemble_public_key_share",
                [](BFVMultiPartyManager& self,
                   const std::vector<MultipartyPublicKey>& shares,
                   nb::handle options)
                {
                    require_shares(shares, "public key shares");
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    // The list items are copies made on the streams of the
                    // originals; wait for them.
                    HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
                    PublicKey key(self.context);
                    self.manager.assemble_public_key_share(shares, key, opts);
                    finish(opts);
                    return key;
                },
                "shares"_a, "options"_a = nb::none(),
                "Combines the public key shares of all participants (a list) "
                "into the collective public key.")

            // Relinearization key
            .def(
                "generate_relin_key_init",
                [](BFVMultiPartyManager& self, const SecretKey& secret_key,
                   const RNGSeed& seed, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    MultipartyRelinKey key(self.context, seed);
                    self.manager.generate_relin_key_init(key, secret_key, opts);
                    finish(opts);
                    return key;
                },
                "secret_key"_a, "seed"_a, "options"_a = nb::none(),
                "Round 1 of the relinearization key generation: the "
                "participant's first share, from its secret key and the "
                "common seed.")
            .def(
                "assemble_relin_key_init",
                [](BFVMultiPartyManager& self,
                   const std::vector<MultipartyRelinKey>& shares,
                   const RNGSeed* seed, nb::handle options)
                {
                    require_shares(shares, "relinearization key shares");
                    heongpu::ExecutionOptions opts = to_options(options);
                    RNGSeed key_seed = seed ? *seed : shares[0].seed();
                    nb::gil_scoped_release release;
                    // The list items are copies made on the streams of the
                    // originals; wait for them.
                    HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
                    MultipartyRelinKey key(self.context, key_seed);
                    self.manager.assemble_relin_key_init(shares, key, opts);
                    finish(opts);
                    return key;
                },
                "shares"_a, "seed"_a = nb::none(), "options"_a = nb::none(),
                "Combines the round 1 shares of all participants (a list) "
                "into the common round 1 key, which goes back to the "
                "participants. seed defaults to the seed of the first share.")
            .def(
                "generate_relin_key_share",
                [](BFVMultiPartyManager& self, const MultipartyRelinKey& common,
                   const SecretKey& secret_key, const RNGSeed* seed,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    RNGSeed key_seed = seed ? *seed : common.seed();
                    nb::gil_scoped_release release;
                    MultipartyRelinKey key(self.context, key_seed);
                    self.manager.generate_relin_key_share(common, key,
                                                          secret_key, opts);
                    finish(opts);
                    return key;
                },
                "common"_a, "secret_key"_a, "seed"_a = nb::none(),
                "options"_a = nb::none(),
                "Round 2 of the relinearization key generation: the "
                "participant's second share, from the common round 1 key and "
                "its secret key. Use the manager that generated the "
                "participant's round 1 share. seed defaults to the seed of "
                "common.")
            .def(
                "assemble_relin_key_share",
                [](BFVMultiPartyManager& self,
                   const std::vector<MultipartyRelinKey>& shares,
                   const MultipartyRelinKey& common, nb::handle options)
                {
                    require_shares(shares, "relinearization key shares");
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    // The list items are copies made on the streams of the
                    // originals; wait for them.
                    HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
                    RelinKey key(self.context);
                    self.manager.assemble_relin_key_share(shares, common, key,
                                                          opts);
                    finish(opts);
                    return key;
                },
                "shares"_a, "common"_a, "options"_a = nb::none(),
                "Combines the round 2 shares of all participants (a list) and "
                "the common round 1 key into the collective relinearization "
                "key.")

            // Galois key
            .def(
                "generate_galois_key_share",
                [](BFVMultiPartyManager& self, const SecretKey& secret_key,
                   const RNGSeed& seed, std::optional<std::vector<int>> shifts,
                   std::optional<std::vector<std::uint32_t>> galois_elements,
                   nb::handle options)
                {
                    if (shifts && galois_elements)
                    {
                        throw nb::value_error(
                            "Give only one of shifts and galois_elements.");
                    }
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    auto make_key = [&]()
                    {
                        if (shifts)
                        {
                            return MultipartyGaloisKey(self.context, *shifts,
                                                       seed);
                        }
                        if (galois_elements)
                        {
                            return MultipartyGaloisKey(self.context,
                                                       *galois_elements, seed);
                        }
                        return MultipartyGaloisKey(self.context, seed);
                    };
                    MultipartyGaloisKey key = make_key();
                    self.manager.generate_galois_key_share(key, secret_key,
                                                           opts);
                    finish(opts);
                    return key;
                },
                "secret_key"_a, "seed"_a, nb::kw_only(),
                "shifts"_a = nb::none(), "galois_elements"_a = nb::none(),
                "options"_a = nb::none(),
                "Generates the Galois key share of a participant for the "
                "given row rotation steps, for the given Galois elements, or "
                "by default for the row rotations of GaloisKey(context); the "
                "column rotation is always included. All participants must "
                "use the same seed and rotations. assemble_galois_key_share() "
                "combines shares of rotation steps only, so use shifts for "
                "keys that are combined.")
            .def(
                "assemble_galois_key_share",
                [](BFVMultiPartyManager& self,
                   const std::vector<MultipartyGaloisKey>& shares,
                   nb::handle options)
                {
                    require_shares(shares, "Galois key shares");
                    check_galois_shares(shares);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    // The list items are copies made on the streams of the
                    // originals; wait for them.
                    HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
                    // The collective key takes the rotations of the shares;
                    // its data is replaced by the aggregation.
                    GaloisKey key(static_cast<const GaloisKey&>(shares[0]));
                    self.manager.assemble_galois_key_share(shares, key, opts);
                    finish(opts);
                    return key;
                },
                "shares"_a, "options"_a = nb::none(),
                "Combines the Galois key shares of all participants (a list) "
                "into the collective Galois key, for the rotations of the "
                "shares. Raises ParameterError if the shares are for "
                "different rotation steps or for explicit Galois elements.")

            // Decryption
            .def(
                "decrypt_partial",
                [](BFVMultiPartyManager& self, const Ciphertext& ciphertext,
                   const SecretKey& secret_key, nb::handle options)
                {
                    check_ciphertext(self.context, ciphertext,
                                     "The ciphertext");
                    check_secret_key(self.context, secret_key);
                    require_relinearized(ciphertext);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    // decrypt_partial() writes the output to the device only.
                    heongpu::ExecutionOptions device_opts = opts;
                    device_opts.storage_ = heongpu::storage_type::device;
                    Ciphertext partial(self.context, device_opts);
                    self.manager.decrypt_partial(ciphertext, secret_key,
                                                 partial, device_opts);
                    partial.*member_pointer(CiphertextGeneratedTag{}) = true;
                    if (opts.storage_ == heongpu::storage_type::host)
                    {
                        partial.store_in_host(opts.stream_);
                    }
                    finish(opts);
                    return partial;
                },
                "ciphertext"_a, "secret_key"_a, "options"_a = nb::none(),
                "The partial decryption of a ciphertext with one "
                "participant's secret key. Raises StateError for a product "
                "that is not relinearized yet.")
            .def(
                "decrypt",
                [](BFVMultiPartyManager& self,
                   const std::vector<Ciphertext>& partials, nb::handle options)
                {
                    require_shares(partials, "partial decryptions");
                    for (const Ciphertext& partial : partials)
                    {
                        check_ciphertext(self.context, partial,
                                         "A partial decryption");
                    }
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    // The list items are copies made on the streams of the
                    // originals; wait for them.
                    HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
                    Plaintext plain(self.context, opts);
                    self.manager.decrypt(partials, plain, opts);
                    finish(opts);
                    return plain;
                },
                "partials"_a, "options"_a = nb::none(),
                "Fuses the partial decryptions of all participants (a list) "
                "into the plaintext; decode it with Encoder.decode().")

            // Collective bootstrapping
            .def(
                "distributed_bootstrapping_participant",
                [](BFVMultiPartyManager& self, const Ciphertext& common,
                   const SecretKey& secret_key, const RNGSeed& seed,
                   nb::handle options)
                {
                    check_ciphertext(self.context, common, "The ciphertext");
                    check_secret_key(self.context, secret_key);
                    require_relinearized(common);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    Ciphertext share(self.context, opts);
                    self.manager.distributed_bootstrapping_participant(
                        common, share, secret_key, seed, opts);
                    finish(opts);
                    return share;
                },
                "common"_a, "secret_key"_a, "seed"_a, "options"_a = nb::none(),
                "A participant's share of collective bootstrapping for the "
                "ciphertext common, with its secret key and the common seed "
                "(a fresh seed for each refresh).")
            .def(
                "distributed_bootstrapping_coordinator",
                [](BFVMultiPartyManager& self,
                   const std::vector<Ciphertext>& shares,
                   const Ciphertext& common, const RNGSeed& seed,
                   nb::handle options)
                {
                    require_shares(shares, "bootstrapping shares");
                    check_ciphertext(self.context, common, "The ciphertext");
                    for (const Ciphertext& share : shares)
                    {
                        check_ciphertext(self.context, share,
                                         "A bootstrapping share");
                    }
                    require_relinearized(common);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    // The list items are copies made on the streams of the
                    // originals; wait for them.
                    HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
                    Ciphertext output(self.context, opts);
                    self.manager.distributed_bootstrapping_coordinator(
                        shares, common, output, seed, opts);
                    finish(opts);
                    return output;
                },
                "shares"_a, "common"_a, "seed"_a, "options"_a = nb::none(),
                "Combines the participants' shares (a list) into a refreshed "
                "encryption of common, with a fresh noise budget. seed is the "
                "seed the participants used.");
    }

} // namespace heongpu_py
