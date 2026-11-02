// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

#include <type_traits>

namespace heongpu_py
{
    namespace
    {
        constexpr auto CKKS = heongpu::Scheme::CKKS;

        using ContextPtr = heongpu::HEContext<CKKS>;
        using SecretKey = heongpu::Secretkey<CKKS>;
        using PublicKey = heongpu::Publickey<CKKS>;
        using RelinKey = heongpu::Relinkey<CKKS>;
        using GaloisKey = heongpu::Galoiskey<CKKS>;
        using MultipartyPublicKey = heongpu::MultipartyPublickey<CKKS>;
        using MultipartyRelinKey = heongpu::MultipartyRelinkey<CKKS>;
        using MultipartyGaloisKey = heongpu::MultipartyGaloiskey<CKKS>;
        using Plaintext = heongpu::Plaintext<CKKS>;
        using Ciphertext = heongpu::Ciphertext<CKKS>;
        using Encoder = heongpu::HEEncoder<CKKS>;
        using Manager = heongpu::HEMultiPartyManager<CKKS>;
        using heongpu::RNGSeed;

        /**
         * @brief The multiparty manager together with the context and the
         * scale it was created with. The context is needed to create the
         * objects the manager methods return.
         */
        struct MultiPartyManager
        {
            MultiPartyManager(ContextPtr context_, Encoder& encoder,
                              double scale_)
                : context(context_), scale(scale_),
                  manager(std::move(context_), encoder, scale_)
            {
            }

            ContextPtr context;
            double scale;
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
         * @brief Copies the objects of a Python list into the vector the C++
         * API takes. Objects in host memory may still be written by an
         * asynchronous copy of store_in_host(), so the device is synchronized
         * before their host data is copied. The copy constructors copy device
         * data on the stream of the source object, so the copies are waited
         * for before the operation uses them on its own stream. Called
         * without the GIL.
         */
        template <typename T>
        std::vector<T> collect(const std::vector<const T*>& objects)
        {
            bool on_host = false;
            for (const T* object : objects)
            {
                on_host = on_host || !object->is_on_device();
            }
            if (on_host)
            {
                HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
            }
            std::vector<T> result;
            result.reserve(objects.size());
            for (const T* object : objects)
            {
                result.push_back(*object);
            }
            if constexpr (std::is_same_v<T, Ciphertext>)
            {
                for (const T* object : objects)
                {
                    HEONGPU_PY_CUDA_CHECK(
                        cudaStreamSynchronize(object->stream()));
                }
            }
            else
            {
                // Waits for the copies of the keys on the whole device.
                HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
            }
            return result;
        }

        /**
         * @brief Waits for an output stored in host memory: HEonGPU copies it
         * to host memory asynchronously, and the returned object may be read
         * on the host right away (for example by to_bytes()). Called without
         * the GIL.
         */
        void finish(const heongpu::ExecutionOptions& options)
        {
            if (options.storage_ == heongpu::storage_type::host)
            {
                HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
            }
        }

        /**
         * @brief Raises TypeError if a list item is None.
         */
        template <typename T>
        void check_items(const std::vector<const T*>& objects, const char* name)
        {
            for (const T* object : objects)
            {
                if (object == nullptr)
                {
                    throw nb::type_error(
                        (std::string(name) + " must not contain None.")
                            .c_str());
                }
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
                         return "<RNGSeed key=" +
                                std::to_string(self.key_.size()) +
                                " bytes nonce=" +
                                std::to_string(self.nonce_.size()) + " bytes>";
                     });
            def_copy<RNGSeed>(seed);
        }

        /**
         * @brief Adds the serialization methods of a multiparty key. The
         * data is the HEonGPU format of the base key (save() and to_bytes()
         * are inherited); the seed is not part of it, so loading needs the
         * context and, optionally, the seed. Pickling is not supported
         * because a key cannot be created without its context.
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

    void bind_ckks_mpc(nb::module_& m)
    {
        bind_rng_seed(m);

        // --------------------------------------------------------------
        // Multiparty keys
        // --------------------------------------------------------------

        nb::class_<MultipartyPublicKey, PublicKey> public_key(
            m, "MultipartyPublicKey",
            "The public key share of one participant, generated with a "
            "common seed. MultiPartyManager.assemble_public_key_share() "
            "combines the shares into the collective PublicKey.");
        public_key
            .def(nb::init<ContextPtr, const RNGSeed>(), "context"_a, "seed"_a)
            .def_prop_ro("seed", &MultipartyPublicKey::seed,
                         "The common seed of the key.");
        def_multiparty_key_serialization<MultipartyPublicKey>(
            public_key, "MultipartyPublicKey");
        def_copy<MultipartyPublicKey>(public_key);

        nb::class_<MultipartyRelinKey, RelinKey> relin_key(
            m, "MultipartyRelinKey",
            "A relinearization key share or a partial aggregate in the "
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
            "The Galois key share of one participant, generated with a "
            "common seed. MultiPartyManager.assemble_galois_key_share() "
            "combines the shares into the collective GaloisKey.");
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
                "Key share for the given rotation steps, for the given Galois "
                "elements, or by default for the rotations by powers of two.")
            .def_prop_ro("seed", &MultipartyGaloisKey::seed,
                         "The common seed of the key.");
        def_multiparty_key_serialization<MultipartyGaloisKey>(
            galois_key, "MultipartyGaloisKey");
        def_copy<MultipartyGaloisKey>(galois_key);

        // --------------------------------------------------------------
        // Multiparty manager
        // --------------------------------------------------------------

        nb::class_<MultiPartyManager>(
            m, "MultiPartyManager",
            "Multiparty CKKS protocols: collective public, relinearization "
            "and Galois key generation from the participants' secret keys, "
            "collective (distributed) decryption from partial decryptions, "
            "and collective bootstrapping. Each participant and the server "
            "use their own manager; a participant must use the same manager "
            "for both rounds of the relinearization key generation.")
            .def(nb::init<ContextPtr, Encoder&, double>(), "context"_a,
                 "encoder"_a, "scale"_a,
                 "A manager for the context. scale is the scale of the "
                 "ciphertexts that collective bootstrapping returns.")
            .def_prop_ro("context", [](const MultiPartyManager& self)
                         { return self.context; })
            .def_prop_ro(
                "scale",
                [](const MultiPartyManager& self) { return self.scale; },
                "The scale of the ciphertexts collective bootstrapping "
                "returns.")

            // Public key
            .def(
                "generate_public_key_share",
                [](MultiPartyManager& self, const SecretKey& secret_key,
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
                [](MultiPartyManager& self,
                   const std::vector<const MultipartyPublicKey*>& shares,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    check_items(shares, "shares");
                    nb::gil_scoped_release release;
                    PublicKey key(self.context);
                    self.manager.assemble_public_key_share(collect(shares), key,
                                                           opts);
                    finish(opts);
                    return key;
                },
                "shares"_a, "options"_a = nb::none(),
                "Combines the public key shares of all participants into the "
                "collective public key.")

            // Relinearization key
            .def(
                "generate_relin_key_init",
                [](MultiPartyManager& self, const SecretKey& secret_key,
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
                [](MultiPartyManager& self,
                   const std::vector<const MultipartyRelinKey*>& shares,
                   const RNGSeed* seed, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    check_items(shares, "shares");
                    if (shares.empty())
                    {
                        throw nb::value_error(
                            "No participant to generate common relinkey!");
                    }
                    RNGSeed key_seed = seed ? *seed : shares[0]->seed();
                    nb::gil_scoped_release release;
                    MultipartyRelinKey key(self.context, key_seed);
                    self.manager.assemble_relin_key_init(collect(shares), key,
                                                         opts);
                    finish(opts);
                    return key;
                },
                "shares"_a, "seed"_a = nb::none(), "options"_a = nb::none(),
                "Combines the round 1 shares of all participants into the "
                "common round 1 key, which goes back to the participants. "
                "seed defaults to the seed of the first share.")
            .def(
                "generate_relin_key_share",
                [](MultiPartyManager& self, const MultipartyRelinKey& common,
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
                [](MultiPartyManager& self,
                   const std::vector<const MultipartyRelinKey*>& shares,
                   const MultipartyRelinKey& common, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    check_items(shares, "shares");
                    nb::gil_scoped_release release;
                    RelinKey key(self.context);
                    self.manager.assemble_relin_key_share(collect(shares),
                                                          common, key, opts);
                    finish(opts);
                    return key;
                },
                "shares"_a, "common"_a, "options"_a = nb::none(),
                "Combines the round 2 shares of all participants and the "
                "common round 1 key into the collective relinearization key.")

            // Galois key
            .def(
                "generate_galois_key_share",
                [](MultiPartyManager& self, const SecretKey& secret_key,
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
                "given rotation steps, for the given Galois elements, or by "
                "default for the rotations by powers of two. All "
                "participants must use the same seed and rotations.")
            .def(
                "assemble_galois_key_share",
                [](MultiPartyManager& self,
                   const std::vector<const MultipartyGaloisKey*>& shares,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    check_items(shares, "shares");
                    if (shares.empty())
                    {
                        throw nb::value_error(
                            "No participant to generate common galois!");
                    }
                    nb::gil_scoped_release release;
                    std::vector<MultipartyGaloisKey> all = collect(shares);
                    // The collective key has the rotations of the shares;
                    // its data is replaced by the aggregation.
                    GaloisKey key(static_cast<const GaloisKey&>(all[0]));
                    self.manager.assemble_galois_key_share(all, key, opts);
                    finish(opts);
                    return key;
                },
                "shares"_a, "options"_a = nb::none(),
                "Combines the Galois key shares of all participants into the "
                "collective Galois key, for the rotations of the shares.")

            // Decryption
            .def(
                "decrypt_partial",
                [](MultiPartyManager& self, const Ciphertext& ciphertext,
                   const SecretKey& secret_key, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    Ciphertext partial(self.context, opts);
                    self.manager.decrypt_partial(ciphertext, secret_key,
                                                 partial, opts);
                    finish(opts);
                    return partial;
                },
                "ciphertext"_a, "secret_key"_a, "options"_a = nb::none(),
                "The partial decryption of a ciphertext with one "
                "participant's secret key.")
            .def(
                "decrypt",
                [](MultiPartyManager& self,
                   const std::vector<const Ciphertext*>& partials,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    check_items(partials, "partials");
                    nb::gil_scoped_release release;
                    Plaintext plain(self.context, opts);
                    self.manager.decrypt(collect(partials), plain, opts);
                    finish(opts);
                    return plain;
                },
                "partials"_a, "options"_a = nb::none(),
                "Fuses the partial decryptions of all participants into the "
                "plaintext.")

            // Collective bootstrapping
            .def(
                "distributed_bootstrapping_participant",
                [](MultiPartyManager& self, const Ciphertext& common,
                   const SecretKey& secret_key, const RNGSeed& seed,
                   nb::handle options)
                {
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
                "ciphertext common, with its secret key and the common "
                "seed.")
            .def(
                "distributed_bootstrapping_coordinator",
                [](MultiPartyManager& self,
                   const std::vector<const Ciphertext*>& shares,
                   const Ciphertext& common, const RNGSeed& seed,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    check_items(shares, "shares");
                    nb::gil_scoped_release release;
                    Ciphertext output(self.context, opts);
                    self.manager.distributed_bootstrapping_coordinator(
                        collect(shares), common, output, seed, opts);
                    finish(opts);
                    return output;
                },
                "shares"_a, "common"_a, "seed"_a, "options"_a = nb::none(),
                "Combines the participants' shares into a refreshed "
                "encryption of common at the top level (depth 0), with the "
                "scale of the manager.");
    }

} // namespace heongpu_py
