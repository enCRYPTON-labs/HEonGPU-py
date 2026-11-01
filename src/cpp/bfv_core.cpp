// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"
#include "generated.h"

namespace heongpu_py
{
    namespace
    {
        constexpr auto BFV = heongpu::Scheme::BFV;

        using Context = heongpu::HEContextImpl<BFV>;
        using ContextPtr = heongpu::HEContext<BFV>;
        using Params = heongpu::HEParams<BFV>;
        using ParameterSet = heongpu::HEParameterSet<BFV>;
        using SecretKey = heongpu::Secretkey<BFV>;
        using PublicKey = heongpu::Publickey<BFV>;
        using RelinKey = heongpu::Relinkey<BFV>;
        using GaloisKey = heongpu::Galoiskey<BFV>;
        using SwitchKey = heongpu::Switchkey<BFV>;
        using Plaintext = heongpu::Plaintext<BFV>;
        using Ciphertext = heongpu::Ciphertext<BFV>;
        using Encoder = heongpu::HEEncoder<BFV>;
        using Encryptor = heongpu::HEEncryptor<BFV>;
        using Decryptor = heongpu::HEDecryptor<BFV>;
        using KeyGenerator = heongpu::HEKeyGenerator<BFV>;

        // HEEncoder<Scheme::BFV> has no public accessor for its context, and
        // the encoder needs the plaintext modulus to reduce the input values.
        // The pointer to the private member is formed in an explicit
        // instantiation, where access checking does not apply.
        struct EncoderContextTag
        {
            using type = ContextPtr Encoder::*;
            friend type member_pointer(EncoderContextTag);
        };

        template <typename Tag, typename Tag::type Member> struct MemberAccess
        {
            friend typename Tag::type member_pointer(Tag) { return Member; }
        };

        template struct MemberAccess<EncoderContextTag, &Encoder::context_>;

        const ContextPtr& encoder_context(const Encoder& encoder)
        {
            const ContextPtr& context =
                encoder.*member_pointer(EncoderContextTag{});
            if (!context)
            {
                throw heongpu::StateError("The encoder has no context.");
            }
            return context;
        }

        std::vector<std::uint64_t>
        modulus_values(const std::vector<Modulus64>& moduli)
        {
            std::vector<std::uint64_t> values;
            values.reserve(moduli.size());
            for (const auto& modulus : moduli)
            {
                values.push_back(modulus.value);
            }
            return values;
        }

        /**
         * @brief Converts integer values (a scalar, a sequence or an array of
         * any integer dtype, also Python ints of any size) to their residues
         * modulo the plaintext modulus. A scalar is repeated in every slot.
         */
        std::vector<std::uint64_t> plain_values(nb::handle values,
                                                std::uint64_t plain_modulus,
                                                int slot_count)
        {
            nb::module_ numpy = nb::module_::import_("numpy");
            nb::object array;
            try
            {
                array = numpy.attr("asarray")(values);
            }
            catch (nb::python_error& error)
            {
                throw nb::type_error(
                    (std::string("values must be an integer, a sequence or an "
                                 "array of integers: ") +
                     error.what())
                        .c_str());
            }
            int dimensions = nb::cast<int>(array.attr("ndim"));
            if (dimensions > 1)
            {
                throw nb::value_error("values must be one-dimensional.");
            }
            if (nb::cast<std::size_t>(array.attr("size")) == 0)
            {
                return {};
            }

            std::string kind =
                nb::cast<std::string>(array.attr("dtype").attr("kind"));
            nb::object reduced;
            if (kind == "b")
            {
                reduced = array.attr("astype")("uint64");
            }
            else if (kind == "u")
            {
                reduced = numpy.attr("remainder")(
                    array.attr("astype")("uint64"),
                    numpy.attr("uint64")(plain_modulus));
            }
            else if (kind == "i")
            {
                reduced =
                    numpy.attr("remainder")(array.attr("astype")("int64"),
                                            numpy.attr("int64")(plain_modulus));
            }
            else if (kind == "O")
            {
                // Python ints that do not fit a NumPy integer type.
                nb::object index =
                    nb::module_::import_("operator").attr("index");
                nb::object items = array.attr("reshape")(-1).attr("tolist")();
                nb::list residues;
                nb::int_ modulus(plain_modulus);
                for (nb::handle item : items)
                {
                    nb::object value;
                    try
                    {
                        value = index(item);
                    }
                    catch (nb::python_error&)
                    {
                        throw nb::type_error(
                            "values must be integers (got a non-integer "
                            "element).");
                    }
                    residues.append(nb::steal(
                        PyNumber_Remainder(value.ptr(), modulus.ptr())));
                }
                reduced = numpy.attr("asarray")(residues, "uint64");
            }
            else
            {
                throw nb::type_error(
                    ("values must be integers, got an array of dtype " +
                     nb::cast<std::string>(array.attr("dtype").attr("name")) +
                     "; convert it explicitly, for example with "
                     "numpy.rint(values).astype(numpy.int64).")
                        .c_str());
            }

            std::vector<std::uint64_t> message =
                to_uint64_vector(reduced, "values");
            if (dimensions == 0)
            {
                message.resize(slot_count, message[0]);
            }
            return message;
        }

        /**
         * @brief Returns the raw RNS data of a ciphertext or plaintext as a
         * NumPy array of uint64 of the given shape.
         */
        nb::ndarray<nb::numpy, std::uint64_t>
        raw_data(std::vector<Data64>&& data, std::vector<size_t> shape)
        {
            size_t total = 1;
            for (size_t dimension : shape)
            {
                total *= dimension;
            }
            if (total != data.size())
            {
                shape = {data.size()};
            }
            auto* owner_data =
                new std::vector<std::uint64_t>(data.begin(), data.end());
            data.clear();
            nb::capsule owner(
                owner_data, [](void* pointer) noexcept
                { delete static_cast<std::vector<std::uint64_t>*>(pointer); });
            return nb::ndarray<nb::numpy, std::uint64_t>(
                owner_data->data(), shape.size(), shape.data(), owner);
        }

        /**
         * @brief The decryptor reads the first two polynomials of a
         * ciphertext only, so a product must be relinearized before it is
         * decrypted.
         */
        void require_relinearized(const Ciphertext& cipher)
        {
            if (cipher.relinearization_required())
            {
                throw heongpu::StateError(
                    "The ciphertext must be relinearized before it is "
                    "decrypted.");
            }
        }

        std::string context_parameters(const Context& context)
        {
            std::ostringstream stream;
            context.print_parameters(stream);
            return stream.str();
        }
    } // namespace

    void bind_bfv_core(nb::module_& m)
    {
        // --------------------------------------------------------------
        // Context
        // --------------------------------------------------------------

        nb::class_<Context> context(
            m, "Context",
            "Encryption parameters and the precomputed tables of the BFV "
            "scheme. Set the ring size, the moduli and the plaintext modulus, "
            "then call generate(); or create it from Params.make_context().");
        context
            .def(nb::init<heongpu::sec_level_type>(),
                 "security_level"_a = heongpu::sec_level_type::sec128)
            .def("set_poly_modulus_degree", &Context::set_poly_modulus_degree,
                 "poly_modulus_degree"_a,
                 "Sets the ring size N (a power of two).")
            .def(
                "set_coeff_modulus_bit_sizes",
                [](Context& self, nb::handle q_bit_sizes,
                   nb::handle p_bit_sizes)
                {
                    self.set_coeff_modulus_bit_sizes(
                        to_int_vector(q_bit_sizes, "q_bit_sizes"),
                        to_int_vector(p_bit_sizes, "p_bit_sizes"));
                },
                "q_bit_sizes"_a, "p_bit_sizes"_a,
                "Sets the bit sizes of the ciphertext primes (Q) and of the "
                "special primes (P); HEonGPU chooses the primes.")
            .def(
                "set_coeff_modulus_values",
                [](Context& self, nb::handle q_primes, nb::handle p_primes)
                {
                    self.set_coeff_modulus_values(
                        to_uint64_vector(q_primes, "q_primes"),
                        to_uint64_vector(p_primes, "p_primes"));
                },
                "q_primes"_a, "p_primes"_a,
                "Sets the ciphertext primes (Q) and the special primes (P).")
            .def("set_coeff_modulus_default_values",
                 &Context::set_coeff_modulus_default_values,
                 "p_modulus_count"_a,
                 "Takes the default primes of the ring size and the security "
                 "level, the last p_modulus_count of them as special primes "
                 "(P). Needs the ring size.")
            .def("set_plain_modulus", &Context::set_plain_modulus,
                 "plain_modulus"_a,
                 "Sets the plaintext modulus t. Batching (slot encoding) "
                 "needs a prime t = 1 (mod 2N).")
            .def(
                "generate",
                [](Context& self,
                   std::optional<heongpu::MemoryPoolConfig> pool_config)
                {
                    nb::gil_scoped_release release;
                    if (pool_config)
                    {
                        self.generate(*pool_config);
                    }
                    else
                    {
                        self.generate();
                    }
                },
                "pool_config"_a = nb::none(),
                "Validates the parameters and computes the tables on the GPU. "
                "The first generated context sets up the memory pool, with "
                "pool_config if it is given.")
            .def_prop_ro("poly_modulus_degree",
                         &Context::get_poly_modulus_degree, "The ring size N.")
            .def_prop_ro("log_poly_modulus_degree",
                         &Context::get_log_poly_modulus_degree)
            .def_prop_ro("security_level", &Context::get_security_level)
            .def_prop_ro("ciphertext_modulus_count",
                         &Context::get_ciphertext_modulus_count,
                         "The number of ciphertext primes (Q).")
            .def_prop_ro("key_modulus_count", &Context::get_key_modulus_count,
                         "The number of ciphertext and special primes (Q and "
                         "P).")
            .def_prop_ro(
                "key_modulus",
                [](const Context& self)
                { return modulus_values(self.get_key_modulus()); },
                "The ciphertext primes followed by the special primes.")
            .def_prop_ro(
                "plain_modulus",
                [](const Context& self)
                { return std::uint64_t(self.get_plain_modulus().value); },
                "The plaintext modulus t.")
            .def_prop_ro(
                "slot_count",
                [](const Context& self)
                { return self.get_poly_modulus_degree(); },
                "The number of slots, N: a matrix of 2 rows of N/2 slots.")
            .def_prop_ro("context_id", &Context::context_id,
                         "Identity of the parameters; objects can only be "
                         "used with a context of the same identity.")
            .def(
                "print_parameters",
                [](const Context& self) {
                    nb::print(nb::str(context_parameters(self).c_str()),
                              nb::str(""));
                },
                "Prints the encryption parameters.")
            .def("__str__", &Context::to_string)
            .def("__repr__",
                 [](const Context& self)
                 {
                     std::ostringstream stream;
                     stream << "<heongpu.bfv.Context N="
                            << self.get_poly_modulus_degree()
                            << " Q=" << self.get_ciphertext_modulus_count()
                            << " P="
                            << self.get_key_modulus_count() -
                                   self.get_ciphertext_modulus_count()
                            << " t=" << self.get_plain_modulus().value
                            << " id=0x" << std::hex << self.context_id() << ">";
                     return stream.str();
                 })
            .def(
                "to_bytes",
                [](const Context& self)
                {
                    std::string data = save_to_string(self);
                    return nb::bytes(data.data(), data.size());
                },
                "Serializes the parameters to bytes in the HEonGPU format.")
            .def_static(
                "from_bytes",
                [](nb::bytes data)
                {
                    auto object = std::make_shared<Context>();
                    load_from_string(*object, bytes_to_string(data));
                    return object;
                },
                "data"_a,
                "Deserializes a context and generates it (the tables are "
                "computed again).")
            .def(
                "save",
                [](const Context& self, nb::handle file)
                { write_file(file, save_to_string(self)); },
                "file"_a,
                "Writes the parameters to a path or to a binary file object.")
            .def_static(
                "load",
                [](nb::handle file)
                {
                    auto object = std::make_shared<Context>();
                    load_from_string(*object, read_file(file));
                    return object;
                },
                "file"_a,
                "Reads a context from a path or from a binary file object.")
            .def("__getstate__",
                 [](const Context& self)
                 {
                     std::string data = save_to_string(self);
                     return nb::bytes(data.data(), data.size());
                 })
            .def("__setstate__",
                 [](Context& self, nb::bytes data)
                 {
                     new (&self) Context();
                     load_from_string(self, bytes_to_string(data));
                 });

        // --------------------------------------------------------------
        // Parameters
        // --------------------------------------------------------------

        nb::class_<ParameterSet>(
            m, "ParameterSet",
            "A complete set of BFV parameters, as chosen by "
            "Params.generate().")
            .def(nb::init<>())
            .def_rw("security_level", &ParameterSet::security_level)
            .def_rw("poly_modulus_degree", &ParameterSet::poly_modulus_degree)
            .def_rw("coeff_modulus", &ParameterSet::coeff_modulus,
                    "The ciphertext primes (Q).")
            .def_rw("special_modulus", &ParameterSet::special_modulus,
                    "The special primes (P).")
            .def_rw("plain_modulus", &ParameterSet::plain_modulus,
                    "The plaintext modulus t.")
            .def_prop_ro("total_bit_count", &ParameterSet::total_bit_count,
                         "The sum of the bit sizes of the Q and P primes.")
            .def_prop_ro("max_bit_count", &ParameterSet::max_bit_count,
                         "The largest total bit count of the security level "
                         "(0: no limit).")
            .def_prop_ro("digit_count", &ParameterSet::digit_count,
                         "The number of key switching digits.")
            .def("apply", &ParameterSet::apply, "context"_a,
                 "Sets the parameters on an ungenerated context.")
            .def("make_context", &ParameterSet::make_context,
                 "Returns a new context with these parameters, not generated "
                 "yet: call generate() on it.")
            .def("__str__", &ParameterSet::to_string)
            .def("__repr__", &ParameterSet::to_string);

        nb::class_<Params>(
            m, "Params",
            "Requirements from which HEonGPU chooses BFV parameters: the "
            "plaintext modulus (or its bit size) and the multiplicative "
            "depth, or explicit moduli. Unset values are chosen by "
            "generate().")
            .def(
                "__init__",
                [](Params* self, int multiplicative_depth, Data64 plain_modulus,
                   int plain_modulus_bit_size,
                   heongpu::sec_level_type security_level,
                   std::size_t poly_modulus_degree, int special_modulus_count,
                   std::optional<std::vector<int>> coeff_modulus_bit_sizes,
                   std::optional<std::vector<int>> special_modulus_bit_sizes,
                   std::optional<std::vector<Data64>> coeff_modulus,
                   std::optional<std::vector<Data64>> special_modulus)
                {
                    new (self) Params();
                    self->multiplicative_depth = multiplicative_depth;
                    self->plain_modulus = plain_modulus;
                    self->plain_modulus_bit_size = plain_modulus_bit_size;
                    self->security_level = security_level;
                    self->poly_modulus_degree = poly_modulus_degree;
                    self->special_modulus_count = special_modulus_count;
                    if (coeff_modulus_bit_sizes)
                        self->coeff_modulus_bit_sizes =
                            *coeff_modulus_bit_sizes;
                    if (special_modulus_bit_sizes)
                        self->special_modulus_bit_sizes =
                            *special_modulus_bit_sizes;
                    if (coeff_modulus)
                        self->coeff_modulus = *coeff_modulus;
                    if (special_modulus)
                        self->special_modulus = *special_modulus;
                },
                nb::kw_only(), "multiplicative_depth"_a = 1,
                "plain_modulus"_a = 0, "plain_modulus_bit_size"_a = 20,
                "security_level"_a = heongpu::sec_level_type::sec128,
                "poly_modulus_degree"_a = 0, "special_modulus_count"_a = 0,
                "coeff_modulus_bit_sizes"_a = nb::none(),
                "special_modulus_bit_sizes"_a = nb::none(),
                "coeff_modulus"_a = nb::none(),
                "special_modulus"_a = nb::none())
            .def_rw("multiplicative_depth", &Params::multiplicative_depth,
                    "The number of multiplications in a row the computation "
                    "needs.")
            .def_rw("plain_modulus", &Params::plain_modulus,
                    "The plaintext modulus t; 0 takes the largest prime "
                    "t = 1 (mod 2N) of plain_modulus_bit_size bits.")
            .def_rw("plain_modulus_bit_size", &Params::plain_modulus_bit_size,
                    "The bit size of t when plain_modulus is 0.")
            .def_rw("security_level", &Params::security_level)
            .def_rw("poly_modulus_degree", &Params::poly_modulus_degree,
                    "The ring size; 0 chooses the smallest secure one.")
            .def_rw("special_modulus_count", &Params::special_modulus_count,
                    "The number of special primes; 0 chooses it.")
            .def_rw("coeff_modulus_bit_sizes", &Params::coeff_modulus_bit_sizes)
            .def_rw("special_modulus_bit_sizes",
                    &Params::special_modulus_bit_sizes)
            .def_rw("coeff_modulus", &Params::coeff_modulus)
            .def_rw("special_modulus", &Params::special_modulus)
            .def("generate", &Params::generate,
                 "Chooses the parameters. Raises ParameterError if the "
                 "requirements cannot be met.")
            .def("make_context", &Params::make_context,
                 "Chooses the parameters and returns a new context with them, "
                 "not generated yet: call generate() on it.");

        // --------------------------------------------------------------
        // Keys
        // --------------------------------------------------------------

        nb::class_<SecretKey> secret_key(m, "SecretKey", "The secret key.");
        secret_key.def(nb::init<ContextPtr>(), "context"_a)
            .def(nb::init<ContextPtr, int>(), "context"_a, "hamming_weight"_a,
                 "A sparse secret key with the given number of nonzero "
                 "coefficients.")
            .def_static(
                "from_coefficients",
                [](nb::handle coefficients, ContextPtr context,
                   nb::handle stream)
                {
                    std::vector<int> values =
                        to_int_vector(coefficients, "coefficients");
                    cudaStream_t s = to_stream(stream);
                    nb::gil_scoped_release release;
                    return SecretKey(values, std::move(context), s);
                },
                "coefficients"_a, "context"_a, "stream"_a = nb::none(),
                "A secret key with the given ternary coefficients.")
            .def_prop_ro("ring_size", &SecretKey::ring_size)
            .def_prop_ro("coeff_modulus_count", &SecretKey::coeff_modulus_count)
            .def("set_context", &SecretKey::set_context, "context"_a,
                 "Attaches a context to a loaded key.");
        def_storage<SecretKey>(secret_key);
        def_stream<SecretKey>(secret_key);
        def_serialization<SecretKey>(secret_key);
        def_copy<SecretKey>(secret_key);

        nb::class_<PublicKey> public_key(m, "PublicKey", "The public key.");
        public_key.def(nb::init<ContextPtr>(), "context"_a)
            .def_prop_ro("ring_size", &PublicKey::ring_size)
            .def_prop_ro("coeff_modulus_count", &PublicKey::coeff_modulus_count)
            .def("set_context", &PublicKey::set_context, "context"_a,
                 "Attaches a context to a loaded key.");
        def_storage<PublicKey>(public_key);
        def_stream<PublicKey>(public_key);
        def_serialization<PublicKey>(public_key);
        def_copy<PublicKey>(public_key);

        nb::class_<RelinKey> relin_key(m, "RelinKey",
                                       "The relinearization key.");
        relin_key.def(nb::init<ContextPtr>(), "context"_a)
            .def("set_context", &RelinKey::set_context, "context"_a,
                 "Attaches a context to a loaded key.");
        def_storage<RelinKey>(relin_key);
        def_serialization<RelinKey>(relin_key);
        def_copy<RelinKey>(relin_key);

        nb::class_<GaloisKey> galois_key(
            m, "GaloisKey",
            "Galois keys for row rotations and the column rotation. By "
            "default they cover the row rotations by the small powers of two "
            "(other steps are composed of them).");
        galois_key.def(nb::init<ContextPtr>(), "context"_a)
            .def(
                "__init__",
                [](GaloisKey* self, ContextPtr context,
                   std::optional<std::vector<int>> shifts,
                   std::optional<int> max_shift,
                   std::optional<std::vector<std::uint32_t>> galois_elements)
                {
                    int given = (shifts ? 1 : 0) + (max_shift ? 1 : 0) +
                                (galois_elements ? 1 : 0);
                    if (given > 1)
                    {
                        throw nb::value_error("Give only one of shifts, "
                                              "max_shift and galois_elements.");
                    }
                    if (shifts)
                    {
                        new (self) GaloisKey(std::move(context), *shifts);
                    }
                    else if (max_shift)
                    {
                        new (self) GaloisKey(std::move(context), *max_shift);
                    }
                    else if (galois_elements)
                    {
                        new (self)
                            GaloisKey(std::move(context), *galois_elements);
                    }
                    else
                    {
                        new (self) GaloisKey(std::move(context));
                    }
                },
                "context"_a, nb::kw_only(), "shifts"_a = nb::none(),
                "max_shift"_a = nb::none(), "galois_elements"_a = nb::none(),
                "Keys for the given row rotation steps, for the steps +-1, "
                "+-2, +-4, ..., +-2^max_shift, or for the given Galois "
                "elements. The column rotation is always included.")
            .def("set_context", &GaloisKey::set_context, "context"_a,
                 "Attaches a context to a loaded key.");
        def_storage<GaloisKey>(galois_key);
        def_serialization<GaloisKey>(galois_key);
        def_copy<GaloisKey>(galois_key);

        nb::class_<SwitchKey> switch_key(
            m, "SwitchKey",
            "A key switching key from one secret key to another.");
        switch_key.def(nb::init<ContextPtr>(), "context"_a)
            .def("set_context", &SwitchKey::set_context, "context"_a,
                 "Attaches a context to a loaded key.");
        def_storage<SwitchKey>(switch_key);
        def_serialization<SwitchKey>(switch_key);
        def_copy<SwitchKey>(switch_key);

        // --------------------------------------------------------------
        // Plaintext and ciphertext
        // --------------------------------------------------------------

        nb::class_<Plaintext> plaintext(
            m, "Plaintext",
            "An encoded BFV message: a polynomial with coefficients modulo "
            "the plaintext modulus.");
        plaintext.def(nb::init<>())
            .def(
                "__init__",
                [](Plaintext* self, ContextPtr context, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    new (self) Plaintext(std::move(context), opts);
                },
                "context"_a, "options"_a = nb::none())
            .def_prop_ro("size", &Plaintext::size,
                         "The number of coefficients (N, or N times the "
                         "number of Q primes in the NTT domain).")
            .def_prop_ro("in_ntt_domain", &Plaintext::in_ntt_domain)
            .def(
                "get_data",
                [](Plaintext& self, nb::handle stream)
                {
                    cudaStream_t s = to_stream(stream);
                    std::vector<Data64> data(self.size());
                    {
                        nb::gil_scoped_release release;
                        // Wait for the work on the stream of the object.
                        HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
                        if (self.is_on_device())
                        {
                            HEONGPU_PY_CUDA_CHECK(
                                cudaMemcpyAsync(data.data(), self.data(),
                                                data.size() * sizeof(Data64),
                                                cudaMemcpyDeviceToHost, s));
                            HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(s));
                        }
                        else
                        {
                            std::memcpy(data.data(), self.data(),
                                        data.size() * sizeof(Data64));
                        }
                    }
                    size_t size = data.size();
                    return raw_data(std::move(data), {size});
                },
                "stream"_a = nb::none(),
                "Returns a copy of the coefficients as a uint64 NumPy array.")
            .def("__repr__", &Plaintext::to_string);
        def_storage<Plaintext>(plaintext);
        def_serialization<Plaintext>(plaintext);
        def_copy<Plaintext>(plaintext);

        nb::class_<Ciphertext> ciphertext(m, "Ciphertext", "A BFV ciphertext.");
        ciphertext.def(nb::init<>())
            .def(
                "__init__",
                [](Ciphertext* self, ContextPtr context, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    new (self) Ciphertext(std::move(context), opts);
                },
                "context"_a, "options"_a = nb::none())
            .def_prop_ro("ring_size", &Ciphertext::ring_size)
            .def_prop_ro("coeff_modulus_count",
                         &Ciphertext::coeff_modulus_count,
                         "The number of ciphertext primes (Q).")
            .def_prop_ro("size", &Ciphertext::size,
                         "The number of polynomials held: 2, or 3 after a "
                         "multiplication (the memory of the third one is "
                         "kept after relinearization). relinearization_"
                         "required tells whether a relinearization is "
                         "pending.")
            .def_prop_ro("in_ntt_domain", &Ciphertext::in_ntt_domain)
            .def_prop_ro("relinearization_required",
                         &Ciphertext::relinearization_required)
            .def_prop_ro(
                "stream",
                [](const Ciphertext& self)
                { return from_stream(self.stream()); },
                "Handle of the CUDA stream of the data.")
            .def(
                "switch_stream",
                [](Ciphertext& self, nb::handle stream)
                { self.switch_stream(to_stream(stream)); },
                nb::arg("stream").none(),
                "Uses another stream for the data (None: the default "
                "stream).")
            .def(
                "get_data",
                [](Ciphertext& self, nb::handle stream)
                {
                    cudaStream_t s = to_stream(stream);
                    std::vector<Data64> data;
                    {
                        nb::gil_scoped_release release;
                        // Wait for the work on the stream of the object.
                        HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
                        self.get_data(data, s);
                        HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(s));
                    }
                    size_t polynomials = self.size();
                    size_t n = self.ring_size();
                    size_t moduli =
                        polynomials * n ? data.size() / (polynomials * n) : 0;
                    return raw_data(std::move(data), {polynomials, moduli, n});
                },
                "stream"_a = nb::none(),
                "Returns a copy of the raw RNS data as a uint64 NumPy array of "
                "shape (size, moduli, ring_size).")
            .def("__repr__", &Ciphertext::to_string);
        def_storage<Ciphertext>(ciphertext);
        def_serialization<Ciphertext>(ciphertext);
        def_copy<Ciphertext>(ciphertext);

        // --------------------------------------------------------------
        // Encoder, encryptor, decryptor and key generator
        // --------------------------------------------------------------

        nb::class_<Encoder>(
            m, "Encoder",
            "Batch encoder: encodes a vector of N integers modulo the "
            "plaintext modulus t into the slots of a plaintext, seen as a "
            "matrix of 2 rows of N/2 slots, and decodes it.")
            .def(nb::init<ContextPtr>(), "context"_a)
            .def_prop_ro("slot_count", &Encoder::slot_count,
                         "The number of slots, N.")
            .def_prop_ro(
                "row_size",
                [](const Encoder& self) { return self.slot_count() / 2; },
                "The number of slots of a row, N/2.")
            .def_prop_ro(
                "plain_modulus",
                [](const Encoder& self) {
                    return std::uint64_t(
                        encoder_context(self)->get_plain_modulus().value);
                },
                "The plaintext modulus t.")
            .def(
                "encode",
                [](Encoder& self, nb::handle values, nb::handle options)
                {
                    std::uint64_t t =
                        encoder_context(self)->get_plain_modulus().value;
                    std::vector<std::uint64_t> message =
                        plain_values(values, t, self.slot_count());
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.encode(message, opts);
                },
                "values"_a, "options"_a = nb::none(),
                "Encodes an integer (into every slot), a sequence or an array "
                "of at most slot_count integers; the other slots are 0. The "
                "values are taken modulo t, so negative values and values of "
                "any size are allowed.")
            .def(
                "decode",
                [](Encoder& self, const Plaintext& plain, bool is_signed,
                   nb::handle options) -> nb::object
                {
                    require_generated(plain);
                    heongpu::ExecutionOptions opts = to_options(options);
                    if (is_signed)
                    {
                        std::vector<std::int64_t> message;
                        {
                            nb::gil_scoped_release release;
                            self.decode(message, plain, opts);
                        }
                        return nb::cast(to_numpy(std::move(message)));
                    }
                    std::vector<std::uint64_t> message;
                    {
                        nb::gil_scoped_release release;
                        self.decode(message, plain, opts);
                    }
                    return nb::cast(to_numpy(std::move(message)));
                },
                "plain"_a, nb::kw_only(), "signed"_a = false,
                "options"_a = nb::none(),
                "Decodes the slot_count slots. Returns a uint64 NumPy array of "
                "values in [0, t), or with signed=True an int64 array of the "
                "centered values in (-t/2, t/2].");

        nb::class_<Encryptor>(m, "Encryptor",
                              "Encrypts plaintexts with a public key.")
            .def(
                "__init__",
                [](Encryptor* self, ContextPtr context,
                   const PublicKey& public_key)
                {
                    require_generated(public_key);
                    new (self) Encryptor(std::move(context), public_key);
                },
                "context"_a, "public_key"_a)
            .def(
                "encrypt",
                [](Encryptor& self, const Plaintext& plain, nb::handle options)
                {
                    require_generated(plain);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.encrypt(plain, opts);
                },
                "plain"_a, "options"_a = nb::none());

        nb::class_<Decryptor>(m, "Decryptor",
                              "Decrypts ciphertexts with the secret key.")
            .def(nb::init<ContextPtr, const SecretKey&>(), "context"_a,
                 "secret_key"_a)
            .def(
                "decrypt",
                [](Decryptor& self, const Ciphertext& cipher,
                   nb::handle options)
                {
                    require_relinearized(cipher);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.decrypt(cipher, opts);
                },
                "cipher"_a, "options"_a = nb::none(),
                "Decrypts a ciphertext of size 2. Raises StateError for a "
                "product that is not relinearized yet.")
            .def(
                "remainder_noise_budget",
                [](Decryptor& self, const Ciphertext& cipher,
                   nb::handle options)
                {
                    require_relinearized(cipher);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.remainder_noise_budget(cipher, opts);
                },
                "cipher"_a, "options"_a = nb::none(),
                "Returns the noise budget left in the ciphertext, in bits; "
                "it decrypts correctly while the budget is above 0.");

        nb::class_<KeyGenerator>(m, "KeyGenerator", "Generates keys.")
            .def(nb::init<ContextPtr>(), "context"_a)
            .def(
                "secret_key",
                [](KeyGenerator& self, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.secret_key(opts);
                },
                "options"_a = nb::none(),
                "Generates a new secret key. For a sparse key, create "
                "SecretKey(context, hamming_weight) and call "
                "generate_secret_key().")
            .def(
                "public_key",
                [](KeyGenerator& self, const SecretKey& sk, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.public_key(sk, opts);
                },
                "secret_key"_a, "options"_a = nb::none())
            .def(
                "relin_key",
                [](KeyGenerator& self, const SecretKey& sk, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.relin_key(sk, opts);
                },
                "secret_key"_a, "options"_a = nb::none())
            .def(
                "galois_key",
                [](KeyGenerator& self, const SecretKey& sk,
                   std::optional<std::vector<int>> shifts, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    if (shifts)
                    {
                        return self.galois_key(sk, *shifts, opts);
                    }
                    return self.galois_key(sk, opts);
                },
                "secret_key"_a, "shifts"_a = nb::none(),
                "options"_a = nb::none(),
                "Generates Galois keys for the given row rotation steps, or "
                "the default keys of GaloisKey(context); the column rotation "
                "is always included.")
            .def(
                "generate_secret_key",
                [](KeyGenerator& self, SecretKey& sk, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.generate_secret_key(sk, opts);
                },
                "secret_key"_a, "options"_a = nb::none(),
                "Generates into an existing key (for example a sparse one "
                "created with SecretKey(context, hamming_weight)).")
            .def(
                "generate_public_key",
                [](KeyGenerator& self, PublicKey& pk, const SecretKey& sk,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.generate_public_key(pk, sk, opts);
                },
                "public_key"_a, "secret_key"_a, "options"_a = nb::none())
            .def(
                "generate_relin_key",
                [](KeyGenerator& self, RelinKey& rk, const SecretKey& sk,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.generate_relin_key(rk, sk, opts);
                },
                "relin_key"_a, "secret_key"_a, "options"_a = nb::none())
            .def(
                "generate_galois_key",
                [](KeyGenerator& self, GaloisKey& gk, const SecretKey& sk,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.generate_galois_key(gk, sk, opts);
                },
                "galois_key"_a, "secret_key"_a, "options"_a = nb::none(),
                "Generates the keys for the steps or Galois elements the "
                "GaloisKey was created with.")
            .def(
                "generate_switch_key",
                [](KeyGenerator& self, SwitchKey& swk, const SecretKey& new_sk,
                   const SecretKey& old_sk, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.generate_switch_key(swk, new_sk, old_sk, opts);
                },
                "switch_key"_a, "new_secret_key"_a, "old_secret_key"_a,
                "options"_a = nb::none(),
                "Generates a key that switches ciphertexts of old_secret_key "
                "to new_secret_key.");
    }

} // namespace heongpu_py
