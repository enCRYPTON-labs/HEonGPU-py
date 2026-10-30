// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

namespace heongpu_py
{
    namespace
    {
        constexpr auto CKKS = heongpu::Scheme::CKKS;

        using Context = heongpu::HEContextImpl<CKKS>;
        using ContextPtr = heongpu::HEContext<CKKS>;
        using Params = heongpu::HEParams<CKKS>;
        using ParameterSet = heongpu::HEParameterSet<CKKS>;
        using SecretKey = heongpu::Secretkey<CKKS>;
        using PublicKey = heongpu::Publickey<CKKS>;
        using RelinKey = heongpu::Relinkey<CKKS>;
        using GaloisKey = heongpu::Galoiskey<CKKS>;
        using SwitchKey = heongpu::Switchkey<CKKS>;
        using Plaintext = heongpu::Plaintext<CKKS>;
        using Ciphertext = heongpu::Ciphertext<CKKS>;
        using Encoder = heongpu::HEEncoder<CKKS>;
        using Encryptor = heongpu::HEEncryptor<CKKS>;
        using Decryptor = heongpu::HEDecryptor<CKKS>;
        using KeyGenerator = heongpu::HEKeyGenerator<CKKS>;

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

        template <typename T> std::string to_text(const T& object)
        {
            std::ostringstream stream;
            stream << object;
            return stream.str();
        }

        /**
         * @brief Encodes the values into a plaintext. Without a scale the
         * context must be in the automatic rescale mode, which gives the
         * scale of the level. A scalar is encoded into every slot.
         */
        Plaintext encode_values(Encoder& encoder, nb::handle values,
                                std::optional<double> scale,
                                std::optional<int> depth,
                                heongpu::encoding type, nb::handle options)
        {
            heongpu::ExecutionOptions opts = to_options(options);
            bool scalar = PyFloat_Check(values.ptr()) ||
                          PyLong_Check(values.ptr()) ||
                          PyComplex_Check(values.ptr());
            if (scalar && scale && !depth && !PyComplex_Check(values.ptr()))
            {
                double value = nb::cast<double>(values);
                nb::gil_scoped_release release;
                Plaintext plain;
                encoder.encode(plain, value, *scale, opts, type);
                return plain;
            }

            auto encode = [&](auto message) -> Plaintext
            {
                if (scalar)
                {
                    message.resize(encoder.slot_count(), message[0]);
                }
                nb::gil_scoped_release release;
                if (scale && depth)
                {
                    return encoder.encode(message, *scale, *depth, opts, type);
                }
                if (scale)
                {
                    return encoder.encode(message, *scale, opts, type);
                }
                if (depth)
                {
                    return encoder.encode(message, *depth, opts, type);
                }
                return encoder.encode(message, opts, type);
            };

            if (is_complex(values))
            {
                return encode(to_complex_vector(values, "values"));
            }
            return encode(to_double_vector(values, "values"));
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
            auto* owner_data = new std::vector<std::uint64_t>(
                data.begin(), data.end());
            data.clear();
            nb::capsule owner(owner_data, [](void* pointer) noexcept
                              {
                                  delete static_cast<
                                      std::vector<std::uint64_t>*>(pointer);
                              });
            return nb::ndarray<nb::numpy, std::uint64_t>(
                owner_data->data(), shape.size(), shape.data(), owner);
        }
    } // namespace

    void bind_ckks_core(nb::module_& m)
    {
        // --------------------------------------------------------------
        // Context
        // --------------------------------------------------------------

        nb::class_<Context> context(
            m, "Context",
            "Encryption parameters and the precomputed tables of the CKKS "
            "scheme. Set the ring size and the moduli, then call generate(); "
            "or create it from Params.make_context().");
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
            .def("set_rescale_mode", &Context::set_rescale_mode, "mode"_a,
                 "scale"_a = 0.0, "top_level"_a = -1,
                 "Selects manual or automatic rescaling. In the automatic mode "
                 "scale is the scale of fresh ciphertexts and top_level the "
                 "level they are encrypted at (-1: the highest level).")
            .def_prop_ro("rescale_mode", &Context::get_rescale_mode)
            .def("get_level_scale", &Context::get_level_scale, "level"_a,
                 "Returns the scale of a level in the automatic rescale mode.")
            .def_prop_ro("top_level", &Context::get_top_level,
                         "The level of fresh ciphertexts in the automatic "
                         "rescale mode.")
            .def_prop_ro("poly_modulus_degree",
                         &Context::get_poly_modulus_degree,
                         "The ring size N.")
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
                "key_modulus", [](const Context& self)
                { return modulus_values(self.get_key_modulus()); },
                "The ciphertext primes followed by the special primes.")
            .def_prop_ro("slot_count", [](const Context& self)
                         { return self.get_poly_modulus_degree() / 2; })
            .def_prop_ro("context_id", &Context::context_id,
                         "Identity of the parameters; objects can only be "
                         "used with a context of the same identity.")
            .def("print_parameters",
                 [](const Context& self)
                 { nb::print(nb::str(self.to_string().c_str())); })
            .def("__str__", &Context::to_string)
            .def("__repr__",
                 [](const Context& self)
                 {
                     std::ostringstream stream;
                     stream << "<heongpu.ckks.Context N="
                            << self.get_poly_modulus_degree()
                            << " Q=" << self.get_ciphertext_modulus_count()
                            << " P="
                            << self.get_key_modulus_count() -
                                   self.get_ciphertext_modulus_count()
                            << " id=0x" << std::hex << self.context_id()
                            << ">";
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
                "save", [](const Context& self, nb::handle file)
                { write_file(file, save_to_string(self)); }, "file"_a,
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
                "Reads a context from a path or from a binary file object.");

        // --------------------------------------------------------------
        // Parameters
        // --------------------------------------------------------------

        nb::class_<ParameterSet>(
            m, "ParameterSet",
            "A complete set of CKKS parameters, as chosen by "
            "Params.generate().")
            .def(nb::init<>())
            .def_rw("security_level", &ParameterSet::security_level)
            .def_rw("poly_modulus_degree", &ParameterSet::poly_modulus_degree)
            .def_rw("coeff_modulus", &ParameterSet::coeff_modulus,
                    "The ciphertext primes (Q).")
            .def_rw("special_modulus", &ParameterSet::special_modulus,
                    "The special primes (P).")
            .def_rw("scale", &ParameterSet::scale)
            .def_rw("rescale_mode", &ParameterSet::rescale_mode)
            .def_prop_ro("multiplicative_depth",
                         &ParameterSet::multiplicative_depth)
            .def_prop_ro("total_bit_count", &ParameterSet::total_bit_count)
            .def_prop_ro("max_bit_count", &ParameterSet::max_bit_count,
                         "The largest total bit count of the security level.")
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
            "Requirements from which HEonGPU chooses CKKS parameters: the "
            "multiplicative depth and the scale, or explicit moduli. Unset "
            "values are chosen by generate().")
            .def(
                "__init__",
                [](Params* self, int multiplicative_depth, int scale_bit_size,
                   int first_modulus_bit_size,
                   heongpu::sec_level_type security_level,
                   std::size_t poly_modulus_degree, int special_modulus_count,
                   std::optional<heongpu::rescale_mode> rescale_mode,
                   std::optional<std::vector<int>> coeff_modulus_bit_sizes,
                   std::optional<std::vector<int>> special_modulus_bit_sizes,
                   std::optional<std::vector<Data64>> coeff_modulus,
                   std::optional<std::vector<Data64>> special_modulus)
                {
                    new (self) Params();
                    self->multiplicative_depth = multiplicative_depth;
                    self->scale_bit_size = scale_bit_size;
                    self->first_modulus_bit_size = first_modulus_bit_size;
                    self->security_level = security_level;
                    self->poly_modulus_degree = poly_modulus_degree;
                    self->special_modulus_count = special_modulus_count;
                    self->rescale_mode = rescale_mode;
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
                "scale_bit_size"_a = 40, "first_modulus_bit_size"_a = 60,
                "security_level"_a = heongpu::sec_level_type::sec128,
                "poly_modulus_degree"_a = 0, "special_modulus_count"_a = 0,
                "rescale_mode"_a = nb::none(),
                "coeff_modulus_bit_sizes"_a = nb::none(),
                "special_modulus_bit_sizes"_a = nb::none(),
                "coeff_modulus"_a = nb::none(),
                "special_modulus"_a = nb::none())
            .def_rw("multiplicative_depth", &Params::multiplicative_depth)
            .def_rw("scale_bit_size", &Params::scale_bit_size)
            .def_rw("first_modulus_bit_size", &Params::first_modulus_bit_size)
            .def_rw("security_level", &Params::security_level)
            .def_rw("poly_modulus_degree", &Params::poly_modulus_degree,
                    "The ring size; 0 chooses the smallest secure one.")
            .def_rw("special_modulus_count", &Params::special_modulus_count,
                    "The number of special primes; 0 chooses it.")
            .def_rw("rescale_mode", &Params::rescale_mode, nb::arg().none())
            .def_rw("coeff_modulus_bit_sizes",
                    &Params::coeff_modulus_bit_sizes)
            .def_rw("special_modulus_bit_sizes",
                    &Params::special_modulus_bit_sizes)
            .def_rw("coeff_modulus", &Params::coeff_modulus)
            .def_rw("special_modulus", &Params::special_modulus)
            .def("generate", &Params::generate,
                 "Chooses the parameters. Raises ParameterError if the "
                 "requirements cannot be met.")
            .def("make_context", &Params::make_context,
                 "Chooses the parameters and returns a new context with them, "
                 "not generated yet: call generate() on it.")
            .def("scale", &Params::scale, "Returns 2^scale_bit_size.");

        // --------------------------------------------------------------
        // Bootstrapping configuration
        // --------------------------------------------------------------

        nb::enum_<heongpu::SineType>(m, "SineType",
                                     "Approximation of the modular reduction "
                                     "in EvalMod.")
            .value("COS1", heongpu::SineType::cos1);

        nb::enum_<heongpu::PolyType>(m, "PolyType",
                                     "Polynomial basis of an approximation.")
            .value("MONOMIAL", heongpu::PolyType::monomial)
            .value("CHEBYSHEV", heongpu::PolyType::chebyshev);

        nb::enum_<heongpu::LinearTransformType>(
            m, "LinearTransformType",
            "The homomorphic encoding transforms of bootstrapping.")
            .value("COEFFS_TO_SLOTS",
                   heongpu::LinearTransformType::coeffs_to_slots)
            .value("SLOTS_TO_COEFFS",
                   heongpu::LinearTransformType::slots_to_coeffs);

        nb::class_<heongpu::EncodingMatrixConfig>(
            m, "EncodingMatrixConfig",
            "Configuration of CoeffToSlot or SlotToCoeff: the level it "
            "starts at, the baby-step giant-step ratio and the number of "
            "pieces (levels) the transform is split into.")
            .def(nb::init<>())
            .def(nb::init<heongpu::LinearTransformType, int>(), "lt_type"_a,
                 "level_start"_a)
            .def(nb::init<heongpu::LinearTransformType, int, double>(),
                 "lt_type"_a, "level_start"_a, "bsgs_ratio"_a)
            .def(nb::init<heongpu::LinearTransformType, int, double, int>(),
                 "lt_type"_a, "level_start"_a, "bsgs_ratio"_a, "piece"_a)
            .def_rw("lt_type", &heongpu::EncodingMatrixConfig::lt_type_)
            .def_rw("level_start", &heongpu::EncodingMatrixConfig::level_start_)
            .def_rw("bsgs_ratio", &heongpu::EncodingMatrixConfig::bsgs_ratio_)
            .def_rw("piece", &heongpu::EncodingMatrixConfig::piece_);

        nb::class_<heongpu::EvalModConfig>(
            m, "EvalModConfig",
            "Configuration of EvalMod, the approximate modular reduction of "
            "bootstrapping.")
            .def(nb::init<>())
            .def(nb::init<int>(), "level_start"_a,
                 "The default configuration starting at level_start.")
            .def(nb::init<Data64, int, double, int, int, int, int, double>(),
                 "q"_a, "level_start"_a, "message_ratio"_a, "k"_a,
                 "sine_degree"_a, "double_angle"_a, "arcsine_degree"_a,
                 "scaling_factor"_a)
            .def_rw("q", &heongpu::EvalModConfig::Q_)
            .def_rw("level_start", &heongpu::EvalModConfig::level_start_)
            .def_rw("message_ratio", &heongpu::EvalModConfig::message_ratio_)
            .def_rw("k", &heongpu::EvalModConfig::K_)
            .def_rw("sine_degree", &heongpu::EvalModConfig::sine_deg_)
            .def_rw("double_angle", &heongpu::EvalModConfig::double_angle_)
            .def_rw("arcsine_degree", &heongpu::EvalModConfig::arcsine_deg_)
            .def_rw("scaling_factor", &heongpu::EvalModConfig::scaling_factor_)
            .def_rw("piece", &heongpu::EvalModConfig::piece_)
            .def_rw("sine_type", &heongpu::EvalModConfig::sine_type_)
            .def_rw("poly_type", &heongpu::EvalModConfig::poly_type_);

        nb::class_<heongpu::BootstrappingConfig>(
            m, "BootstrappingConfig",
            "Configuration of bootstrapping: SlotToCoeff, EvalMod and "
            "CoeffToSlot.")
            .def(nb::init<heongpu::EncodingMatrixConfig, heongpu::EvalModConfig,
                          heongpu::EncodingMatrixConfig>(),
                 "stc_config"_a, "eval_mod_config"_a, "cts_config"_a)
            .def_rw("cts_config", &heongpu::BootstrappingConfig::cts_config_)
            .def_rw("stc_config", &heongpu::BootstrappingConfig::stc_config_)
            .def_rw("eval_mod_config",
                    &heongpu::BootstrappingConfig::eval_mod_config_)
            .def_rw("cts_piece", &heongpu::BootstrappingConfig::CtoS_piece_)
            .def_rw("stc_piece", &heongpu::BootstrappingConfig::StoC_piece_);

        nb::class_<heongpu::BootstrappingLevels>(
            m, "BootstrappingLevels",
            "The levels bootstrapping uses and leaves.")
            .def(nb::init<>())
            .def_rw("coeff_to_slot",
                    &heongpu::BootstrappingLevels::coeff_to_slot,
                    "Levels used by CoeffToSlot.")
            .def_rw("eval_mod", &heongpu::BootstrappingLevels::eval_mod,
                    "Levels used by EvalMod.")
            .def_rw("slot_to_coeff",
                    &heongpu::BootstrappingLevels::slot_to_coeff,
                    "Levels used by SlotToCoeff.")
            .def_rw("input_level", &heongpu::BootstrappingLevels::input_level,
                    "The level the input is brought to.")
            .def_rw("output_level", &heongpu::BootstrappingLevels::output_level,
                    "The level of the result.")
            .def_prop_ro("depth", &heongpu::BootstrappingLevels::depth,
                         "The levels bootstrapping consumes.")
            .def_prop_ro("usable_levels",
                         &heongpu::BootstrappingLevels::usable_levels,
                         "The levels left for computation after "
                         "bootstrapping.")
            .def("__repr__",
                 [](const heongpu::BootstrappingLevels& self)
                 {
                     std::ostringstream stream;
                     stream << "BootstrappingLevels(coeff_to_slot="
                            << self.coeff_to_slot
                            << ", eval_mod=" << self.eval_mod
                            << ", slot_to_coeff=" << self.slot_to_coeff
                            << ", input_level=" << self.input_level
                            << ", output_level=" << self.output_level << ")";
                     return stream.str();
                 });

        m.def(
            "bootstrapping_levels",
            [](const heongpu::BootstrappingConfig& config,
               heongpu::arithmetic_bootstrapping_type type)
            { return heongpu::bootstrapping_levels(config, type); },
            "config"_a, "type"_a,
            "Returns the levels arithmetic bootstrapping with this "
            "configuration uses.");
        m.def(
            "bootstrapping_levels",
            [](const heongpu::BootstrappingConfig& config,
               heongpu::logic_bootstrapping_type type)
            { return heongpu::bootstrapping_levels(config, type); },
            "config"_a, "type"_a,
            "Returns the levels logic bootstrapping with this configuration "
            "uses.");

        using Preset = heongpu::CKKSBootstrappingPreset;
        nb::class_<Preset>(
            m, "BootstrappingPreset",
            "Complete parameters for bootstrapping: ring size, moduli, scale, "
            "secret key Hamming weights and bootstrapping configuration. "
            "Create one with regular(), slim(), bit() or gate(), optionally "
            "with the number of levels left for computation.")
            .def_static("regular", nb::overload_cast<>(&Preset::regular))
            .def_static("regular", nb::overload_cast<int>(&Preset::regular),
                        "usable_levels"_a)
            .def_static("slim", nb::overload_cast<>(&Preset::slim))
            .def_static("slim", nb::overload_cast<int>(&Preset::slim),
                        "usable_levels"_a)
            .def_static("bit", nb::overload_cast<>(&Preset::bit))
            .def_static("bit", nb::overload_cast<int>(&Preset::bit),
                        "usable_levels"_a)
            .def_static("gate", nb::overload_cast<>(&Preset::gate))
            .def_static("gate", nb::overload_cast<int>(&Preset::gate),
                        "usable_levels"_a)
            .def_rw("arithmetic_type", &Preset::arithmetic_type)
            .def_rw("logic_type", &Preset::logic_type)
            .def_rw("poly_modulus_degree", &Preset::poly_modulus_degree)
            .def_rw("coeff_modulus", &Preset::coeff_modulus)
            .def_rw("special_modulus", &Preset::special_modulus)
            .def_rw("coeff_modulus_bit_sizes", &Preset::coeff_modulus_bit_sizes)
            .def_rw("special_modulus_bit_sizes",
                    &Preset::special_modulus_bit_sizes)
            .def_rw("scale", &Preset::scale)
            .def_rw("secret_key_hamming_weight",
                    &Preset::secret_key_hamming_weight)
            .def_rw("sparse_secret_key_hamming_weight",
                    &Preset::sparse_secret_key_hamming_weight)
            .def_rw("config", &Preset::config)
            .def("apply", &Preset::apply, "context"_a,
                 "Sets the ring size and the moduli on an ungenerated "
                 "context.")
            .def("levels", &Preset::levels,
                 "Returns the levels bootstrapping uses and leaves.");

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
                    return SecretKey(
                        to_int_vector(coefficients, "coefficients"),
                        std::move(context), to_stream(stream));
                },
                "coefficients"_a, "context"_a, "stream"_a = nb::none(),
                "A secret key with the given ternary coefficients.")
            .def_prop_ro("ring_size", &SecretKey::ring_size)
            .def_prop_ro("coeff_modulus_count",
                         &SecretKey::coeff_modulus_count)
            .def("set_context", &SecretKey::set_context, "context"_a,
                 "Attaches a context to a loaded key.");
        def_storage<SecretKey>(secret_key);
        def_serialization<SecretKey>(secret_key);
        def_copy<SecretKey>(secret_key);

        nb::class_<PublicKey> public_key(m, "PublicKey", "The public key.");
        public_key.def(nb::init<ContextPtr>(), "context"_a)
            .def_prop_ro("ring_size", &PublicKey::ring_size)
            .def_prop_ro("coeff_modulus_count",
                         &PublicKey::coeff_modulus_count);
        def_storage<PublicKey>(public_key);
        def_serialization<PublicKey>(public_key);
        def_copy<PublicKey>(public_key);

        nb::class_<RelinKey> relin_key(m, "RelinKey",
                                       "The relinearization key.");
        relin_key.def(nb::init<ContextPtr>(), "context"_a);
        def_storage<RelinKey>(relin_key);
        def_serialization<RelinKey>(relin_key);
        def_copy<RelinKey>(relin_key);

        nb::class_<GaloisKey> galois_key(
            m, "GaloisKey",
            "Galois keys for rotations and conjugation. By default they "
            "cover the rotations by powers of two.");
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
                "Keys for the given rotation steps, for the powers of two up "
                "to max_shift, or for the given Galois elements.")
            .def("set_context", &GaloisKey::set_context, "context"_a,
                 "Attaches a context to a loaded key.");
        def_storage<GaloisKey>(galois_key);
        def_serialization<GaloisKey>(galois_key);
        def_copy<GaloisKey>(galois_key);

        nb::class_<SwitchKey> switch_key(
            m, "SwitchKey",
            "A key switching key from one secret key to another.");
        switch_key.def(nb::init<ContextPtr>(), "context"_a);
        def_storage<SwitchKey>(switch_key);
        def_serialization<SwitchKey>(switch_key);
        def_copy<SwitchKey>(switch_key);

        nb::class_<heongpu::CKKSBootstrappingKeys>(
            m, "BootstrappingKeys",
            "The keys bootstrapping needs, from "
            "KeyGenerator.bootstrapping_keys().")
            .def_rw("relin_key", &heongpu::CKKSBootstrappingKeys::relin_key)
            .def_rw("galois_key", &heongpu::CKKSBootstrappingKeys::galois_key)
            .def_rw("dense_to_sparse",
                    &heongpu::CKKSBootstrappingKeys::dense_to_sparse,
                    nb::arg().none(),
                    "Switch key to the sparse secret key (or None).")
            .def_rw("sparse_to_dense",
                    &heongpu::CKKSBootstrappingKeys::sparse_to_dense,
                    nb::arg().none(),
                    "Switch key back to the dense secret key (or None).");

        // --------------------------------------------------------------
        // Plaintext and ciphertext
        // --------------------------------------------------------------

        nb::class_<Plaintext> plaintext(m, "Plaintext",
                                        "An encoded CKKS message.");
        plaintext.def(nb::init<>())
            .def(
                "__init__",
                [](Plaintext* self, ContextPtr context, nb::handle options)
                { new (self) Plaintext(std::move(context), to_options(options)); },
                "context"_a, "options"_a = nb::none())
            .def_prop_ro("size", &Plaintext::size)
            .def_prop_ro("depth", &Plaintext::depth,
                         "The number of rescales (and modulus drops) applied.")
            .def_prop_ro("scale", &Plaintext::scale)
            .def_prop_ro("in_ntt_domain", &Plaintext::in_ntt_domain)
            .def_prop_ro("encoding", &Plaintext::encoding_type)
            .def(
                "get_data",
                [](Plaintext& self, nb::handle stream)
                {
                    cudaStream_t s = to_stream(stream);
                    std::vector<Data64> data(self.size());
                    {
                        nb::gil_scoped_release release;
                        bool on_device = self.is_on_device();
                        if (on_device)
                        {
                            HEONGPU_PY_CUDA_CHECK(cudaMemcpyAsync(
                                data.data(), self.data(),
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
                    return raw_data(std::move(data), {data.size()});
                },
                "stream"_a = nb::none(),
                "Returns a copy of the raw RNS data as a uint64 NumPy array.")
            .def("__repr__", &Plaintext::to_string);
        def_storage<Plaintext>(plaintext);
        def_serialization<Plaintext>(plaintext);
        def_copy<Plaintext>(plaintext);

        nb::class_<Ciphertext> ciphertext(m, "Ciphertext",
                                          "A CKKS ciphertext.");
        ciphertext.def(nb::init<>())
            .def(
                "__init__",
                [](Ciphertext* self, ContextPtr context, nb::handle options)
                {
                    new (self)
                        Ciphertext(std::move(context), to_options(options));
                },
                "context"_a, "options"_a = nb::none())
            .def_prop_ro("ring_size", &Ciphertext::ring_size)
            .def_prop_ro("coeff_modulus_count",
                         &Ciphertext::coeff_modulus_count)
            .def_prop_ro("size", &Ciphertext::size,
                         "The number of polynomials (2, or 3 before "
                         "relinearization).")
            .def_prop_ro("depth", &Ciphertext::depth,
                         "The number of rescales (and modulus drops) applied.")
            .def_prop_ro("level", &Ciphertext::level,
                         "The number of rescales left.")
            .def_prop_ro("scale", &Ciphertext::scale)
            .def_prop_ro("in_ntt_domain", &Ciphertext::in_ntt_domain)
            .def_prop_ro("encoding", &Ciphertext::encoding_type)
            .def_prop_ro("rescale_required", &Ciphertext::rescale_required)
            .def_prop_ro("relinearization_required",
                         &Ciphertext::relinearization_required)
            .def_prop_ro(
                "stream", [](const Ciphertext& self)
                { return from_stream(self.stream()); },
                "Handle of the CUDA stream of the data.")
            .def(
                "switch_stream",
                [](Ciphertext& self, nb::handle stream)
                { self.switch_stream(to_stream(stream)); },
                nb::arg("stream").none(),
                "Uses another stream (None: the default stream) for the "
                "data.")
            .def(
                "get_data",
                [](Ciphertext& self, nb::handle stream)
                {
                    cudaStream_t s = to_stream(stream);
                    std::vector<Data64> data;
                    {
                        nb::gil_scoped_release release;
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

        nb::class_<Encoder>(m, "Encoder",
                            "Encodes real or complex vectors into plaintexts "
                            "and decodes them.")
            .def(nb::init<ContextPtr>(), "context"_a)
            .def_prop_ro("slot_count", &Encoder::slot_count)
            .def("encode", &encode_values, "values"_a,
                 "scale"_a = nb::none(), nb::kw_only(), "depth"_a = nb::none(),
                 "encoding"_a = heongpu::encoding::slot,
                 "options"_a = nb::none(),
                 "Encodes a scalar (into every slot), a sequence or an array "
                 "of real or complex numbers. Without a scale, the context "
                 "must be in the automatic rescale mode. depth encodes for a "
                 "ciphertext at that depth.")
            .def(
                "decode",
                [](Encoder& self, const Plaintext& plain, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    std::vector<double> message;
                    {
                        nb::gil_scoped_release release;
                        message = self.decode(plain, opts);
                    }
                    return to_numpy(std::move(message));
                },
                "plain"_a, "options"_a = nb::none(),
                "Decodes the real parts as a float64 NumPy array.")
            .def(
                "decode_complex",
                [](Encoder& self, const Plaintext& plain, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    std::vector<Complex64> message;
                    {
                        nb::gil_scoped_release release;
                        self.decode(message, plain, opts);
                    }
                    return to_numpy(std::move(message));
                },
                "plain"_a, "options"_a = nb::none(),
                "Decodes as a complex128 NumPy array.");

        nb::class_<Encryptor>(m, "Encryptor",
                              "Encrypts plaintexts with a public key.")
            .def(nb::init<ContextPtr, const PublicKey&>(), "context"_a,
                 "public_key"_a)
            .def(
                "encrypt",
                [](Encryptor& self, const Plaintext& plain, nb::handle options)
                {
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
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.decrypt(cipher, opts);
                },
                "cipher"_a, "options"_a = nb::none());

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
                "Generates Galois keys for the given rotation steps, or for "
                "the powers of two.")
            .def(
                "bootstrapping_keys",
                [](KeyGenerator& self, const SecretKey& sk,
                   std::vector<int> galois_shifts,
                   int sparse_secret_key_hamming_weight, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.bootstrapping_keys(
                        sk, std::move(galois_shifts),
                        sparse_secret_key_hamming_weight, opts);
                },
                "secret_key"_a, "galois_shifts"_a,
                "sparse_secret_key_hamming_weight"_a = 0,
                "options"_a = nb::none(),
                "Generates the relinearization and Galois keys of "
                "bootstrapping (galois_shifts from "
                "ArithmeticOperator.bootstrapping_key_indices()), and the "
                "switch keys of a sparse secret key if its Hamming weight is "
                "given.")
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
                "galois_key"_a, "secret_key"_a, "options"_a = nb::none())
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
