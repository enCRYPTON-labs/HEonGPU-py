// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

#include <nanobind/stl/tuple.h>

#include <limits>
#include <type_traits>

namespace heongpu_py
{
    namespace
    {
        constexpr auto TFHE = heongpu::Scheme::TFHE;

        using Context = heongpu::HEContextImpl<TFHE>;
        using ContextPtr = heongpu::HEContext<TFHE>;
        using SecretKey = heongpu::Secretkey<TFHE>;
        using BootstrappingKey = heongpu::Bootstrappingkey<TFHE>;
        using Ciphertext = heongpu::Ciphertext<TFHE>;

        /**
         * @brief The key generator with its context, from which it creates
         * the keys it returns.
         */
        class KeyGenerator : public heongpu::HEKeyGenerator<TFHE>
        {
          public:
            explicit KeyGenerator(ContextPtr context)
                : heongpu::HEKeyGenerator<TFHE>(context),
                  context_(std::move(context))
            {
            }

            const ContextPtr& context() const noexcept { return context_; }

          private:
            ContextPtr context_;
        };
        using Encryptor = heongpu::HEEncryptor<TFHE>;
        using Decryptor = heongpu::HEDecryptor<TFHE>;
        using LogicOperator = heongpu::HELogicOperator<TFHE>;

        // --------------------------------------------------------------
        // Serialization format
        //
        // HEonGPU has no save()/load() for the TFHE classes yet, so the
        // Python package writes them in its own format:
        //   "HGPYTFHE" | uint32 version | uint32 kind | fields
        // Scalars are stored in the byte order of the machine, arrays as a
        // uint64 element count followed by the elements.
        // --------------------------------------------------------------

        constexpr char format_magic[8] = {'H', 'G', 'P', 'Y',
                                          'T', 'F', 'H', 'E'};
        constexpr std::uint32_t format_version = 1;

        enum class Kind : std::uint32_t
        {
            secret_key = 1,
            ciphertext = 2,
            bootstrapping_key = 3
        };

        const char* kind_name(Kind kind)
        {
            switch (kind)
            {
                case Kind::secret_key:
                    return "SecretKey";
                case Kind::ciphertext:
                    return "Ciphertext";
                case Kind::bootstrapping_key:
                    return "BootstrappingKey";
            }
            return "object";
        }

        class Writer
        {
          public:
            explicit Writer(Kind kind)
            {
                data_.append(format_magic, sizeof(format_magic));
                value(format_version);
                value(static_cast<std::uint32_t>(kind));
            }

            template <typename T> void value(const T& value)
            {
                data_.append(reinterpret_cast<const char*>(&value), sizeof(T));
            }

            template <typename T, typename Allocator>
            void array(const std::vector<T, Allocator>& values)
            {
                value(static_cast<std::uint64_t>(values.size()));
                data_.append(reinterpret_cast<const char*>(values.data()),
                             values.size() * sizeof(T));
            }

            std::string take() { return std::move(data_); }

          private:
            std::string data_;
        };

        class Reader
        {
          public:
            Reader(const std::string& data, Kind kind)
                : position_(data.data()), end_(data.data() + data.size()),
                  kind_(kind)
            {
                char magic[sizeof(format_magic)];
                if (remaining() < sizeof(magic))
                {
                    fail("the data is too short");
                }
                std::memcpy(magic, position_, sizeof(magic));
                position_ += sizeof(magic);
                if (std::memcmp(magic, format_magic, sizeof(magic)) != 0)
                {
                    fail("the data is not in the HEonGPU-py TFHE format");
                }
                std::uint32_t version = value<std::uint32_t>();
                if (version != format_version)
                {
                    fail("unsupported format version " +
                         std::to_string(version));
                }
                std::uint32_t stored = value<std::uint32_t>();
                if (stored != static_cast<std::uint32_t>(kind))
                {
                    fail(std::string("the data holds a ") +
                         kind_name(static_cast<Kind>(stored)));
                }
            }

            template <typename T> T value()
            {
                T result;
                if (remaining() < sizeof(T))
                {
                    fail("the data is truncated");
                }
                std::memcpy(&result, position_, sizeof(T));
                position_ += sizeof(T);
                return result;
            }

            template <typename T> std::vector<T> array(std::size_t expected)
            {
                std::uint64_t count = value<std::uint64_t>();
                if (count != expected)
                {
                    fail("an array has " + std::to_string(count) +
                         " elements instead of " + std::to_string(expected));
                }
                if (remaining() / sizeof(T) < count)
                {
                    fail("the data is truncated");
                }
                std::vector<T> result(count);
                std::memcpy(result.data(), position_, count * sizeof(T));
                position_ += count * sizeof(T);
                return result;
            }

            void finish()
            {
                if (position_ != end_)
                {
                    fail("unexpected data after the end");
                }
            }

            [[noreturn]] void fail(const std::string& reason) const
            {
                throw heongpu::SerializationError(
                    std::string("Cannot load the TFHE ") + kind_name(kind_) +
                    ": " + reason + ".");
            }

          private:
            std::size_t remaining() const { return end_ - position_; }

            const char* position_;
            const char* end_;
            Kind kind_;
        };

        template <typename T>
        std::vector<T> device_to_vector(const T* data, std::size_t size,
                                        cudaStream_t stream)
        {
            std::vector<T> result(size);
            if (size != 0)
            {
                HEONGPU_PY_CUDA_CHECK(
                    cudaMemcpyAsync(result.data(), data, size * sizeof(T),
                                    cudaMemcpyDeviceToHost, stream));
            }
            HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(stream));
            return result;
        }

        template <typename T>
        std::vector<T> stored_data(bool on_device,
                                   const heongpu::DeviceVector<T>& device,
                                   const heongpu::HostVector<T>& host)
        {
            if (on_device)
            {
                return device_to_vector(device.data(), device.size(),
                                        device.stream());
            }
            return std::vector<T>(host.begin(), host.end());
        }

        // --------------------------------------------------------------
        // Access to the state of the TFHE classes
        //
        // Serialization and the state checks of the bindings need members
        // that the TFHE classes keep private. The classes declare every
        // specialization of heongpu::input_storage_manager a friend; the
        // specializations below, for the State type of this file, are the
        // only code that uses this access.
        // --------------------------------------------------------------

        enum class Operation
        {
            save,
            load,
            is_generated,
            reset
        };

        struct State
        {
            Operation operation;
            Writer* writer = nullptr;
            Reader* reader = nullptr;
            bool* result = nullptr;
            const SecretKey* reference = nullptr;
        };

        template <typename T> void access(T& object, State state)
        {
            heongpu::input_storage_manager<T, State>(
                object, state, heongpu::ExecutionOptions(), false);
        }

        template <typename T> bool is_generated(const T& object)
        {
            bool result = false;
            State state{Operation::is_generated};
            state.result = &result;
            access(const_cast<T&>(object), state);
            return result;
        }
    } // namespace
} // namespace heongpu_py

namespace heongpu
{
    template <>
    void input_storage_manager<Ciphertext<Scheme::TFHE>, heongpu_py::State>(
        Ciphertext<Scheme::TFHE>& object, heongpu_py::State state,
        ExecutionOptions, bool)
    {
        using heongpu_py::Operation;
        switch (state.operation)
        {
            case Operation::is_generated:
                *state.result = object.ciphertext_generated_;
                return;
            case Operation::reset:
                // The context constructor leaves the size uninitialized.
                object.shape_ = 0;
                object.variances_.clear();
                object.ciphertext_generated_ = false;
                return;
            case Operation::save:
            {
                if (!object.ciphertext_generated_)
                {
                    throw StateError("Ciphertext is not generated! Encrypt "
                                     "it before saving it.");
                }
                bool on_device = object.is_on_device();
                auto& writer = *state.writer;
                writer.value<std::int32_t>(object.n_);
                writer.value<double>(object.alpha_min_);
                writer.value<double>(object.alpha_max_);
                writer.value<std::int32_t>(object.shape_);
                writer.value<std::uint8_t>(on_device ? 1 : 0);
                writer.array(object.variances_);
                writer.array(heongpu_py::stored_data(on_device,
                                                     object.a_device_location_,
                                                     object.a_host_location_));
                writer.array(heongpu_py::stored_data(on_device,
                                                     object.b_device_location_,
                                                     object.b_host_location_));
                return;
            }
            case Operation::load:
            {
                auto& reader = *state.reader;
                std::int32_t n = reader.value<std::int32_t>();
                if (n != object.n_)
                {
                    throw ContextMismatchError(
                        "Ciphertext context mismatch! The data has n=" +
                        std::to_string(n) + ", the context has n=" +
                        std::to_string(object.n_) + ".");
                }
                double alpha_min = reader.value<double>();
                double alpha_max = reader.value<double>();
                std::int32_t shape = reader.value<std::int32_t>();
                if (shape < 0 ||
                    (n != 0 && shape > std::numeric_limits<int>::max() / n))
                {
                    reader.fail("the size is invalid");
                }
                bool on_device = reader.value<std::uint8_t>() != 0;
                auto variances = reader.array<double>(shape);
                auto a = reader.array<std::int32_t>(
                    static_cast<std::size_t>(shape) * n);
                auto b = reader.array<std::int32_t>(shape);
                reader.finish();

                cudaStream_t stream = cudaStreamDefault;
                object.alpha_min_ = alpha_min;
                object.alpha_max_ = alpha_max;
                object.shape_ = shape;
                object.variances_ = std::move(variances);
                object.a_device_location_ = DeviceVector<int32_t>(a, stream);
                object.b_device_location_ = DeviceVector<int32_t>(b, stream);
                HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(stream));
                object.storage_type_ = storage_type::device;
                object.ciphertext_generated_ = true;
                if (!on_device)
                {
                    object.store_in_host(stream);
                }
                return;
            }
        }
    }

    template <>
    void input_storage_manager<Secretkey<Scheme::TFHE>, heongpu_py::State>(
        Secretkey<Scheme::TFHE>& object, heongpu_py::State state,
        ExecutionOptions, bool)
    {
        using heongpu_py::Operation;
        switch (state.operation)
        {
            case Operation::is_generated:
                *state.result = object.secret_key_generated_;
                return;
            case Operation::reset:
                return;
            case Operation::save:
            {
                if (!object.secret_key_generated_)
                {
                    throw StateError("Secretkey is not generated! Call "
                                     "generate_secret_key first.");
                }
                bool on_device = object.is_on_device();
                auto& writer = *state.writer;
                writer.value<std::int32_t>(object.n_);
                writer.value<std::int32_t>(object.N_);
                writer.value<std::int32_t>(object.k_);
                writer.value<std::int32_t>(object.bk_l_);
                writer.value<std::int32_t>(object.bk_bg_bit_);
                writer.value<double>(object.lwe_alpha_min);
                writer.value<double>(object.tlwe_alpha_min);
                writer.value<std::uint8_t>(on_device ? 1 : 0);
                writer.array(heongpu_py::stored_data(
                    on_device, object.lwe_key_device_location_,
                    object.lwe_key_host_location_));
                writer.array(heongpu_py::stored_data(
                    on_device, object.tlwe_key_device_location_,
                    object.tlwe_key_host_location_));
                return;
            }
            case Operation::load:
            {
                auto& reader = *state.reader;
                std::int32_t n = reader.value<std::int32_t>();
                std::int32_t N = reader.value<std::int32_t>();
                std::int32_t k = reader.value<std::int32_t>();
                std::int32_t bk_l = reader.value<std::int32_t>();
                std::int32_t bk_bg_bit = reader.value<std::int32_t>();
                double lwe_alpha_min = reader.value<double>();
                double tlwe_alpha_min = reader.value<double>();
                if (n != object.n_ || N != object.N_ || k != object.k_ ||
                    bk_l != object.bk_l_ || bk_bg_bit != object.bk_bg_bit_ ||
                    lwe_alpha_min != object.lwe_alpha_min ||
                    tlwe_alpha_min != object.tlwe_alpha_min)
                {
                    throw ContextMismatchError(
                        "Secretkey context mismatch! The data has n=" +
                        std::to_string(n) + ", N=" + std::to_string(N) +
                        ", k=" + std::to_string(k) +
                        ", the context has n=" + std::to_string(object.n_) +
                        ", N=" + std::to_string(object.N_) +
                        ", k=" + std::to_string(object.k_) +
                        " (the other parameters must match as well).");
                }
                bool on_device = reader.value<std::uint8_t>() != 0;
                auto lwe = reader.array<std::int32_t>(n);
                auto tlwe =
                    reader.array<std::int32_t>(static_cast<std::size_t>(k) * N);
                reader.finish();
                for (const auto* key : {&lwe, &tlwe})
                {
                    for (std::int32_t value : *key)
                    {
                        if (value != 0 && value != 1)
                        {
                            reader.fail("the key is not binary");
                        }
                    }
                }

                cudaStream_t stream = cudaStreamDefault;
                object.lwe_key_device_location_ =
                    DeviceVector<int32_t>(lwe, stream);
                object.tlwe_key_device_location_ =
                    DeviceVector<int32_t>(tlwe, stream);
                HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(stream));
                object.storage_type_ = storage_type::device;
                object.secret_key_generated_ = true;
                if (!on_device)
                {
                    object.store_in_host(stream);
                }
                return;
            }
        }
    }

    template <>
    void
    input_storage_manager<Bootstrappingkey<Scheme::TFHE>, heongpu_py::State>(
        Bootstrappingkey<Scheme::TFHE>& object, heongpu_py::State state,
        ExecutionOptions, bool)
    {
        using heongpu_py::Operation;
        switch (state.operation)
        {
            case Operation::is_generated:
                *state.result = object.boot_key_generated_;
                return;
            case Operation::reset:
                return;
            case Operation::save:
            {
                if (!object.boot_key_generated_)
                {
                    throw StateError("Bootstrappingkey is not generated! Call "
                                     "generate_bootstrapping_key first.");
                }
                bool on_device = object.is_on_device();
                auto& writer = *state.writer;
                writer.value<std::int32_t>(object.bk_k_);
                writer.value<std::int32_t>(object.bk_base_bit_);
                writer.value<std::int32_t>(object.bk_length_);
                writer.value<double>(object.bk_stdev_);
                writer.value<std::int32_t>(object.ks_base_bit_);
                writer.value<std::int32_t>(object.ks_length_);
                writer.value<std::uint8_t>(on_device ? 1 : 0);
                writer.array(heongpu_py::stored_data(
                    on_device, object.boot_key_device_location_,
                    object.boot_key_host_location_));
                writer.array(object.boot_key_variances_);
                writer.array(heongpu_py::stored_data(
                    on_device, object.switch_key_device_location_a_,
                    object.switch_key_host_location_a_));
                writer.array(heongpu_py::stored_data(
                    on_device, object.switch_key_device_location_b_,
                    object.switch_key_host_location_b_));
                writer.array(object.switch_key_variances_);
                return;
            }
            case Operation::load:
            {
                auto& reader = *state.reader;
                std::int32_t bk_k = reader.value<std::int32_t>();
                std::int32_t bk_base_bit = reader.value<std::int32_t>();
                std::int32_t bk_length = reader.value<std::int32_t>();
                double bk_stdev = reader.value<double>();
                std::int32_t ks_base_bit = reader.value<std::int32_t>();
                std::int32_t ks_length = reader.value<std::int32_t>();
                if (bk_k != object.bk_k_ ||
                    bk_base_bit != object.bk_base_bit_ ||
                    bk_length != object.bk_length_ ||
                    bk_stdev != object.bk_stdev_ ||
                    ks_base_bit != object.ks_base_bit_ ||
                    ks_length != object.ks_length_)
                {
                    throw ContextMismatchError(
                        "Bootstrappingkey context mismatch! The parameters of "
                        "the data differ from those of the context.");
                }
                // The LWE dimension n and the ring size N are not stored in
                // the key; a secret key of the same context has them.
                const auto& reference = *state.reference;
                std::size_t n = reference.lwe_size();
                std::size_t N = reference.tlwe_size() / bk_k;
                std::size_t rows = static_cast<std::size_t>(bk_k + 1);
                std::size_t ks_N = static_cast<std::size_t>(bk_k) * N;
                std::size_t noise_size =
                    ks_N * ks_length * ((std::size_t(1) << ks_base_bit) - 1);

                bool on_device = reader.value<std::uint8_t>() != 0;
                auto boot_key =
                    reader.array<Data64>(n * rows * bk_length * rows * N);
                auto boot_key_variances =
                    reader.array<double>(n * rows * bk_length);
                auto switch_key_a = reader.array<std::int32_t>(noise_size * n);
                auto switch_key_b = reader.array<std::int32_t>(noise_size);
                auto switch_key_variances = reader.array<double>(noise_size);
                reader.finish();

                cudaStream_t stream = cudaStreamDefault;
                object.boot_key_device_location_ =
                    DeviceVector<Data64>(boot_key, stream);
                object.switch_key_device_location_a_ =
                    DeviceVector<int32_t>(switch_key_a, stream);
                object.switch_key_device_location_b_ =
                    DeviceVector<int32_t>(switch_key_b, stream);
                HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(stream));
                object.boot_key_variances_ = std::move(boot_key_variances);
                object.switch_key_variances_ = std::move(switch_key_variances);
                object.storage_type_ = storage_type::device;
                object.boot_key_generated_ = true;
                if (!on_device)
                {
                    object.store_in_host(stream);
                }
                return;
            }
        }
    }
} // namespace heongpu

namespace heongpu_py
{
    namespace
    {
        // --------------------------------------------------------------
        // Serialization helpers
        // --------------------------------------------------------------

        template <typename T> constexpr Kind kind_of();
        template <> constexpr Kind kind_of<SecretKey>()
        {
            return Kind::secret_key;
        }
        template <> constexpr Kind kind_of<Ciphertext>()
        {
            return Kind::ciphertext;
        }
        template <> constexpr Kind kind_of<BootstrappingKey>()
        {
            return Kind::bootstrapping_key;
        }

        template <typename T> std::string save_state(const T& object)
        {
            nb::gil_scoped_release release;
            Writer writer(kind_of<T>());
            State state{Operation::save};
            state.writer = &writer;
            access(const_cast<T&>(object), state);
            return writer.take();
        }

        ContextPtr context_or_default(std::optional<ContextPtr> context)
        {
            if (context && *context)
            {
                return *context;
            }
            if (context)
            {
                throw heongpu::ParameterError(
                    "HEContext is not set! Pass a TFHE Context.");
            }
            return heongpu::GenHEContext<TFHE>();
        }

        template <typename T>
        T load_state(const std::string& data, std::optional<ContextPtr> context)
        {
            nb::gil_scoped_release release;
            ContextPtr ctx = context_or_default(std::move(context));
            Reader reader(data, kind_of<T>());
            State state{Operation::load};
            state.reader = &reader;
            if constexpr (std::is_same_v<T, BootstrappingKey>)
            {
                SecretKey reference(ctx);
                state.reference = &reference;
                T object(ctx);
                access(object, state);
                return object;
            }
            else
            {
                T object(ctx);
                access(object, state);
                return object;
            }
        }

        /**
         * @brief Adds to_bytes(), from_bytes(), save(), load() and pickle
         * support in the TFHE format of the Python package.
         */
        template <typename T, typename Class>
        void def_tfhe_serialization(Class& cls)
        {
            cls.def(
                   "to_bytes",
                   [](const T& self)
                   {
                       std::string data = save_state(self);
                       return nb::bytes(data.data(), data.size());
                   },
                   "Serializes the object to bytes. HEonGPU has no C++ "
                   "serialization for TFHE yet; the format is the one of "
                   "the Python package.")
                .def_static(
                    "from_bytes",
                    [](nb::bytes data, std::optional<ContextPtr> context)
                    { return load_state<T>(bytes_to_string(data), context); },
                    "data"_a, "context"_a = nb::none(),
                    "Deserializes an object written by to_bytes(). Without a "
                    "context, a new TFHE context is created (the TFHE "
                    "parameters are fixed).")
                .def(
                    "save",
                    [](const T& self, nb::handle file)
                    { write_file(file, save_state(self)); },
                    "file"_a,
                    "Writes the object to a path or to a binary file object.")
                .def_static(
                    "load",
                    [](nb::handle file, std::optional<ContextPtr> context)
                    { return load_state<T>(read_file(file), context); },
                    "file"_a, "context"_a = nb::none(),
                    "Reads an object from a path or from a binary file "
                    "object.")
                .def("__getstate__",
                     [](const T& self)
                     {
                         std::string data = save_state(self);
                         return nb::bytes(data.data(), data.size());
                     })
                .def("__setstate__",
                     [](T& self, nb::bytes data) {
                         new (&self) T(load_state<T>(bytes_to_string(data),
                                                     std::nullopt));
                     });
        }

        // --------------------------------------------------------------
        // Bits
        // --------------------------------------------------------------

        /**
         * @brief Converts a bool, a sequence or an array of booleans (or of
         * the integers 0 and 1) to bits.
         */
        std::vector<bool> to_bits(nb::handle values)
        {
            nb::module_ numpy = nb::module_::import_("numpy");
            nb::object array = numpy.attr("asarray")(values);
            int ndim = nb::cast<int>(array.attr("ndim"));
            if (ndim == 0)
            {
                array = array.attr("reshape")(1);
            }
            else if (ndim != 1)
            {
                throw nb::value_error("bits must be one-dimensional.");
            }
            std::size_t size = nb::cast<std::size_t>(array.attr("size"));
            if (size == 0)
            {
                throw heongpu::ParameterError("bits must not be empty.");
            }
            std::string kind =
                nb::cast<std::string>(array.attr("dtype").attr("kind"));
            if (kind == "i" || kind == "u")
            {
                nb::object valid =
                    numpy.attr("logical_or")(numpy.attr("equal")(array, 0),
                                             numpy.attr("equal")(array, 1));
                int all = PyObject_IsTrue(valid.attr("all")().ptr());
                if (all < 0)
                {
                    throw nb::python_error();
                }
                if (all == 0)
                {
                    throw nb::value_error(
                        "bits must be booleans or the integers 0 and 1.");
                }
            }
            else if (kind != "b")
            {
                throw nb::type_error(
                    "bits must be booleans or the integers 0 and 1.");
            }
            nb::object bytes =
                numpy.attr("ascontiguousarray")(array, "dtype"_a = "uint8");
            auto view =
                nb::cast<nb::ndarray<const std::uint8_t, nb::ndim<1>,
                                     nb::c_contig, nb::device::cpu>>(bytes);
            std::vector<bool> bits(view.shape(0));
            for (std::size_t i = 0; i < bits.size(); i++)
            {
                bits[i] = view.data()[i] != 0;
            }
            return bits;
        }

        nb::ndarray<nb::numpy, bool, nb::ndim<1>>
        bits_to_numpy(const std::vector<bool>& bits)
        {
            auto* data =
                new std::vector<std::uint8_t>(bits.begin(), bits.end());
            nb::capsule owner(
                data, [](void* pointer) noexcept
                { delete static_cast<std::vector<std::uint8_t>*>(pointer); });
            return nb::ndarray<nb::numpy, bool, nb::ndim<1>>(
                reinterpret_cast<bool*>(data->data()), {data->size()}, owner);
        }

        nb::ndarray<nb::numpy, std::int32_t>
        int32_to_numpy(std::vector<std::int32_t>&& values,
                       std::vector<std::size_t> shape)
        {
            auto* data = new std::vector<std::int32_t>(std::move(values));
            nb::capsule owner(
                data, [](void* pointer) noexcept
                { delete static_cast<std::vector<std::int32_t>*>(pointer); });
            return nb::ndarray<nb::numpy, std::int32_t>(
                data->data(), shape.size(), shape.data(), owner);
        }

        /**
         * @brief Copies data of a key or a ciphertext to the host.
         */
        std::vector<std::int32_t> copy_data(const std::int32_t* data,
                                            std::size_t size, bool on_device,
                                            cudaStream_t stream)
        {
            if (on_device)
            {
                return device_to_vector(data, size, stream);
            }
            return std::vector<std::int32_t>(data, data + size);
        }

        // --------------------------------------------------------------
        // Checks
        // --------------------------------------------------------------

        void require_generated(const Ciphertext& cipher, const char* name)
        {
            if (!is_generated(cipher))
            {
                throw heongpu::StateError(
                    std::string(name) +
                    " is not generated! Encrypt it or use the result of a "
                    "gate.");
            }
        }

        void require_generated(const SecretKey& key)
        {
            if (!key.is_generated())
            {
                throw heongpu::StateError("Secretkey is not generated! Call "
                                          "generate_secret_key first.");
            }
        }

        void require_generated(const BootstrappingKey& key)
        {
            if (!key.is_generated())
            {
                throw heongpu::StateError(
                    "Bootstrappingkey is not generated! Call "
                    "generate_bootstrapping_key first.");
            }
        }

        void require_same_size(const Ciphertext& first,
                               const Ciphertext& second)
        {
            if (first.size() != second.size())
            {
                throw heongpu::ParameterError(
                    "Ciphertext sizes differ: " + std::to_string(first.size()) +
                    " and " + std::to_string(second.size()) + ".");
            }
        }

        Ciphertext empty_ciphertext()
        {
            // Value initialization sets the members without initializers
            // to zero.
            return Ciphertext();
        }

        // --------------------------------------------------------------
        // Gates
        // --------------------------------------------------------------

        using BinaryGate = void (LogicOperator::*)(
            const Ciphertext&, const Ciphertext&, Ciphertext&,
            const BootstrappingKey&, const heongpu::ExecutionOptions&);

        /**
         * @brief Runs a gate into out, or into a new ciphertext if out is
         * None, and returns the output.
         */
        template <typename Gate>
        nb::object run_gate(nb::handle out, const Gate& gate)
        {
            if (out.is_none())
            {
                Ciphertext result = empty_ciphertext();
                {
                    nb::gil_scoped_release release;
                    gate(result);
                }
                return nb::cast(std::move(result));
            }
            if (!nb::isinstance<Ciphertext>(out))
            {
                throw nb::type_error("out must be a heongpu.tfhe.Ciphertext.");
            }
            Ciphertext& target = nb::cast<Ciphertext&>(out);
            {
                nb::gil_scoped_release release;
                gate(target);
            }
            return nb::borrow(out);
        }

        template <BinaryGate gate>
        nb::object binary_gate(LogicOperator& self, const Ciphertext& input1,
                               const Ciphertext& input2,
                               const BootstrappingKey& boot_key, nb::handle out,
                               nb::handle options)
        {
            heongpu::ExecutionOptions opts = to_options(options);
            require_generated(input1, "input1");
            require_generated(input2, "input2");
            require_same_size(input1, input2);
            require_generated(boot_key);
            return run_gate(
                out, [&](Ciphertext& output)
                { (self.*gate)(input1, input2, output, boot_key, opts); });
        }

        template <BinaryGate gate>
        void def_binary_gate(nb::class_<LogicOperator>& cls, const char* name,
                             const char* doc)
        {
            cls.def(name, &binary_gate<gate>, "input1"_a, "input2"_a,
                    "boot_key"_a, nb::kw_only(), "out"_a = nb::none(),
                    "options"_a = nb::none(), doc);
        }
    } // namespace

    void bind_tfhe(nb::module_& m)
    {
        // --------------------------------------------------------------
        // Context
        // --------------------------------------------------------------

        nb::class_<Context>(
            m, "Context",
            "The TFHE parameters and their precomputed tables. HEonGPU uses "
            "a fixed parameter set with 128-bit security.")
            .def(
                "__init__",
                [](Context* self,
                   std::optional<heongpu::MemoryPoolConfig> pool_config)
                {
                    nb::gil_scoped_release release;
                    if (pool_config)
                    {
                        new (self) Context(*pool_config);
                    }
                    else
                    {
                        new (self) Context();
                    }
                },
                "pool_config"_a = nb::none(),
                "Creates the context. The first context sets up the memory "
                "pool, with pool_config if it is given.")
            .def_prop_ro(
                "lwe_size",
                [](std::shared_ptr<Context> self)
                { return SecretKey(std::move(self)).lwe_size(); },
                "The LWE dimension n of the ciphertexts.")
            .def_prop_ro(
                "tlwe_size",
                [](std::shared_ptr<Context> self)
                { return SecretKey(std::move(self)).tlwe_size(); },
                "The size k * N of the TLWE key (k polynomials of degree N).")
            .def("__repr__",
                 [](std::shared_ptr<Context> self)
                 {
                     SecretKey key(std::move(self));
                     return "<heongpu.tfhe.Context lwe_size=" +
                            std::to_string(key.lwe_size()) +
                            " tlwe_size=" + std::to_string(key.tlwe_size()) +
                            ">";
                 })
            .def("__getstate__", [](const Context&) { return nb::tuple(); })
            .def("__setstate__",
                 [](Context* self, const nb::tuple&)
                 {
                     nb::gil_scoped_release release;
                     new (self) Context();
                 });

        // --------------------------------------------------------------
        // Keys
        // --------------------------------------------------------------

        nb::class_<SecretKey> secret_key(
            m, "SecretKey",
            "The binary LWE and TLWE secret keys. Create it with "
            "KeyGenerator.secret_key().");
        secret_key.def(nb::init<>())
            .def(nb::init<ContextPtr, bool>(), "context"_a,
                 "store_in_gpu"_a = true,
                 "An empty key; fill it with "
                 "KeyGenerator.generate_secret_key().")
            .def_prop_ro("is_generated", &SecretKey::is_generated)
            .def_prop_ro("lwe_size", &SecretKey::lwe_size,
                         "The size n of the LWE key.")
            .def_prop_ro("tlwe_size", &SecretKey::tlwe_size,
                         "The size k * N of the TLWE key.")
            .def_prop_ro(
                "stream",
                [](const SecretKey& self)
                { return from_stream(self.stream()); },
                "Handle of the CUDA stream of the data.")
            .def(
                "switch_stream",
                [](SecretKey& self, nb::handle stream)
                { self.switch_stream(to_stream(stream)); },
                nb::arg("stream").none(), "Uses another stream for the data.")
            .def("set_context", &SecretKey::set_context, "context"_a,
                 "Attaches a context to an empty key.")
            .def(
                "get_lwe_key",
                [](SecretKey& self)
                {
                    require_generated(self);
                    std::vector<std::int32_t> data;
                    {
                        nb::gil_scoped_release release;
                        data = copy_data(self.lwe_data(), self.lwe_size(),
                                         self.is_on_device(), self.stream());
                    }
                    std::size_t size = data.size();
                    return int32_to_numpy(std::move(data), {size});
                },
                "Returns a copy of the binary LWE key as an int32 NumPy "
                "array.")
            .def(
                "get_tlwe_key",
                [](SecretKey& self)
                {
                    require_generated(self);
                    std::vector<std::int32_t> data;
                    {
                        nb::gil_scoped_release release;
                        data = copy_data(self.tlwe_data(), self.tlwe_size(),
                                         self.is_on_device(), self.stream());
                    }
                    std::size_t size = data.size();
                    return int32_to_numpy(std::move(data), {size});
                },
                "Returns a copy of the binary TLWE key as an int32 NumPy "
                "array.");
        def_storage<SecretKey>(secret_key);
        def_tfhe_serialization<SecretKey>(secret_key);
        def_copy<SecretKey>(secret_key);

        nb::class_<BootstrappingKey> boot_key(
            m, "BootstrappingKey",
            "The bootstrapping key and the key switching key of the gates. "
            "Create it with KeyGenerator.bootstrapping_key().");
        boot_key.def(nb::init<>())
            .def(nb::init<ContextPtr, bool>(), "context"_a,
                 "store_in_gpu"_a = true,
                 "An empty key; fill it with "
                 "KeyGenerator.generate_bootstrapping_key().")
            .def_prop_ro("is_generated", &BootstrappingKey::is_generated)
            .def_prop_ro(
                "stream",
                [](const BootstrappingKey& self)
                { return from_stream(self.stream()); },
                "Handle of the CUDA stream of the data.")
            .def(
                "switch_stream",
                [](BootstrappingKey& self, nb::handle stream)
                { self.switch_stream(to_stream(stream)); },
                nb::arg("stream").none(), "Uses another stream for the data.")
            .def("set_context", &BootstrappingKey::set_context, "context"_a,
                 "Attaches a context to the key.");
        def_storage<BootstrappingKey>(boot_key);
        def_tfhe_serialization<BootstrappingKey>(boot_key);
        def_copy<BootstrappingKey>(boot_key);

        // --------------------------------------------------------------
        // Ciphertext
        // --------------------------------------------------------------

        nb::class_<Ciphertext> ciphertext(
            m, "Ciphertext",
            "A vector of encrypted bits: one LWE ciphertext per bit.");
        ciphertext
            .def(
                "__init__",
                [](Ciphertext* self, ContextPtr context, nb::handle options)
                {
                    new (self)
                        Ciphertext(std::move(context), to_options(options));
                    State state{Operation::reset};
                    access(*self, state);
                },
                "context"_a, "options"_a = nb::none(),
                "An empty ciphertext, for example as the out argument of a "
                "gate.")
            .def_prop_ro("size", &Ciphertext::size,
                         "The number of encrypted bits.")
            .def("__len__", &Ciphertext::size)
            .def_prop_ro("lwe_size", &Ciphertext::lwe_size,
                         "The LWE dimension n.")
            .def_prop_ro(
                "is_generated",
                [](const Ciphertext& self) { return is_generated(self); },
                "Whether the ciphertext holds encrypted bits.")
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
                "Uses another stream for the data. Work queued on the old "
                "stream is not waited for.")
            .def(
                "get_data",
                [](Ciphertext& self)
                {
                    require_generated(self, "Ciphertext");
                    std::size_t size = self.size();
                    std::size_t n = self.lwe_size();
                    std::vector<std::int32_t> a, b;
                    {
                        nb::gil_scoped_release release;
                        a = copy_data(self.a_data(), size * n,
                                      self.is_on_device(), self.stream());
                        b = copy_data(self.b_data(), size, self.is_on_device(),
                                      self.stream());
                    }
                    return nb::make_tuple(
                        int32_to_numpy(std::move(a), {size, n}),
                        int32_to_numpy(std::move(b), {size}));
                },
                "Returns copies of the LWE masks (an int32 array of shape "
                "(size, lwe_size)) and bodies (shape (size,)) on the torus "
                "(2^32).")
            .def("__copy__",
                 [](const Ciphertext& self)
                 {
                     nb::gil_scoped_release release;
                     Ciphertext copy(self);
                     HEONGPU_PY_CUDA_CHECK(
                         cudaStreamSynchronize(self.stream()));
                     return copy;
                 })
            .def(
                "__deepcopy__",
                [](const Ciphertext& self, nb::handle)
                {
                    nb::gil_scoped_release release;
                    Ciphertext copy(self);
                    HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(self.stream()));
                    return copy;
                },
                "memo"_a)
            .def(
                "copy",
                [](const Ciphertext& self)
                {
                    nb::gil_scoped_release release;
                    Ciphertext copy(self);
                    HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(self.stream()));
                    return copy;
                },
                "Returns a copy (the data is copied).")
            .def("__repr__",
                 [](const Ciphertext& self)
                 {
                     std::ostringstream stream;
                     stream << "<heongpu.tfhe.Ciphertext size=" << self.size()
                            << " lwe_size=" << self.lwe_size()
                            << (self.is_on_device() ? " device" : " host")
                            << ">";
                     return stream.str();
                 });
        def_storage<Ciphertext>(ciphertext);
        def_tfhe_serialization<Ciphertext>(ciphertext);

        // --------------------------------------------------------------
        // Key generator, encryptor and decryptor
        // --------------------------------------------------------------

        nb::class_<KeyGenerator>(m, "KeyGenerator", "Generates the TFHE keys.")
            .def(nb::init<ContextPtr>(), "context"_a)
            .def(
                "secret_key",
                [](KeyGenerator& self, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    SecretKey key(self.context());
                    self.generate_secret_key(key, opts);
                    return key;
                },
                "options"_a = nb::none(),
                "Generates a new secret key: independent binary LWE and "
                "TLWE keys from the random generator of the operating "
                "system.")
            .def(
                "bootstrapping_key",
                [](KeyGenerator& self, const SecretKey& sk, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    require_generated(sk);
                    nb::gil_scoped_release release;
                    BootstrappingKey key(self.context());
                    self.generate_bootstrapping_key(key, sk, opts);
                    return key;
                },
                "secret_key"_a, "options"_a = nb::none(),
                "Generates a new bootstrapping key (with its key switching "
                "key) for the secret key.")
            .def(
                "generate_secret_key",
                [](KeyGenerator& self, SecretKey& sk, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    if (sk.is_generated())
                    {
                        throw heongpu::StateError(
                            "Secretkey is already generated!");
                    }
                    nb::gil_scoped_release release;
                    self.generate_secret_key(sk, opts);
                },
                "secret_key"_a, "options"_a = nb::none(),
                "Generates into an empty key.")
            .def(
                "generate_bootstrapping_key",
                [](KeyGenerator& self, BootstrappingKey& bk,
                   const SecretKey& sk, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    require_generated(sk);
                    if (bk.is_generated())
                    {
                        throw heongpu::StateError(
                            "Bootstrappingkey is already generated!");
                    }
                    nb::gil_scoped_release release;
                    self.generate_bootstrapping_key(bk, sk, opts);
                },
                "boot_key"_a, "secret_key"_a, "options"_a = nb::none(),
                "Generates into an empty bootstrapping key.");

        nb::class_<Encryptor>(m, "Encryptor",
                              "Encrypts bits with the secret key.")
            .def(
                "__init__",
                [](Encryptor* self, ContextPtr context, const SecretKey& sk)
                {
                    require_generated(sk);
                    nb::gil_scoped_release release;
                    new (self) Encryptor(std::move(context), sk);
                },
                "context"_a, "secret_key"_a)
            .def(
                "encrypt",
                [](Encryptor& self, nb::handle bits, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    std::vector<bool> messages = to_bits(bits);
                    nb::gil_scoped_release release;
                    Ciphertext cipher = empty_ciphertext();
                    self.encrypt(cipher, messages, opts);
                    return cipher;
                },
                "bits"_a, "options"_a = nb::none(),
                "Encrypts a bool, a sequence or an array of booleans (or of "
                "the integers 0 and 1), one LWE ciphertext per bit.");

        nb::class_<Decryptor>(m, "Decryptor",
                              "Decrypts ciphertexts with the secret key.")
            .def(
                "__init__",
                [](Decryptor* self, ContextPtr context, const SecretKey& sk)
                {
                    nb::gil_scoped_release release;
                    new (self) Decryptor(std::move(context), sk);
                },
                "context"_a, "secret_key"_a)
            .def(
                "decrypt",
                [](Decryptor& self, const Ciphertext& cipher,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    require_generated(cipher, "Ciphertext");
                    std::vector<bool> bits;
                    {
                        nb::gil_scoped_release release;
                        self.decrypt(cipher, bits, opts);
                    }
                    return bits_to_numpy(bits);
                },
                "cipher"_a, "options"_a = nb::none(),
                "Decrypts to a bool NumPy array.");

        // --------------------------------------------------------------
        // Logic operator
        // --------------------------------------------------------------

        nb::class_<LogicOperator> logic(
            m, "LogicOperator",
            "Evaluates Boolean gates bit by bit on ciphertexts of the same "
            "size. Every gate except NOT bootstraps its result with the "
            "bootstrapping key, so the noise does not grow. Each gate "
            "returns a new ciphertext, or writes into out (which may be one "
            "of the inputs) and returns it.");
        logic.def(nb::init<ContextPtr>(), "context"_a);
        def_binary_gate<&LogicOperator::NAND>(logic, "NAND",
                                              "Returns NOT (input1 AND "
                                              "input2).");
        def_binary_gate<&LogicOperator::AND>(logic, "AND",
                                             "Returns input1 AND input2.");
        def_binary_gate<&LogicOperator::NOR>(logic, "NOR",
                                             "Returns NOT (input1 OR "
                                             "input2).");
        def_binary_gate<&LogicOperator::OR>(logic, "OR",
                                            "Returns input1 OR input2.");
        def_binary_gate<&LogicOperator::XNOR>(logic, "XNOR",
                                              "Returns NOT (input1 XOR "
                                              "input2).");
        def_binary_gate<&LogicOperator::XOR>(logic, "XOR",
                                             "Returns input1 XOR input2.");
        logic.def(
            "NOT",
            [](LogicOperator& self, const Ciphertext& input1, nb::handle out,
               nb::handle options)
            {
                heongpu::ExecutionOptions opts = to_options(options);
                require_generated(input1, "input1");
                return run_gate(out, [&](Ciphertext& output)
                                { self.NOT(input1, output, opts); });
            },
            "input1"_a, nb::kw_only(), "out"_a = nb::none(),
            "options"_a = nb::none(),
            "Returns NOT input1. It needs no bootstrapping key and adds no "
            "noise.");
        logic.def(
            "MUX",
            [](LogicOperator& self, const Ciphertext& input1,
               const Ciphertext& input2, const Ciphertext& control_input,
               const BootstrappingKey& boot_key, nb::handle out,
               nb::handle options)
            {
                heongpu::ExecutionOptions opts = to_options(options);
                require_generated(input1, "input1");
                require_generated(input2, "input2");
                require_generated(control_input, "control_input");
                require_same_size(input1, input2);
                require_same_size(input1, control_input);
                require_generated(boot_key);
                return run_gate(out,
                                [&](Ciphertext& output) {
                                    self.MUX(input1, input2, control_input,
                                             output, boot_key, opts);
                                });
            },
            "input1"_a, "input2"_a, "control_input"_a, "boot_key"_a,
            nb::kw_only(), "out"_a = nb::none(), "options"_a = nb::none(),
            "Returns input1 where control_input is 1 and input2 where it is "
            "0 (two bootstrappings per bit).");
    }

} // namespace heongpu_py
