// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

#include <mutex>

namespace heongpu_py
{
    namespace
    {
        /**
         * @brief Streams of closed Stream objects. HEonGPU objects keep the
         * stream they were created on and use it when they are freed, so a
         * stream is only destroyed by shutdown(), when no HEonGPU object is
         * left.
         */
        std::mutex retired_streams_mutex;
        std::vector<cudaStream_t> retired_streams;

        void retire_stream(cudaStream_t stream)
        {
            std::lock_guard<std::mutex> lock(retired_streams_mutex);
            retired_streams.push_back(stream);
        }

        void destroy_retired_streams()
        {
            std::lock_guard<std::mutex> lock(retired_streams_mutex);
            for (cudaStream_t stream : retired_streams)
            {
                cudaStreamDestroy(stream);
            }
            retired_streams.clear();
        }

        /**
         * @brief A CUDA stream owned by Python. close() (or leaving a with
         * block, or deleting the object) waits for the stream and retires
         * it; the CUDA stream itself is destroyed by shutdown().
         */
        class Stream
        {
          public:
            explicit Stream(bool non_blocking)
            {
                HEONGPU_PY_CUDA_CHECK(cudaStreamCreateWithFlags(
                    &stream_, non_blocking ? cudaStreamNonBlocking
                                           : cudaStreamDefault));
            }

            Stream(const Stream&) = delete;
            Stream& operator=(const Stream&) = delete;

            ~Stream() { close(); }

            void close() noexcept
            {
                if (stream_ != nullptr)
                {
                    cudaStreamSynchronize(stream_);
                    retire_stream(stream_);
                    stream_ = nullptr;
                }
            }

            bool closed() const noexcept { return stream_ == nullptr; }

            cudaStream_t get() const
            {
                if (stream_ == nullptr)
                {
                    throw heongpu::StateError("The stream is closed.");
                }
                return stream_;
            }

            void synchronize() const
            {
                cudaStream_t stream = get();
                nb::gil_scoped_release release;
                HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

          private:
            cudaStream_t stream_ = nullptr;
        };

        heongpu::MemoryPoolConfig
        make_pool_config(std::optional<float> initial_device_fraction,
                         std::optional<float> max_device_fraction,
                         std::optional<size_t> initial_device_bytes,
                         std::optional<size_t> max_device_bytes,
                         std::optional<float> initial_host_fraction,
                         std::optional<float> max_host_fraction,
                         std::optional<size_t> initial_host_bytes,
                         std::optional<size_t> max_host_bytes,
                         bool use_memory_pool, bool set_as_rmm_default,
                         bool raise_kernel_stack_size)
        {
            heongpu::MemoryPoolConfig config;
            config.initial_device_fraction = initial_device_fraction;
            config.max_device_fraction = max_device_fraction;
            config.initial_device_bytes = initial_device_bytes;
            config.max_device_bytes = max_device_bytes;
            config.initial_host_fraction = initial_host_fraction;
            config.max_host_fraction = max_host_fraction;
            config.initial_host_bytes = initial_host_bytes;
            config.max_host_bytes = max_host_bytes;
            config.use_memory_pool = use_memory_pool;
            config.set_as_rmm_default = set_as_rmm_default;
            config.raise_kernel_stack_size = raise_kernel_stack_size;
            return config;
        }

        template <typename T> std::string optional_repr(const std::optional<T>& v)
        {
            if (!v)
            {
                return "None";
            }
            std::ostringstream stream;
            stream << *v;
            return stream.str();
        }
    } // namespace

    void bind_core(nb::module_& m)
    {
        m.attr("__version__") = HEONGPU_PY_VERSION;
        m.attr("heongpu_version") = HEONGPU_PY_HEONGPU_VERSION;

        // --------------------------------------------------------------
        // Enumerations
        // --------------------------------------------------------------

        nb::enum_<heongpu::sec_level_type>(m, "SecurityLevel",
                                           "Security level of the parameters "
                                           "(HomomorphicEncryption.org "
                                           "standard).")
            .value("NONE", heongpu::sec_level_type::none,
                   "No security check (for testing only).")
            .value("SEC128", heongpu::sec_level_type::sec128, "128-bit.")
            .value("SEC192", heongpu::sec_level_type::sec192, "192-bit.")
            .value("SEC256", heongpu::sec_level_type::sec256, "256-bit.");

        nb::enum_<heongpu::storage_type>(m, "StorageType",
                                         "Where the data of an object is.")
            .value("HOST", heongpu::storage_type::host, "Host memory.")
            .value("DEVICE", heongpu::storage_type::device, "GPU memory.");

        nb::enum_<heongpu::encoding>(m, "Encoding",
                                     "How a CKKS message is encoded.")
            .value("SLOT", heongpu::encoding::slot,
                   "One value per slot (the default).")
            .value("COEFFICIENT", heongpu::encoding::coefficient,
                   "One value per polynomial coefficient.");

        nb::enum_<heongpu::rescale_mode>(m, "RescaleMode",
                                         "How a CKKS context rescales.")
            .value("MANUAL", heongpu::rescale_mode::manual,
                   "The application rescales and relinearizes.")
            .value("AUTOMATIC", heongpu::rescale_mode::automatic,
                   "The operators rescale and relinearize, and align levels "
                   "and scales.");

        nb::enum_<heongpu::session_role>(m, "SessionRole",
                                         "Which keys a session holds.")
            .value("OWNER", heongpu::session_role::owner,
                   "All keys, including the secret key.")
            .value("EVALUATOR", heongpu::session_role::evaluator,
                   "Public and evaluation keys: computes, cannot decrypt.")
            .value("ENCRYPTOR", heongpu::session_role::encryptor,
                   "Public key only: encrypts, cannot compute or decrypt.");

        nb::enum_<heongpu::arithmetic_bootstrapping_type>(
            m, "ArithmeticBootstrappingType",
            "Type of CKKS arithmetic bootstrapping.")
            .value("NONE", heongpu::arithmetic_bootstrapping_type::none)
            .value("REGULAR", heongpu::arithmetic_bootstrapping_type::regular)
            .value("SLIM", heongpu::arithmetic_bootstrapping_type::slim);

        nb::enum_<heongpu::logic_bootstrapping_type>(
            m, "LogicBootstrappingType", "Type of CKKS logic bootstrapping.")
            .value("NONE", heongpu::logic_bootstrapping_type::none)
            .value("BIT", heongpu::logic_bootstrapping_type::bit)
            .value("GATE", heongpu::logic_bootstrapping_type::gate);

        nb::enum_<heongpu::scheme_type>(m, "SchemeType", "Encryption scheme.")
            .value("NONE", heongpu::scheme_type::none)
            .value("BFV", heongpu::scheme_type::bfv)
            .value("CKKS", heongpu::scheme_type::ckks)
            .value("BGV", heongpu::scheme_type::bgv)
            .value("TFHE", heongpu::scheme_type::tfhe);

        // --------------------------------------------------------------
        // Execution options and streams
        // --------------------------------------------------------------

        nb::class_<heongpu::ExecutionOptions>(
            m, "ExecutionOptions",
            "Options of an operation: the CUDA stream it runs on, where its "
            "output is stored and whether its inputs go back to where they "
            "were.")
            .def(
                "__init__",
                [](heongpu::ExecutionOptions* self, nb::handle stream,
                   heongpu::storage_type storage, bool keep_initial_condition)
                {
                    new (self) heongpu::ExecutionOptions();
                    self->set_stream(to_stream(stream));
                    self->set_storage_type(storage);
                    self->set_initial_location(keep_initial_condition);
                },
                "stream"_a = nb::none(),
                "storage"_a = heongpu::storage_type::device,
                "keep_initial_condition"_a = true)
            .def_prop_rw(
                "stream",
                [](const heongpu::ExecutionOptions& self)
                { return from_stream(self.stream_); },
                [](heongpu::ExecutionOptions& self, nb::handle stream)
                { self.stream_ = to_stream(stream); },
                nb::arg("stream").none(),
                "Handle of the CUDA stream (0 is the default stream).")
            .def_rw("storage", &heongpu::ExecutionOptions::storage_,
                    "Where the output is stored.")
            .def_rw("keep_initial_condition",
                    &heongpu::ExecutionOptions::keep_initial_condition_,
                    "Whether inputs that are moved for the operation go back "
                    "to where they were.")
            .def("__repr__",
                 [](const heongpu::ExecutionOptions& self)
                 {
                     std::ostringstream stream;
                     stream << "ExecutionOptions(stream=" << from_stream(self.stream_)
                            << ", storage="
                            << heongpu::detail::name(self.storage_)
                            << ", keep_initial_condition="
                            << (self.keep_initial_condition_ ? "True" : "False")
                            << ")";
                     return stream.str();
                 });

        nb::class_<Stream>(m, "Stream",
                           "A CUDA stream created and owned by HEonGPU. Any "
                           "object with __cuda_stream__, cuda_stream, ptr or "
                           "handle (for example a CuPy or PyTorch stream) or "
                           "an int handle can be used wherever a stream is "
                           "accepted.")
            .def(nb::init<bool>(), "non_blocking"_a = true,
                 "Creates a stream; a non-blocking stream does not "
                 "synchronize with the legacy default stream.")
            .def_prop_ro(
                "handle", [](const Stream& self)
                { return from_stream(self.get()); },
                "The cudaStream_t handle as an int.")
            .def(
                "__cuda_stream__", [](const Stream& self)
                { return std::make_pair(0, from_stream(self.get())); },
                "CUDA stream protocol: returns (0, handle).")
            .def("synchronize", &Stream::synchronize,
                 "Waits until all work on the stream is done.")
            .def("close", &Stream::close,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "Waits for the stream and retires it. The CUDA stream is "
                 "destroyed by shutdown(), since HEonGPU objects created on "
                 "it still use it when they are freed.")
            .def_prop_ro("closed", &Stream::closed)
            .def("__enter__", [](Stream& self) -> Stream& { return self; },
                 nb::rv_policy::reference)
            .def("__exit__",
                 [](Stream& self, nb::handle, nb::handle, nb::handle)
                 { self.close(); },
                 nb::arg().none(), nb::arg().none(), nb::arg().none())
            .def("__int__",
                 [](const Stream& self) { return from_stream(self.get()); });

        // --------------------------------------------------------------
        // Devices
        // --------------------------------------------------------------

        m.def(
            "device_count",
            []()
            {
                int count = 0;
                cudaError_t error = cudaGetDeviceCount(&count);
                if (error != cudaSuccess)
                {
                    cudaGetLastError();
                    return 0;
                }
                return count;
            },
            "Returns the number of CUDA devices (0 if there is no usable "
            "device or driver).");
        m.def(
            "set_device", [](int device)
            { HEONGPU_PY_CUDA_CHECK(cudaSetDevice(device)); }, "device"_a,
            "Selects the CUDA device of the calling thread. Select it before "
            "creating any HEonGPU object.");
        m.def(
            "get_device",
            []()
            {
                int device = 0;
                HEONGPU_PY_CUDA_CHECK(cudaGetDevice(&device));
                return device;
            },
            "Returns the CUDA device of the calling thread.");
        m.def(
            "synchronize",
            []()
            {
                nb::gil_scoped_release release;
                HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
            },
            "Waits until all work on the current device is done.");

        // --------------------------------------------------------------
        // Memory pool and process-wide state
        // --------------------------------------------------------------

        nb::class_<heongpu::MemoryPoolConfig>(
            m, "MemoryPoolConfig",
            "Sizes of the device and host memory pools. Fractions are of the "
            "total memory (0.0 to 1.0, or 0 to 100 as a percentage); byte "
            "sizes take precedence over fractions. Unset values keep the "
            "HEonGPU defaults.")
            .def(
                "__init__",
                [](heongpu::MemoryPoolConfig* self,
                   std::optional<float> initial_device_fraction,
                   std::optional<float> max_device_fraction,
                   std::optional<size_t> initial_device_bytes,
                   std::optional<size_t> max_device_bytes,
                   std::optional<float> initial_host_fraction,
                   std::optional<float> max_host_fraction,
                   std::optional<size_t> initial_host_bytes,
                   std::optional<size_t> max_host_bytes, bool use_memory_pool,
                   bool set_as_rmm_default, bool raise_kernel_stack_size)
                {
                    new (self) heongpu::MemoryPoolConfig(make_pool_config(
                        initial_device_fraction, max_device_fraction,
                        initial_device_bytes, max_device_bytes,
                        initial_host_fraction, max_host_fraction,
                        initial_host_bytes, max_host_bytes, use_memory_pool,
                        set_as_rmm_default, raise_kernel_stack_size));
                },
                nb::kw_only(), "initial_device_fraction"_a = nb::none(),
                "max_device_fraction"_a = nb::none(),
                "initial_device_bytes"_a = nb::none(),
                "max_device_bytes"_a = nb::none(),
                "initial_host_fraction"_a = nb::none(),
                "max_host_fraction"_a = nb::none(),
                "initial_host_bytes"_a = nb::none(),
                "max_host_bytes"_a = nb::none(), "use_memory_pool"_a = true,
                "set_as_rmm_default"_a = false,
                "raise_kernel_stack_size"_a = true)
            .def_static("defaults", &heongpu::MemoryPoolConfig::Defaults,
                        "Returns the default configuration.")
            .def_rw("initial_device_fraction",
                    &heongpu::MemoryPoolConfig::initial_device_fraction,
                    nb::arg().none())
            .def_rw("max_device_fraction",
                    &heongpu::MemoryPoolConfig::max_device_fraction,
                    nb::arg().none())
            .def_rw("initial_device_bytes",
                    &heongpu::MemoryPoolConfig::initial_device_bytes,
                    nb::arg().none())
            .def_rw("max_device_bytes",
                    &heongpu::MemoryPoolConfig::max_device_bytes,
                    nb::arg().none())
            .def_rw("initial_host_fraction",
                    &heongpu::MemoryPoolConfig::initial_host_fraction,
                    nb::arg().none())
            .def_rw("max_host_fraction",
                    &heongpu::MemoryPoolConfig::max_host_fraction,
                    nb::arg().none())
            .def_rw("initial_host_bytes",
                    &heongpu::MemoryPoolConfig::initial_host_bytes,
                    nb::arg().none())
            .def_rw("max_host_bytes",
                    &heongpu::MemoryPoolConfig::max_host_bytes,
                    nb::arg().none())
            .def_rw("use_memory_pool",
                    &heongpu::MemoryPoolConfig::use_memory_pool)
            .def_rw("set_as_rmm_default",
                    &heongpu::MemoryPoolConfig::set_as_rmm_default)
            .def_rw("raise_kernel_stack_size",
                    &heongpu::MemoryPoolConfig::raise_kernel_stack_size)
            .def("__repr__",
                 [](const heongpu::MemoryPoolConfig& self)
                 {
                     std::ostringstream stream;
                     stream << "MemoryPoolConfig(initial_device_fraction="
                            << optional_repr(self.initial_device_fraction)
                            << ", max_device_fraction="
                            << optional_repr(self.max_device_fraction)
                            << ", initial_device_bytes="
                            << optional_repr(self.initial_device_bytes)
                            << ", max_device_bytes="
                            << optional_repr(self.max_device_bytes)
                            << ", initial_host_fraction="
                            << optional_repr(self.initial_host_fraction)
                            << ", max_host_fraction="
                            << optional_repr(self.max_host_fraction)
                            << ", initial_host_bytes="
                            << optional_repr(self.initial_host_bytes)
                            << ", max_host_bytes="
                            << optional_repr(self.max_host_bytes)
                            << ", use_memory_pool="
                            << (self.use_memory_pool ? "True" : "False")
                            << ", set_as_rmm_default="
                            << (self.set_as_rmm_default ? "True" : "False")
                            << ", raise_kernel_stack_size="
                            << (self.raise_kernel_stack_size ? "True"
                                                             : "False")
                            << ")";
                     return stream.str();
                 });

        m.def(
            "initialize",
            [](std::optional<heongpu::MemoryPoolConfig> config)
            {
                nb::gil_scoped_release release;
                heongpu::initialize(
                    config ? *config : heongpu::MemoryPoolConfig::Defaults());
            },
            "config"_a = nb::none(),
            "Sets up the process-wide state of HEonGPU (memory pool, random "
            "number generator, kernel stack size). Optional: generating the "
            "first context does it with the context's configuration. Calling "
            "it again keeps the existing pool; its sizes only change after "
            "shutdown().");
        m.def(
            "shutdown",
            []()
            {
                nb::gil_scoped_release release;
                heongpu::shutdown();
                destroy_retired_streams();
            },
            "Releases the process-wide state of HEonGPU and destroys the "
            "closed Stream objects' CUDA streams. All HEonGPU objects must be "
            "destroyed first, otherwise StateError is raised and nothing is "
            "released.");

        m.def(
            "memory_pool_usage",
            []()
            {
                auto& pool = heongpu::MemoryPool::instance();
                nb::dict result;
                result["device_used"] =
                    pool.get_current_device_pool_memory_usage();
                result["device_free"] = pool.get_free_device_pool_memory();
                result["host_used"] = pool.get_current_host_pool_memory_usage();
                result["host_free"] = pool.get_free_host_pool_memory();
                return result;
            },
            "Returns the used and free bytes of the device and host memory "
            "pools as a dict.");
        m.def(
            "print_memory_pool_status",
            []()
            {
                std::ostringstream stream;
                heongpu::MemoryPool::instance().print_memory_pool_status(
                    stream);
                nb::print(nb::str(stream.str().c_str()), nb::str(""));
            },
            "Prints the sizes and usage of the memory pools.");
    }

} // namespace heongpu_py
