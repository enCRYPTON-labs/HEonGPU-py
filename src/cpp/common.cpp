// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

#include <complex>

namespace heongpu_py
{
    namespace
    {
        cudaStream_t handle_to_stream(nb::handle value)
        {
            if (!nb::isinstance<nb::int_>(value))
            {
                throw nb::type_error("A CUDA stream handle must be an int.");
            }
            return reinterpret_cast<cudaStream_t>(
                nb::cast<std::uintptr_t>(value));
        }

        nb::object numpy() { return nb::module_::import_("numpy"); }

        nb::object as_array(nb::handle values, const char* dtype,
                            const char* name)
        {
            nb::object array;
            try
            {
                array = numpy().attr("ascontiguousarray")(values, dtype);
            }
            catch (nb::python_error& error)
            {
                throw nb::type_error(
                    (std::string(name) + " must be a number, a sequence or an "
                                         "array of numbers: " +
                     error.what())
                        .c_str());
            }
            nb::object ndim = array.attr("ndim");
            int dimensions = nb::cast<int>(ndim);
            if (dimensions == 0)
            {
                array = array.attr("reshape")(1);
            }
            else if (dimensions != 1)
            {
                throw nb::value_error(
                    (std::string(name) + " must be one-dimensional.").c_str());
            }
            return array;
        }
    } // namespace

    cudaStream_t to_stream(nb::handle stream)
    {
        if (!stream.is_valid() || stream.is_none())
        {
            return cudaStreamDefault;
        }
        if (nb::isinstance<nb::int_>(stream))
        {
            return handle_to_stream(stream);
        }
        if (nb::hasattr(stream, "__cuda_stream__"))
        {
            // CUDA stream protocol: returns (version, handle).
            nb::object result = stream.attr("__cuda_stream__")();
            nb::tuple pair = nb::cast<nb::tuple>(result);
            if (pair.size() != 2)
            {
                throw nb::type_error(
                    "__cuda_stream__() must return (version, handle).");
            }
            return handle_to_stream(pair[1]);
        }
        for (const char* attribute : {"cuda_stream", "ptr", "handle"})
        {
            if (nb::hasattr(stream, attribute))
            {
                nb::object value = stream.attr(attribute);
                if (nb::isinstance<nb::int_>(value))
                {
                    return handle_to_stream(value);
                }
                if (nb::hasattr(value, "__int__"))
                {
                    return reinterpret_cast<cudaStream_t>(
                        nb::cast<std::uintptr_t>(nb::int_(value)));
                }
            }
        }
        throw nb::type_error(
            "stream must be None, an int (a cudaStream_t handle) or a CUDA "
            "stream object (with __cuda_stream__, cuda_stream, ptr or "
            "handle).");
    }

    heongpu::ExecutionOptions to_options(nb::handle options)
    {
        if (!options.is_valid() || options.is_none())
        {
            return heongpu::ExecutionOptions();
        }
        if (nb::isinstance<heongpu::ExecutionOptions>(options))
        {
            return nb::cast<heongpu::ExecutionOptions>(options);
        }
        return heongpu::ExecutionOptions().set_stream(to_stream(options));
    }

    bool is_complex(nb::handle values)
    {
        if (PyComplex_Check(values.ptr()))
        {
            return true;
        }
        return nb::cast<bool>(numpy().attr("iscomplexobj")(values));
    }

    template <typename T>
    std::vector<T> to_vector(nb::handle values, const char* dtype,
                             const char* name)
    {
        nb::object array = as_array(values, dtype, name);
        auto view = nb::cast<nb::ndarray<const T, nb::ndim<1>, nb::c_contig,
                                         nb::device::cpu>>(array);
        return std::vector<T>(view.data(), view.data() + view.shape(0));
    }

    template std::vector<double> to_vector<double>(nb::handle, const char*,
                                                   const char*);
    template std::vector<std::int64_t>
    to_vector<std::int64_t>(nb::handle, const char*, const char*);
    template std::vector<std::uint64_t>
    to_vector<std::uint64_t>(nb::handle, const char*, const char*);
    template std::vector<std::int32_t>
    to_vector<std::int32_t>(nb::handle, const char*, const char*);
    template std::vector<std::uint32_t>
    to_vector<std::uint32_t>(nb::handle, const char*, const char*);

    std::vector<double> to_double_vector(nb::handle values, const char* name)
    {
        return to_vector<double>(values, "float64", name);
    }

    std::vector<Complex64> to_complex_vector(nb::handle values,
                                                      const char* name)
    {
        nb::object array = as_array(values, "complex128", name);
        auto view =
            nb::cast<nb::ndarray<const std::complex<double>, nb::ndim<1>,
                                 nb::c_contig, nb::device::cpu>>(array);
        std::vector<Complex64> result;
        result.reserve(view.shape(0));
        for (size_t i = 0; i < view.shape(0); i++)
        {
            result.emplace_back(view.data()[i].real(), view.data()[i].imag());
        }
        return result;
    }

    std::vector<std::int64_t> to_int64_vector(nb::handle values,
                                              const char* name)
    {
        return to_vector<std::int64_t>(values, "int64", name);
    }

    std::vector<std::uint64_t> to_uint64_vector(nb::handle values,
                                                const char* name)
    {
        return to_vector<std::uint64_t>(values, "uint64", name);
    }

    std::vector<int> to_int_vector(nb::handle values, const char* name)
    {
        return to_vector<std::int32_t>(values, "int32", name);
    }

    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>
    to_numpy(std::vector<Complex64>&& values)
    {
        std::vector<std::complex<double>> result;
        result.reserve(values.size());
        for (const auto& value : values)
        {
            result.emplace_back(value.real(), value.imag());
        }
        values.clear();
        return to_numpy(std::move(result));
    }

    void write_file(nb::handle file, const std::string& data)
    {
        if (nb::hasattr(file, "write"))
        {
            file.attr("write")(nb::bytes(data.data(), data.size()));
            return;
        }
        nb::object path = nb::module_::import_("os").attr("fspath")(file);
        nb::object handle =
            nb::module_::import_("builtins").attr("open")(path, "wb");
        try
        {
            handle.attr("write")(nb::bytes(data.data(), data.size()));
        }
        catch (...)
        {
            handle.attr("close")();
            throw;
        }
        handle.attr("close")();
    }

    std::string read_file(nb::handle file)
    {
        nb::object data;
        if (nb::hasattr(file, "read"))
        {
            data = file.attr("read")();
        }
        else
        {
            nb::object path = nb::module_::import_("os").attr("fspath")(file);
            nb::object handle =
                nb::module_::import_("builtins").attr("open")(path, "rb");
            try
            {
                data = handle.attr("read")();
            }
            catch (...)
            {
                handle.attr("close")();
                throw;
            }
            handle.attr("close")();
        }
        if (!nb::isinstance<nb::bytes>(data))
        {
            throw nb::type_error("The file must be opened in binary mode.");
        }
        return bytes_to_string(nb::borrow<nb::bytes>(data));
    }

    std::string bytes_to_string(nb::bytes data)
    {
        return std::string(data.c_str(), data.size());
    }

} // namespace heongpu_py
