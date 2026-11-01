// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_PY_COMMON_H
#define HEONGPU_PY_COMMON_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <heongpu/heongpu.hpp>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Throws heongpu::CudaException (CudaError in Python) if a CUDA call
 * fails.
 */
#define HEONGPU_PY_CUDA_CHECK(call)                                            \
    do                                                                         \
    {                                                                          \
        cudaError_t heongpu_py_error = (call);                                 \
        if (heongpu_py_error != cudaSuccess)                                   \
        {                                                                      \
            throw heongpu::CudaException(__FILE__, __LINE__,                   \
                                         heongpu_py_error);                    \
        }                                                                      \
    } while (0)

namespace heongpu_py
{
    namespace nb = nanobind;
    using namespace nb::literals;

    // ------------------------------------------------------------------
    // Module parts. Each one is defined in its own source file and called
    // from module.cpp.
    // ------------------------------------------------------------------

    void bind_exceptions(nb::module_& m);
    void bind_core(nb::module_& m);

    void bind_ckks_core(nb::module_& m);
    void bind_ckks_operator(nb::module_& m);
    void bind_ckks_session(nb::module_& m);
    void bind_ckks_mpc(nb::module_& m);

    void bind_bfv_core(nb::module_& m);
    void bind_bfv_operator(nb::module_& m);
    void bind_bfv_session(nb::module_& m);
    void bind_bfv_mpc(nb::module_& m);

    void bind_tfhe(nb::module_& m);

    // ------------------------------------------------------------------
    // CUDA streams
    // ------------------------------------------------------------------

    /**
     * @brief Returns the CUDA stream of a Python object: None (the default
     * stream), an int (a cudaStream_t handle), an object with the
     * __cuda_stream__ protocol, or an object with a cuda_stream, ptr or handle
     * attribute (for example torch.cuda.Stream, cupy.cuda.Stream or
     * cuda.core Stream).
     */
    cudaStream_t to_stream(nb::handle stream);

    /**
     * @brief Returns the handle of a CUDA stream as a Python int.
     */
    inline std::uintptr_t from_stream(cudaStream_t stream)
    {
        return reinterpret_cast<std::uintptr_t>(stream);
    }

    /**
     * @brief Returns the ExecutionOptions given by a Python object: None (the
     * default options), an ExecutionOptions, or a stream (see to_stream).
     */
    heongpu::ExecutionOptions to_options(nb::handle options);

    // ------------------------------------------------------------------
    // Arrays
    // ------------------------------------------------------------------

    /**
     * @brief Whether the values (a sequence or an array) hold complex numbers.
     */
    bool is_complex(nb::handle values);

    /**
     * @brief Converts a scalar, a sequence or an array-like object to a
     * contiguous one-dimensional NumPy array of the given dtype (as
     * numpy.ascontiguousarray) and returns its elements. Raises TypeError if
     * the values cannot be converted, and ValueError if they are not
     * one-dimensional.
     */
    template <typename T>
    std::vector<T> to_vector(nb::handle values, const char* dtype,
                             const char* name);

    std::vector<double> to_double_vector(nb::handle values, const char* name);
    std::vector<Complex64> to_complex_vector(nb::handle values,
                                                      const char* name);
    std::vector<std::int64_t> to_int64_vector(nb::handle values,
                                              const char* name);
    std::vector<std::uint64_t> to_uint64_vector(nb::handle values,
                                                const char* name);
    std::vector<int> to_int_vector(nb::handle values, const char* name);

    /**
     * @brief Moves the values into a new one-dimensional NumPy array.
     */
    template <typename T>
    nb::ndarray<nb::numpy, T, nb::ndim<1>> to_numpy(std::vector<T>&& values)
    {
        auto* data = new std::vector<T>(std::move(values));
        nb::capsule owner(data, [](void* pointer) noexcept
                          { delete static_cast<std::vector<T>*>(pointer); });
        return nb::ndarray<nb::numpy, T, nb::ndim<1>>(data->data(),
                                                      {data->size()}, owner);
    }

    /**
     * @brief Moves complex values into a new one-dimensional NumPy array of
     * complex128.
     */
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>
    to_numpy(std::vector<Complex64>&& values);

    // ------------------------------------------------------------------
    // Serialization
    // ------------------------------------------------------------------

    /**
     * @brief Writes the bytes to a path (str or os.PathLike) or to a binary
     * file object (an object with a write method).
     */
    void write_file(nb::handle file, const std::string& data);

    /**
     * @brief Reads all bytes from a path (str or os.PathLike) or from a binary
     * file object (an object with a read method).
     */
    std::string read_file(nb::handle file);

    std::string bytes_to_string(nb::bytes data);

    template <typename T> std::string save_to_string(const T& object)
    {
        std::ostringstream stream(std::ios::binary);
        {
            nb::gil_scoped_release release;
            object.save(stream);
        }
        return stream.str();
    }

    template <typename T> void load_from_string(T& object, std::string data)
    {
        std::istringstream stream(std::move(data), std::ios::binary);
        nb::gil_scoped_release release;
        object.load(stream);
    }

    /**
     * @brief Adds the serialization methods of a class with save(ostream) and
     * load(istream): save(file), load(file) (static), to_bytes(),
     * from_bytes(data) (static) and pickle support. The data is the HEonGPU
     * serialization format, the same that the C++ save() writes, so objects
     * can be exchanged with C++ programs.
     */
    template <typename T, typename Class> void def_serialization(Class& cls)
    {
        cls.def(
               "to_bytes",
               [](const T& self)
               {
                   std::string data = save_to_string(self);
                   return nb::bytes(data.data(), data.size());
               },
               "Serializes the object to bytes in the HEonGPU format.")
            .def_static(
                "from_bytes",
                [](nb::bytes data)
                {
                    T object;
                    load_from_string(object, bytes_to_string(data));
                    return object;
                },
                "data"_a,
                "Deserializes an object from bytes written by to_bytes() or "
                "by the C++ save().")
            .def(
                "save",
                [](const T& self, nb::handle file)
                { write_file(file, save_to_string(self)); },
                "file"_a,
                "Writes the object to a path or to a binary file object.")
            .def_static(
                "load",
                [](nb::handle file)
                {
                    T object;
                    load_from_string(object, read_file(file));
                    return object;
                },
                "file"_a,
                "Reads an object from a path or from a binary file object.")
            .def("__getstate__",
                 [](const T& self)
                 {
                     std::string data = save_to_string(self);
                     return nb::bytes(data.data(), data.size());
                 })
            .def("__setstate__",
                 [](T& self, nb::bytes data)
                 {
                     new (&self) T();
                     load_from_string(self, bytes_to_string(data));
                 });
    }

    /**
     * @brief Adds the storage methods shared by keys, plaintexts and
     * ciphertexts: store_in_device(stream), store_in_host(stream) and the
     * is_on_device property.
     */
    template <typename T, typename Class> void def_storage(Class& cls)
    {
        cls.def(
               "store_in_device",
               [](T& self, nb::handle stream)
               {
                   cudaStream_t s = to_stream(stream);
                   nb::gil_scoped_release release;
                   self.store_in_device(s);
                   HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(s));
               },
               "stream"_a = nb::none(),
               "Moves the data to GPU memory.")
            .def(
                "store_in_host",
                [](T& self, nb::handle stream)
                {
                    cudaStream_t s = to_stream(stream);
                    nb::gil_scoped_release release;
                    // The copy into pinned host memory is asynchronous.
                    self.store_in_host(s);
                    HEONGPU_PY_CUDA_CHECK(cudaStreamSynchronize(s));
                },
                "stream"_a = nb::none(), "Moves the data to host memory.")
            .def_prop_ro("is_on_device", &T::is_on_device,
                         "Whether the data is in GPU memory.");
    }

    /**
     * @brief Adds the stream property and switch_stream(stream) for a class
     * whose data is on a CUDA stream.
     */
    template <typename T, typename Class> void def_stream(Class& cls)
    {
        cls.def_prop_ro(
               "stream", [](const T& self)
               { return from_stream(self.stream()); },
               "Handle of the CUDA stream of the data.")
            .def(
                "switch_stream",
                [](T& self, nb::handle stream)
                { self.switch_stream(to_stream(stream)); },
                nb::arg("stream").none(),
                "Uses another stream (None: the default stream) for the "
                "data.");
    }

    /**
     * @brief Adds __copy__ and __deepcopy__ for a class whose C++ copy
     * constructor copies the data.
     */
    template <typename T, typename Class> void def_copy(Class& cls)
    {
        // The C++ copy constructors copy device data asynchronously on the
        // stream of the source; the copy is complete when it is returned.
        auto copy = [](const T& self)
        {
            nb::gil_scoped_release release;
            T result(self);
            HEONGPU_PY_CUDA_CHECK(cudaDeviceSynchronize());
            return result;
        };
        cls.def("__copy__", copy)
            .def(
                "__deepcopy__",
                [copy](const T& self, nb::handle) { return copy(self); },
                "memo"_a)
            .def("copy", copy, "Returns a copy (the data is copied).");
    }

} // namespace heongpu_py

#endif // HEONGPU_PY_COMMON_H
