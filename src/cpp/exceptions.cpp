// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

namespace heongpu_py
{
    namespace
    {
        PyObject* error_type = nullptr;
        PyObject* parameter_error_type = nullptr;
        PyObject* state_error_type = nullptr;
        PyObject* context_mismatch_error_type = nullptr;
        PyObject* level_mismatch_error_type = nullptr;
        PyObject* key_not_found_error_type = nullptr;
        PyObject* serialization_error_type = nullptr;
        PyObject* system_error_type = nullptr;
        PyObject* cuda_error_type = nullptr;

        PyObject* add_exception(nb::module_& m, const char* name,
                                PyObject* base1, PyObject* base2,
                                const char* doc)
        {
            std::string qualified = std::string("heongpu.") + name;
            nb::object bases = base2 ? nb::make_tuple(nb::handle(base1),
                                                      nb::handle(base2))
                                     : nb::make_tuple(nb::handle(base1));
            PyObject* type = PyErr_NewExceptionWithDoc(
                qualified.c_str(), doc, bases.ptr(), nullptr);
            if (!type)
            {
                throw nb::python_error();
            }
            m.attr(name) = nb::handle(type);
            return type;
        }
    } // namespace

    void bind_exceptions(nb::module_& m)
    {
        error_type = add_exception(
            m, "Error", PyExc_Exception, nullptr,
            "Common base of the exceptions raised by HEonGPU.");
        parameter_error_type = add_exception(
            m, "ParameterError", error_type, PyExc_ValueError,
            "A parameter or an argument has an invalid value, for example a "
            "scale out of bounds or a ciphertext with the wrong size.");
        state_error_type = add_exception(
            m, "StateError", error_type, PyExc_ValueError,
            "An object is used before it is ready or after it can no longer "
            "change, for example a key that is not generated yet.");
        context_mismatch_error_type = add_exception(
            m, "ContextMismatchError", error_type, PyExc_ValueError,
            "An object created with one context is used with a different "
            "context.");
        level_mismatch_error_type = add_exception(
            m, "LevelMismatchError", error_type, PyExc_ValueError,
            "The operands of an operation are not at the levels it requires.");
        key_not_found_error_type = add_exception(
            m, "KeyNotFoundError", error_type, PyExc_ValueError,
            "A key needed by an operation is missing, for example a Galois "
            "key for a rotation step that was not generated.");
        serialization_error_type = add_exception(
            m, "SerializationError", error_type, PyExc_RuntimeError,
            "Serialized data is malformed, truncated or written for a "
            "different scheme.");
        system_error_type = add_exception(
            m, "SystemServiceError", error_type, PyExc_RuntimeError,
            "A system service HEonGPU depends on failed, for example the "
            "random number generator of the operating system.");
        cuda_error_type = add_exception(
            m, "CudaError", error_type, PyExc_RuntimeError,
            "A CUDA runtime call failed.");

        nb::register_exception_translator(
            [](const std::exception_ptr& pointer, void*)
            {
                try
                {
                    std::rethrow_exception(pointer);
                }
                catch (const heongpu::ParameterError& error)
                {
                    PyErr_SetString(parameter_error_type, error.what());
                }
                catch (const heongpu::StateError& error)
                {
                    PyErr_SetString(state_error_type, error.what());
                }
                catch (const heongpu::ContextMismatchError& error)
                {
                    PyErr_SetString(context_mismatch_error_type, error.what());
                }
                catch (const heongpu::LevelMismatchError& error)
                {
                    PyErr_SetString(level_mismatch_error_type, error.what());
                }
                catch (const heongpu::KeyNotFoundError& error)
                {
                    PyErr_SetString(key_not_found_error_type, error.what());
                }
                catch (const heongpu::SerializationError& error)
                {
                    PyErr_SetString(serialization_error_type, error.what());
                }
                catch (const heongpu::SystemError& error)
                {
                    PyErr_SetString(system_error_type, error.what());
                }
                catch (const heongpu::CudaException& error)
                {
                    PyErr_SetString(cuda_error_type, error.what());
                }
            });
    }

} // namespace heongpu_py
