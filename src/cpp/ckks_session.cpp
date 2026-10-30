// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

#include <cmath>
#include <exception>
#include <functional>
#include <optional>

namespace heongpu_py
{
    namespace
    {
        constexpr auto CKKS = heongpu::Scheme::CKKS;

        using Session = heongpu::HESession<CKKS>;
        using SessionCiphertext = heongpu::SessionCiphertext<CKKS>;
        using Params = heongpu::HEParams<CKKS>;
        using ParameterSet = heongpu::HEParameterSet<CKKS>;
        using Preset = heongpu::CKKSBootstrappingPreset;
        using Plaintext = heongpu::Plaintext<CKKS>;
        using Ciphertext = heongpu::Ciphertext<CKKS>;

        const char* role_name(heongpu::session_role role)
        {
            switch (role)
            {
                case heongpu::session_role::owner:
                    return "OWNER";
                case heongpu::session_role::evaluator:
                    return "EVALUATOR";
                case heongpu::session_role::encryptor:
                    return "ENCRYPTOR";
            }
            return "UNKNOWN";
        }

        std::vector<int> to_shifts(nb::handle shifts)
        {
            if (!shifts.is_valid() || shifts.is_none())
            {
                return {};
            }
            return to_int_vector(shifts, "rotations");
        }

        nb::bytes to_bytes_object(const std::string& data)
        {
            return nb::bytes(data.data(), data.size());
        }

        // ------------------------------------------------------------------
        // Operands of the Python operators of SessionCiphertext
        // ------------------------------------------------------------------

        /**
         * @brief The right operand of an arithmetic operator: another
         * ciphertext of the session, a real constant (added to or multiplied
         * with every slot) or real values, one per slot.
         */
        struct Operand
        {
            enum class Kind
            {
                ciphertext,
                scalar,
                values,
                unsupported
            };

            Kind kind = Kind::unsupported;
            const SessionCiphertext* ciphertext = nullptr;
            double scalar = 0.0;
            std::vector<double> values;
        };

        bool is_scalar(nb::handle value)
        {
            if (PyFloat_Check(value.ptr()) || PyLong_Check(value.ptr()))
            {
                return true;
            }
            // NumPy scalars and zero-dimensional arrays.
            if (nb::hasattr(value, "ndim") && nb::hasattr(value, "dtype"))
            {
                return nb::cast<int>(value.attr("ndim")) == 0;
            }
            return false;
        }

        Operand to_operand(nb::handle value)
        {
            Operand operand;
            if (nb::isinstance<SessionCiphertext>(value))
            {
                operand.kind = Operand::Kind::ciphertext;
                operand.ciphertext = nb::inst_ptr<SessionCiphertext>(value);
                return operand;
            }
            if (value.is_none() || nb::isinstance<nb::str>(value) ||
                nb::isinstance<nb::bytes>(value))
            {
                return operand;
            }
            if (PyComplex_Check(value.ptr()) || is_complex(value))
            {
                throw nb::type_error(
                    "The arithmetic of SessionCiphertext takes real "
                    "constants and values; encrypt complex values with "
                    "Session.encrypt() and use the ciphertext as the "
                    "operand.");
            }
            if (is_scalar(value))
            {
                operand.kind = Operand::Kind::scalar;
                operand.scalar = nb::cast<double>(nb::float_(value));
                return operand;
            }
            try
            {
                operand.values = to_double_vector(value, "values");
            }
            catch (const nb::builtin_exception& error)
            {
                // Not a number, a sequence or an array of numbers.
                if (error.type() != nb::exception_type::type_error)
                {
                    throw;
                }
                return operand;
            }
            operand.kind = Operand::Kind::values;
            return operand;
        }

        nb::object not_implemented()
        {
            return nb::borrow(Py_NotImplemented);
        }

        /**
         * @brief Applies one of three functions to self and the operand,
         * with the GIL released, and returns the result, or NotImplemented
         * for an unsupported operand.
         */
        template <typename WithCiphertext, typename WithScalar,
                  typename WithValues>
        nb::object apply(const SessionCiphertext& self, nb::handle other,
                         WithCiphertext with_ciphertext, WithScalar with_scalar,
                         WithValues with_values)
        {
            Operand operand = to_operand(other);
            SessionCiphertext result;
            switch (operand.kind)
            {
                case Operand::Kind::ciphertext:
                {
                    nb::gil_scoped_release release;
                    result = with_ciphertext(self, *operand.ciphertext);
                    break;
                }
                case Operand::Kind::scalar:
                {
                    nb::gil_scoped_release release;
                    result = with_scalar(self, operand.scalar);
                    break;
                }
                case Operand::Kind::values:
                {
                    nb::gil_scoped_release release;
                    result = with_values(self, operand.values);
                    break;
                }
                default:
                    return not_implemented();
            }
            return nb::cast(std::move(result));
        }

        /**
         * @brief The same as apply(), but modifies self and returns it.
         */
        template <typename WithCiphertext, typename WithScalar,
                  typename WithValues>
        nb::object apply_inplace(nb::handle self_object, nb::handle other,
                                 WithCiphertext with_ciphertext,
                                 WithScalar with_scalar, WithValues with_values)
        {
            SessionCiphertext& self =
                *nb::inst_ptr<SessionCiphertext>(self_object);
            Operand operand = to_operand(other);
            switch (operand.kind)
            {
                case Operand::Kind::ciphertext:
                {
                    nb::gil_scoped_release release;
                    with_ciphertext(self, *operand.ciphertext);
                    break;
                }
                case Operand::Kind::scalar:
                {
                    nb::gil_scoped_release release;
                    with_scalar(self, operand.scalar);
                    break;
                }
                case Operand::Kind::values:
                {
                    nb::gil_scoped_release release;
                    with_values(self, operand.values);
                    break;
                }
                default:
                    return not_implemented();
            }
            return nb::borrow(self_object);
        }

        /**
         * @brief input to the power exponent by repeated squaring.
         */
        SessionCiphertext power(const SessionCiphertext& input, int exponent)
        {
            std::optional<SessionCiphertext> result;
            SessionCiphertext base = input;
            while (true)
            {
                if (exponent & 1)
                {
                    result = result ? *result * base : base;
                }
                exponent >>= 1;
                if (exponent == 0)
                {
                    break;
                }
                base = base * base;
            }
            return *result;
        }

        // ------------------------------------------------------------------
        // Session helpers
        // ------------------------------------------------------------------

        SessionCiphertext encrypt_values(const Session& self, nb::handle values)
        {
            if (nb::isinstance<Plaintext>(values))
            {
                const Plaintext& plain = nb::cast<const Plaintext&>(values);
                nb::gil_scoped_release release;
                return self.encrypt(plain);
            }
            bool scalar = is_scalar(values) || PyComplex_Check(values.ptr());
            int slots = self.slot_count();
            if (is_complex(values))
            {
                std::vector<Complex64> message =
                    to_complex_vector(values, "values");
                if (scalar)
                {
                    message.resize(slots, message[0]);
                }
                nb::gil_scoped_release release;
                return self.encrypt(message);
            }
            std::vector<double> message = to_double_vector(values, "values");
            if (scalar)
            {
                message.resize(slots, message[0]);
            }
            nb::gil_scoped_release release;
            return self.encrypt(message);
        }

        SessionCiphertext evaluate_chebyshev(const Session& self,
                                             const SessionCiphertext& input,
                                             nb::handle function, double a,
                                             double b,
                                             std::optional<int> degree)
        {
            if (!PyCallable_Check(function.ptr()))
            {
                if (degree)
                {
                    throw nb::value_error("degree is only used when the "
                                          "function is a callable; the "
                                          "coefficients give the degree.");
                }
                std::vector<double> coefficients =
                    to_double_vector(function, "coefficients");
                nb::gil_scoped_release release;
                return self.evaluate_chebyshev(input, coefficients, a, b);
            }
            if (!degree)
            {
                throw nb::type_error("evaluate_chebyshev() needs the degree "
                                     "of the approximation of a callable.");
            }

            // The function is called from C++ with the GIL released: it
            // takes the GIL, and its first error is raised afterwards.
            std::exception_ptr error;
            std::function<double(double)> wrapped = [function,
                                                     &error](double x) -> double
            {
                nb::gil_scoped_acquire acquire;
                if (error)
                {
                    return 0.0;
                }
                try
                {
                    return nb::cast<double>(nb::float_(function(x)));
                }
                catch (...)
                {
                    error = std::current_exception();
                    return 0.0;
                }
            };
            SessionCiphertext result;
            {
                nb::gil_scoped_release release;
                result = self.evaluate_chebyshev(input, wrapped, a, b, *degree);
            }
            if (error)
            {
                std::rethrow_exception(error);
            }
            return result;
        }

        std::string save_session(const Session& self,
                                 heongpu::session_role role)
        {
            std::ostringstream stream(std::ios::binary);
            {
                nb::gil_scoped_release release;
                self.save(stream, role);
            }
            return stream.str();
        }

        Session load_session(std::string data,
                             std::optional<heongpu::session_role> expected)
        {
            std::istringstream stream(std::move(data), std::ios::binary);
            nb::gil_scoped_release release;
            if (expected)
            {
                return Session::load(stream, *expected);
            }
            return Session::load(stream);
        }

        SessionCiphertext load_ciphertext(const Session& self, std::string data)
        {
            std::istringstream stream(std::move(data), std::ios::binary);
            nb::gil_scoped_release release;
            return self.load_ciphertext(stream);
        }

        /**
         * @brief A gate bootstrapping method of Session.
         */
        template <SessionCiphertext (Session::*Gate)(
            const SessionCiphertext&, const SessionCiphertext&) const>
        SessionCiphertext gate(const Session& self,
                               const SessionCiphertext& input1,
                               const SessionCiphertext& input2)
        {
            nb::gil_scoped_release release;
            return (self.*Gate)(input1, input2);
        }
    } // namespace

    void bind_ckks_session(nb::module_& m)
    {
        // --------------------------------------------------------------
        // SessionCiphertext
        // --------------------------------------------------------------

        nb::class_<SessionCiphertext> ciphertext(
            m, "SessionCiphertext",
            "A ciphertext of a Session. The operators +, -, * (and +=, -=, "
            "*=) take another ciphertext of the same session, a real "
            "constant, which applies to every slot, or real values, one per "
            "slot (the other slots are 0). A product of two ciphertexts is "
            "relinearized; in the automatic rescale mode the operands are "
            "also rescaled and brought to the same level. The ciphertext "
            "keeps the state of its session alive.");

        // NumPy arrays on the left of an operator defer to the ciphertext.
        ciphertext.attr("__array_ufunc__") = nb::none();

        ciphertext
            .def(nb::init<>(),
                 "An empty ciphertext; operations on it raise StateError.")
            .def_prop_ro(
                "ciphertext",
                [](SessionCiphertext& self) -> Ciphertext&
                { return self.ciphertext(); },
                nb::rv_policy::reference_internal,
                "The underlying Ciphertext of the lower level API (a "
                "reference, not a copy).")
            .def_prop_ro("level", &SessionCiphertext::level,
                         "The number of rescales left.")
            .def_prop_ro("depth", &SessionCiphertext::depth,
                         "The number of rescales applied.")
            .def_prop_ro("scale", &SessionCiphertext::scale)
            .def_prop_ro(
                "stream",
                [](const SessionCiphertext& self)
                { return from_stream(self.stream()); },
                "Handle of the CUDA stream of the operations whose left "
                "operand is this ciphertext.")
            .def_prop_ro("levels_left", &SessionCiphertext::levels_left,
                         "The levels left before the ciphertext needs a "
                         "bootstrapping: its level, one less if it waits "
                         "for a rescale, minus the input level of the "
                         "bootstrapping of the session. Below 0, the slim, "
                         "bit and gate bootstrappings no longer take it.")
            .def(
                "bootstrap",
                [](const SessionCiphertext& self)
                {
                    nb::gil_scoped_release release;
                    return self.bootstrap();
                },
                "Returns the ciphertext bootstrapped with the regular, slim "
                "or bit bootstrapping of the session, at the top level. "
                "Raises StateError for a session without bootstrapping or "
                "with a gate bootstrapping (see Session.AND_bootstrapping).")
            .def(
                "rotate",
                [](const SessionCiphertext& self, int shift)
                {
                    nb::gil_scoped_release release;
                    return self.rotate(shift);
                },
                "shift"_a,
                "Returns the slots rotated left by shift (right for a "
                "negative shift). The Galois key of a new shift is generated "
                "the first time it is used, if the session has the secret "
                "key.")
            .def(
                "conjugate",
                [](const SessionCiphertext& self)
                {
                    nb::gil_scoped_release release;
                    return self.conjugate();
                },
                "Returns the complex conjugate of the slots.")
            .def(
                "rotate_hoisted",
                [](const SessionCiphertext& self, nb::handle shifts)
                {
                    std::vector<int> steps = to_int_vector(shifts, "shifts");
                    nb::gil_scoped_release release;
                    return self.rotate_hoisted(steps);
                },
                "shifts"_a,
                "Returns a list with the ciphertext rotated by each shift; "
                "the rotations share the decomposition of the ciphertext.")
            .def(
                "sum_slots",
                [](const SessionCiphertext& self)
                {
                    nb::gil_scoped_release release;
                    return self.sum_slots();
                },
                "Returns a ciphertext whose slots all hold the sum of the "
                "slots.")
            .def(
                "inner_product",
                [](const SessionCiphertext& self, nb::handle other)
                {
                    if (nb::isinstance<SessionCiphertext>(other))
                    {
                        const SessionCiphertext& operand =
                            nb::cast<const SessionCiphertext&>(other);
                        nb::gil_scoped_release release;
                        return self.inner_product(operand);
                    }
                    if (is_complex(other))
                    {
                        throw nb::type_error("inner_product() takes real "
                                             "values.");
                    }
                    std::vector<double> values =
                        to_double_vector(other, "other");
                    nb::gil_scoped_release release;
                    return self.inner_product(values);
                },
                "other"_a,
                "Returns a ciphertext whose slots all hold the inner product "
                "of the slots with those of another ciphertext, or with real "
                "values, one per slot.")
            .def(
                "save",
                [](const SessionCiphertext& self, nb::handle file)
                { write_file(file, save_to_string(self)); },
                "file"_a,
                "Writes the ciphertext to a path or to a binary file object; "
                "Session.load_ciphertext() of a session with the same keys "
                "reads it.")
            .def(
                "to_bytes",
                [](const SessionCiphertext& self)
                { return to_bytes_object(save_to_string(self)); },
                "Serializes the ciphertext to bytes in the HEonGPU format; "
                "Session.ciphertext_from_bytes() reads it.")
            .def("__reduce_ex__",
                 [](const SessionCiphertext&, nb::handle) -> nb::object
                 {
                     throw nb::type_error(
                         "A SessionCiphertext cannot be pickled without its "
                         "session; use to_bytes() and "
                         "Session.ciphertext_from_bytes().");
                 })
            .def("__getstate__",
                 [](const SessionCiphertext&) -> nb::object
                 {
                     throw nb::type_error(
                         "A SessionCiphertext cannot be pickled without its "
                         "session; use to_bytes() and "
                         "Session.ciphertext_from_bytes().");
                 })
            .def("__repr__",
                 [](const SessionCiphertext& self)
                 {
                     std::ostringstream stream;
                     stream
                         << "<heongpu.ckks.SessionCiphertext level="
                         << self.level() << " depth=" << self.depth()
                         << " scale=2^"
                         << (self.scale() > 0 ? std::log2(self.scale()) : 0.0)
                         << ">";
                     return stream.str();
                 });
        def_copy<SessionCiphertext>(ciphertext);

        using SC = SessionCiphertext;
        using Values = std::vector<double>;

        ciphertext
            .def(
                "__add__",
                [](const SC& self, nb::handle other)
                {
                    return apply(
                        self, other,
                        [](const SC& a, const SC& b) { return a + b; },
                        [](const SC& a, double b) { return a + b; },
                        [](const SC& a, const Values& b) { return a + b; });
                },
                nb::is_operator())
            .def(
                "__radd__",
                [](const SC& self, nb::handle other)
                {
                    return apply(
                        self, other,
                        [](const SC& a, const SC& b) { return b + a; },
                        [](const SC& a, double b) { return b + a; },
                        [](const SC& a, const Values& b) { return b + a; });
                },
                nb::is_operator())
            .def(
                "__sub__",
                [](const SC& self, nb::handle other)
                {
                    return apply(
                        self, other,
                        [](const SC& a, const SC& b) { return a - b; },
                        [](const SC& a, double b) { return a - b; },
                        [](const SC& a, const Values& b) { return a - b; });
                },
                nb::is_operator())
            .def(
                "__rsub__",
                [](const SC& self, nb::handle other)
                {
                    return apply(
                        self, other,
                        [](const SC& a, const SC& b) { return b - a; },
                        [](const SC& a, double b) { return b - a; },
                        [](const SC& a, const Values& b) { return b - a; });
                },
                nb::is_operator())
            .def(
                "__mul__",
                [](const SC& self, nb::handle other)
                {
                    return apply(
                        self, other,
                        [](const SC& a, const SC& b) { return a * b; },
                        [](const SC& a, double b) { return a * b; },
                        [](const SC& a, const Values& b) { return a * b; });
                },
                nb::is_operator())
            .def(
                "__rmul__",
                [](const SC& self, nb::handle other)
                {
                    return apply(
                        self, other,
                        [](const SC& a, const SC& b) { return b * a; },
                        [](const SC& a, double b) { return b * a; },
                        [](const SC& a, const Values& b) { return b * a; });
                },
                nb::is_operator())
            .def(
                "__truediv__",
                [](const SC& self, nb::handle other) -> nb::object
                {
                    if (!is_scalar(other))
                    {
                        return not_implemented();
                    }
                    double value = nb::cast<double>(nb::float_(other));
                    if (value == 0.0)
                    {
                        PyErr_SetString(PyExc_ZeroDivisionError,
                                        "division of a SessionCiphertext "
                                        "by zero");
                        throw nb::python_error();
                    }
                    SC result;
                    {
                        nb::gil_scoped_release release;
                        result = self * (1.0 / value);
                    }
                    return nb::cast(std::move(result));
                },
                nb::is_operator(),
                "Multiplies by the inverse of a real constant.")
            .def(
                "__pow__",
                [](const SC& self, int exponent)
                {
                    if (exponent < 1)
                    {
                        throw nb::value_error("The exponent of a "
                                              "SessionCiphertext must be a "
                                              "positive integer.");
                    }
                    nb::gil_scoped_release release;
                    return power(self, exponent);
                },
                nb::is_operator(), "exponent"_a,
                "Raises to a positive integer power by repeated squaring "
                "(ceil(log2(exponent)) levels).")
            .def(
                "__neg__",
                [](const SC& self)
                {
                    nb::gil_scoped_release release;
                    return -self;
                },
                nb::is_operator())
            .def(
                "__pos__", [](const SC& self) { return SC(self); },
                nb::is_operator())
            .def(
                "__iadd__",
                [](nb::handle self, nb::handle other)
                {
                    return apply_inplace(
                        self, other, [](SC& a, const SC& b) { a += b; },
                        [](SC& a, double b) { a += b; },
                        [](SC& a, const Values& b) { a += b; });
                },
                nb::is_operator())
            .def(
                "__isub__",
                [](nb::handle self, nb::handle other)
                {
                    return apply_inplace(
                        self, other, [](SC& a, const SC& b) { a -= b; },
                        [](SC& a, double b) { a -= b; },
                        [](SC& a, const Values& b) { a -= b; });
                },
                nb::is_operator())
            .def(
                "__imul__",
                [](nb::handle self, nb::handle other)
                {
                    return apply_inplace(
                        self, other, [](SC& a, const SC& b) { a *= b; },
                        [](SC& a, double b) { a *= b; },
                        [](SC& a, const Values& b) { a *= b; });
                },
                nb::is_operator());

        // --------------------------------------------------------------
        // Session
        // --------------------------------------------------------------

        nb::class_<Session> session(
            m, "Session",
            "One party's CKKS computation in one object: the context, the "
            "keys, the encoder, the encryptor, the decryptor and the "
            "operators. A session created from Params, a ParameterSet or a "
            "BootstrappingPreset owns all keys; save() writes it with the "
            "keys of a role and load() reads it, for example for a server "
            "that computes without the secret key. Views from on_stream() "
            "share the keys and the state, and threads can use them at the "
            "same time.");

        session
            .def(
                "__init__",
                [](Session* self, const Params& params, nb::handle rotations)
                {
                    std::vector<int> shifts = to_shifts(rotations);
                    nb::gil_scoped_release release;
                    new (self) Session(params, shifts);
                },
                "params"_a, "rotations"_a = nb::none(),
                "Chooses the parameters with params.generate(), generates "
                "the context and the secret, public and relinearization "
                "keys, and the Galois keys for the shifts in rotations. The "
                "rescale mode is params.rescale_mode, or AUTOMATIC if it is "
                "not set.")
            .def(
                "__init__",
                [](Session* self, const ParameterSet& parameters,
                   nb::handle rotations)
                {
                    std::vector<int> shifts = to_shifts(rotations);
                    nb::gil_scoped_release release;
                    new (self) Session(parameters, shifts);
                },
                "parameters"_a, "rotations"_a = nb::none(),
                "Generates the context of a ParameterSet, with its rescale "
                "mode, and the keys.")
            .def(
                "__init__",
                [](Session* self, const Preset& preset, nb::handle rotations)
                {
                    std::vector<int> shifts = to_shifts(rotations);
                    nb::gil_scoped_release release;
                    new (self) Session(preset, shifts);
                },
                "preset"_a, "rotations"_a = nb::none(),
                "Generates the context of a BootstrappingPreset in the "
                "automatic rescale mode, with the level of the bootstrapping "
                "results as the top level, the secret key of the preset, the "
                "public key, the bootstrapping keys and the Galois keys for "
                "the shifts in rotations.");

        session.def(
            "__init__",
            [](Session* self, int multiplicative_depth, int scale_bit_size,
               int first_modulus_bit_size, std::size_t poly_modulus_degree,
               heongpu::sec_level_type security_level,
               std::optional<heongpu::rescale_mode> rescale_mode,
               nb::handle rotations)
            {
                Params params;
                params.multiplicative_depth = multiplicative_depth;
                params.scale_bit_size = scale_bit_size;
                params.first_modulus_bit_size = first_modulus_bit_size;
                params.poly_modulus_degree = poly_modulus_degree;
                params.security_level = security_level;
                params.rescale_mode = rescale_mode;
                std::vector<int> shifts = to_shifts(rotations);
                nb::gil_scoped_release release;
                new (self) Session(params, shifts);
            },
            nb::kw_only(), "multiplicative_depth"_a, "scale_bit_size"_a = 40,
            "first_modulus_bit_size"_a = 60, "poly_modulus_degree"_a = 0,
            "security_level"_a = heongpu::sec_level_type::sec128,
            "rescale_mode"_a = nb::none(), "rotations"_a = nb::none(),
            "The same as Session(Params(...), rotations) with the given "
            "requirements.");

        session
            .def("encrypt", &encrypt_values, "values"_a,
                 "Encrypts real or complex values (a sequence or an array, "
                 "one value per slot, the other slots are 0; a scalar fills "
                 "every slot) at the top level with the scale of the "
                 "parameters, or encrypts a Plaintext.")
            .def(
                "wrap",
                [](const Session& self, const Ciphertext& cipher)
                {
                    nb::gil_scoped_release release;
                    return self.wrap(cipher);
                },
                "ciphertext"_a,
                "Returns a copy of a Ciphertext as a ciphertext of the "
                "session. Raises ContextMismatchError unless it has the "
                "parameters of the session; it must also be encrypted with "
                "the keys of the session.")
            .def(
                "decrypt",
                [](const Session& self, const SessionCiphertext& input)
                {
                    std::vector<double> values;
                    {
                        nb::gil_scoped_release release;
                        values = self.decrypt(input);
                    }
                    return to_numpy(std::move(values));
                },
                "ciphertext"_a,
                "Returns the real parts of all slots as a float64 NumPy "
                "array. Raises StateError for a session without the secret "
                "key.")
            .def(
                "decrypt_complex",
                [](const Session& self, const SessionCiphertext& input)
                {
                    std::vector<Complex64> values;
                    {
                        nb::gil_scoped_release release;
                        values = self.decrypt_complex(input);
                    }
                    return to_numpy(std::move(values));
                },
                "ciphertext"_a,
                "Returns all slots as a complex128 NumPy array.")
            .def(
                "evaluate_polynomial",
                [](const Session& self, const SessionCiphertext& input,
                   nb::handle coefficients)
                {
                    std::vector<double> values =
                        to_double_vector(coefficients, "coefficients");
                    nb::gil_scoped_release release;
                    return self.evaluate_polynomial(input, values);
                },
                "ciphertext"_a, "coefficients"_a,
                "Evaluates the polynomial with the given coefficients, "
                "constant term first, on every slot.")
            .def("evaluate_chebyshev", &evaluate_chebyshev, "ciphertext"_a,
                 "function"_a, "a"_a, "b"_a, "degree"_a = nb::none(),
                 "Evaluates a function on every slot, whose values are in "
                 "[a, b], with a Chebyshev series: function is either the "
                 "coefficients of the series, or a callable from float to "
                 "float that is interpolated with the given degree.")
            .def("AND_bootstrapping", &gate<&Session::AND_bootstrapping>,
                 "input1"_a, "input2"_a,
                 "Returns AND of two ciphertexts of bits with the gate "
                 "bootstrapping of the session, at the top level. Raises "
                 "StateError unless the session has a gate bootstrapping "
                 "preset.")
            .def("OR_bootstrapping", &gate<&Session::OR_bootstrapping>,
                 "input1"_a, "input2"_a,
                 "Returns OR of two ciphertexts of bits with the gate "
                 "bootstrapping of the session.")
            .def("XOR_bootstrapping", &gate<&Session::XOR_bootstrapping>,
                 "input1"_a, "input2"_a,
                 "Returns XOR of two ciphertexts of bits with the gate "
                 "bootstrapping of the session.")
            .def("NAND_bootstrapping", &gate<&Session::NAND_bootstrapping>,
                 "input1"_a, "input2"_a,
                 "Returns NAND of two ciphertexts of bits with the gate "
                 "bootstrapping of the session.")
            .def("NOR_bootstrapping", &gate<&Session::NOR_bootstrapping>,
                 "input1"_a, "input2"_a,
                 "Returns NOR of two ciphertexts of bits with the gate "
                 "bootstrapping of the session.")
            .def("XNOR_bootstrapping", &gate<&Session::XNOR_bootstrapping>,
                 "input1"_a, "input2"_a,
                 "Returns XNOR of two ciphertexts of bits with the gate "
                 "bootstrapping of the session.")
            .def_prop_ro("has_bootstrapping", &Session::has_bootstrapping,
                         "Whether the session was created from a "
                         "BootstrappingPreset, or loaded from one with the "
                         "keys of the owner or the evaluator.")
            .def("bootstrapping_levels", &Session::bootstrapping_levels,
                 "Returns the levels of the bootstrapping of the session. "
                 "Raises StateError for a session without bootstrapping.")
            .def(
                "save",
                [](const Session& self, nb::handle file,
                   heongpu::session_role role)
                { write_file(file, save_session(self, role)); },
                "file"_a, "role"_a,
                "Writes the session with the keys of a role to a path or to "
                "a binary file object: the context, the rescale mode and the "
                "public key; for EVALUATOR also the relinearization, Galois "
                "and bootstrapping keys; for OWNER also the secret key. "
                "Raises StateError if the session does not have them.")
            .def(
                "to_bytes",
                [](const Session& self, heongpu::session_role role)
                { return to_bytes_object(save_session(self, role)); },
                "role"_a,
                "Serializes the session with the keys of a role, as save() "
                "does.")
            .def_static(
                "load",
                [](nb::handle file,
                   std::optional<heongpu::session_role> expected_role)
                { return load_session(read_file(file), expected_role); },
                "file"_a, "expected_role"_a = nb::none(),
                "Reads a session that save() wrote from a path or a binary "
                "file object. With expected_role, raises StateError, before "
                "reading any key, unless the session was saved with that "
                "role.")
            .def_static(
                "from_bytes",
                [](nb::bytes data,
                   std::optional<heongpu::session_role> expected_role)
                { return load_session(bytes_to_string(data), expected_role); },
                "data"_a, "expected_role"_a = nb::none(),
                "Reads a session that to_bytes() wrote; see load().")
            .def(
                "load_ciphertext",
                [](const Session& self, nb::handle file)
                { return load_ciphertext(self, read_file(file)); },
                "file"_a,
                "Reads a ciphertext that SessionCiphertext.save() or "
                "Ciphertext.save() wrote, from a path or a binary file "
                "object. It must have the parameters of the session.")
            .def(
                "ciphertext_from_bytes",
                [](const Session& self, nb::bytes data)
                { return load_ciphertext(self, bytes_to_string(data)); },
                "data"_a,
                "Reads a ciphertext that SessionCiphertext.to_bytes() or "
                "Ciphertext.to_bytes() wrote.")
            .def_prop_ro("role", &Session::role,
                         "The keys of the session: OWNER has the secret key, "
                         "EVALUATOR the evaluation keys, ENCRYPTOR only the "
                         "public key.")
            .def(
                "add_rotations",
                [](const Session& self, nb::handle shifts)
                {
                    std::vector<int> steps = to_int_vector(shifts, "shifts");
                    nb::gil_scoped_release release;
                    self.add_rotations(steps);
                },
                "shifts"_a,
                "Generates the Galois keys for the shifts the session has no "
                "key for yet. Raises KeyNotFoundError for a session without "
                "the secret key.")
            .def(
                "on_stream",
                [](const Session& self, nb::handle stream)
                {
                    cudaStream_t s = to_stream(stream);
                    nb::gil_scoped_release release;
                    return self.on_stream(s);
                },
                "stream"_a, nb::keep_alive<0, 2>(),
                "Returns a view of the session that runs on a CUDA stream "
                "(a heongpu.Stream, any stream object or an int handle): it "
                "shares the keys and the state, and its ciphertexts and the "
                "results of operations whose left operand is one of them "
                "belong to the stream. Waits for the work queued on the "
                "stream of this session first. A ciphertext of another "
                "stream must be ready before it is used: call synchronize() "
                "on its session.")
            .def_prop_ro(
                "stream",
                [](const Session& self) { return from_stream(self.stream()); },
                "Handle of the CUDA stream of the session (0 is the default "
                "stream).")
            .def(
                "synchronize",
                [](const Session& self)
                {
                    nb::gil_scoped_release release;
                    self.synchronize();
                },
                "Waits for the work queued on the stream of the session.")
            .def_prop_ro("slot_count", &Session::slot_count)
            .def_prop_ro("scale", &Session::scale,
                         "The scale of the parameters, which encrypt() "
                         "uses.")
            .def_prop_ro("rescale_mode", &Session::get_rescale_mode)
            .def("context", &Session::context,
                 "Returns the Context of the session.")
            .def("secret_key", &Session::secret_key,
                 nb::rv_policy::reference_internal,
                 "Returns the SecretKey of the session (a reference). Raises "
                 "StateError for a session without it.")
            .def("public_key", &Session::public_key,
                 nb::rv_policy::reference_internal,
                 "Returns the PublicKey of the session (a reference).")
            .def("relin_key", &Session::relin_key,
                 nb::rv_policy::reference_internal,
                 "Returns the RelinKey of the session (a reference). Raises "
                 "KeyNotFoundError for a session without it.")
            .def(
                "galois_key",
                [](const Session& self) -> heongpu::Galoiskey<CKKS>&
                {
                    nb::gil_scoped_release release;
                    return self.galois_key();
                },
                nb::rv_policy::reference_internal,
                "Returns the GaloisKey with the keys generated so far and "
                "the conjugation key (a reference). Rotations by new shifts "
                "add keys to it: do not use it while other threads rotate "
                "with the session.")
            .def("encoder", &Session::encoder,
                 nb::rv_policy::reference_internal,
                 "Returns the Encoder of the session (a reference).")
            .def("encryptor", &Session::encryptor,
                 nb::rv_policy::reference_internal,
                 "Returns the Encryptor of the session (a reference).")
            .def("decryptor", &Session::decryptor,
                 nb::rv_policy::reference_internal,
                 "Returns the Decryptor of the session (a reference). Raises "
                 "StateError for a session without the secret key.")
            .def("operators", &Session::operators,
                 nb::rv_policy::reference_internal,
                 "Returns the ArithmeticOperator of the session (a "
                 "reference), for example to rescale in the manual rescale "
                 "mode.")
            .def("__reduce_ex__",
                 [](const Session&, nb::handle) -> nb::object
                 {
                     throw nb::type_error(
                         "A Session cannot be pickled, since it would have "
                         "to choose the keys to include; use "
                         "to_bytes(role) and Session.from_bytes().");
                 })
            .def("__repr__",
                 [](const Session& self)
                 {
                     std::ostringstream stream;
                     stream << "<heongpu.ckks.Session role="
                            << role_name(self.role())
                            << " slot_count=" << self.slot_count()
                            << " rescale_mode="
                            << (self.get_rescale_mode() ==
                                        heongpu::rescale_mode::automatic
                                    ? "AUTOMATIC"
                                    : "MANUAL")
                            << " bootstrapping="
                            << (self.has_bootstrapping() ? "True" : "False")
                            << ">";
                     return stream.str();
                 });

        // --------------------------------------------------------------
        // Logic gates computed with the arithmetic of the session
        // --------------------------------------------------------------

        m.def(
            "NOT",
            [](const SC& a)
            {
                nb::gil_scoped_release release;
                return heongpu::NOT(a);
            },
            "a"_a,
            "Returns 1 - a for a ciphertext of bits (0 or 1 in each slot).");

        auto def_gate = [&m](const char* name, auto function, const char* doc)
        {
            m.def(
                name,
                [function](const SC& a, const SC& b)
                {
                    nb::gil_scoped_release release;
                    return function(a, b);
                },
                "a"_a, "b"_a, doc);
        };
        def_gate(
            "AND", [](const SC& a, const SC& b) { return heongpu::AND(a, b); },
            "Returns a AND b = a b for ciphertexts of bits; one level.");
        def_gate(
            "OR", [](const SC& a, const SC& b) { return heongpu::OR(a, b); },
            "Returns a OR b = a + b - a b for ciphertexts of bits; one level.");
        def_gate(
            "XOR", [](const SC& a, const SC& b) { return heongpu::XOR(a, b); },
            "Returns a XOR b = a + b - 2 a b for ciphertexts of bits; one "
            "level.");
        def_gate(
            "NAND",
            [](const SC& a, const SC& b) { return heongpu::NAND(a, b); },
            "Returns NOT (a AND b) for ciphertexts of bits; one level.");
        def_gate(
            "NOR", [](const SC& a, const SC& b) { return heongpu::NOR(a, b); },
            "Returns NOT (a OR b) for ciphertexts of bits; one level.");
        def_gate(
            "XNOR",
            [](const SC& a, const SC& b) { return heongpu::XNOR(a, b); },
            "Returns NOT (a XOR b) for ciphertexts of bits; one level.");
    }

} // namespace heongpu_py
