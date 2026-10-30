// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"
#include "generated.h"

#include <optional>

namespace heongpu_py
{
    namespace
    {
        constexpr auto BFV = heongpu::Scheme::BFV;

        using Params = heongpu::HEParams<BFV>;
        using ParameterSet = heongpu::HEParameterSet<BFV>;
        using Plaintext = heongpu::Plaintext<BFV>;
        using Ciphertext = heongpu::Ciphertext<BFV>;
        using Session = heongpu::HESession<BFV>;
        using SessionCiphertextCpp = heongpu::SessionCiphertext<BFV>;
        using Role = heongpu::session_role;

        /**
         * @brief The Python SessionCiphertext: a SessionCiphertext<BFV> with
         * the plaintext modulus of its session, which the operators need to
         * reduce Python integers (of any size and sign) before they are
         * encoded. The modulus is 0 for an empty ciphertext.
         */
        struct SessionCipher
        {
            SessionCiphertextCpp value;
            std::uint64_t plain_modulus = 0;
        };

        // The plaintext modulus of a, or StateError if a is empty.
        std::uint64_t modulus_of(const SessionCipher& a)
        {
            if (a.plain_modulus == 0)
            {
                throw heongpu::StateError(
                    "The SessionCiphertext is empty! Assign it a ciphertext "
                    "of a Session first.");
            }
            return a.plain_modulus;
        }

        const char* role_name(Role role)
        {
            switch (role)
            {
                case Role::owner:
                    return "OWNER";
                case Role::evaluator:
                    return "EVALUATOR";
                case Role::encryptor:
                    return "ENCRYPTOR";
            }
            return "UNKNOWN";
        }

        /**
         * @brief Integer values reduced modulo the plaintext modulus: a
         * scalar (one value) or a vector of slot values.
         */
        struct IntValues
        {
            bool scalar = false;
            std::vector<std::uint64_t> values;
        };

        /**
         * @brief Converts an integer, a sequence or an array of integers (any
         * integer or bool dtype, also Python ints of any size) to residues
         * modulo t. Raises TypeError for non-integer values and ValueError
         * for arrays of more than one dimension.
         */
        IntValues int_values(nb::handle values, std::uint64_t t)
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
            IntValues result;
            result.scalar = (dimensions == 0);
            if (nb::cast<std::size_t>(array.attr("size")) == 0)
            {
                return result;
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
                    array.attr("astype")("uint64"), numpy.attr("uint64")(t));
            }
            else if (kind == "i")
            {
                reduced = numpy.attr("remainder")(array.attr("astype")("int64"),
                                                  numpy.attr("int64")(t));
            }
            else if (kind == "O")
            {
                // Python ints that do not fit a NumPy integer type.
                nb::object index =
                    nb::module_::import_("operator").attr("index");
                nb::object items = array.attr("reshape")(-1).attr("tolist")();
                nb::list residues;
                nb::int_ modulus(t);
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
            result.values = to_uint64_vector(reduced, "values");
            return result;
        }

        // The slot values of values for a ciphertext with slot_count slots:
        // a scalar is repeated in every slot.
        std::vector<std::uint64_t>
        slot_values(nb::handle values, std::uint64_t t, std::size_t slot_count)
        {
            IntValues converted = int_values(values, t);
            if (converted.scalar)
            {
                std::uint64_t value =
                    converted.values.empty() ? 0 : converted.values[0];
                return std::vector<std::uint64_t>(slot_count, value);
            }
            return std::move(converted.values);
        }

        enum class BinaryOp
        {
            add,
            sub,
            rsub,
            multiply
        };

        /**
         * @brief Returns a op other (other op a for rsub), where other is a
         * SessionCiphertext, an integer or a vector of integers; nothing if
         * other is none of them, for the operators to return NotImplemented.
         */
        std::optional<SessionCiphertextCpp> apply(const SessionCipher& a,
                                                  nb::handle other, BinaryOp op)
        {
            std::uint64_t t = modulus_of(a);
            if (nb::isinstance<SessionCipher>(other))
            {
                const SessionCipher& b = nb::cast<const SessionCipher&>(other);
                nb::gil_scoped_release release;
                switch (op)
                {
                    case BinaryOp::add:
                        return a.value + b.value;
                    case BinaryOp::sub:
                        return a.value - b.value;
                    case BinaryOp::rsub:
                        return b.value - a.value;
                    case BinaryOp::multiply:
                        return a.value * b.value;
                }
                return std::nullopt;
            }

            IntValues operand;
            try
            {
                operand = int_values(other, t);
            }
            catch (nb::builtin_exception& error)
            {
                if (error.type() != nb::exception_type::type_error)
                {
                    throw;
                }
                return std::nullopt;
            }

            if (operand.scalar)
            {
                // A residue modulo t, so it fits an int64 and encodes as is.
                auto value = static_cast<std::int64_t>(
                    operand.values.empty() ? 0 : operand.values[0]);
                nb::gil_scoped_release release;
                switch (op)
                {
                    case BinaryOp::add:
                        return a.value + value;
                    case BinaryOp::sub:
                        return a.value - value;
                    case BinaryOp::rsub:
                        return value - a.value;
                    case BinaryOp::multiply:
                        return a.value * value;
                }
                return std::nullopt;
            }

            const std::vector<std::uint64_t>& values = operand.values;
            nb::gil_scoped_release release;
            switch (op)
            {
                case BinaryOp::add:
                    return a.value + values;
                case BinaryOp::sub:
                    return a.value - values;
                case BinaryOp::rsub:
                    return values - a.value;
                case BinaryOp::multiply:
                    return a.value * values;
            }
            return std::nullopt;
        }

        nb::object not_implemented()
        {
            return nb::borrow(nb::handle(Py_NotImplemented));
        }

        // A binary operator: a op other, or NotImplemented.
        template <BinaryOp Op>
        nb::object binary_operator(const SessionCipher& a, nb::handle other)
        {
            std::optional<SessionCiphertextCpp> result = apply(a, other, Op);
            if (!result)
            {
                return not_implemented();
            }
            return nb::cast(SessionCipher{std::move(*result), a.plain_modulus});
        }

        // A compound assignment: self op= other, or NotImplemented.
        template <BinaryOp Op>
        nb::object inplace_operator(nb::handle self, nb::handle other)
        {
            SessionCipher& a = nb::cast<SessionCipher&>(self);
            std::optional<SessionCiphertextCpp> result = apply(a, other, Op);
            if (!result)
            {
                return not_implemented();
            }
            {
                nb::gil_scoped_release release;
                a.value = std::move(*result);
            }
            return nb::borrow(self);
        }

        SessionCipher wrap_result(const Session& session,
                                  SessionCiphertextCpp value)
        {
            return SessionCipher{std::move(value),
                                 std::uint64_t(session.plain_modulus())};
        }

        std::string save_session(const Session& session, Role role)
        {
            std::ostringstream stream(std::ios::binary);
            {
                nb::gil_scoped_release release;
                session.save(stream, role);
            }
            return stream.str();
        }

        Session load_session(std::string data, std::optional<Role> expected)
        {
            std::istringstream stream(std::move(data), std::ios::binary);
            nb::gil_scoped_release release;
            if (expected)
            {
                return Session::load(stream, *expected);
            }
            return Session::load(stream);
        }

        SessionCipher load_session_ciphertext(const Session& session,
                                              std::string data)
        {
            std::istringstream stream(std::move(data), std::ios::binary);
            nb::gil_scoped_release release;
            return wrap_result(session, session.load_ciphertext(stream));
        }

        std::vector<int> to_rotations(nb::handle rotations)
        {
            if (rotations.is_none())
            {
                return {};
            }
            return to_int_vector(rotations, "rotations");
        }

        // A two-input logic gate of session_logic.cuh.
        template <SessionCiphertextCpp (*Gate)(const SessionCiphertextCpp&,
                                               const SessionCiphertextCpp&)>
        SessionCipher logic_gate(const SessionCipher& a, const SessionCipher& b)
        {
            std::uint64_t t = modulus_of(a);
            nb::gil_scoped_release release;
            return SessionCipher{Gate(a.value, b.value), t};
        }

        constexpr const char* ciphertext_pickle_message =
            "A SessionCiphertext cannot be pickled without its session; use "
            "to_bytes() and Session.ciphertext_from_bytes().";

        constexpr const char* session_pickle_message =
            "A Session cannot be pickled, since it would have to choose the "
            "keys to include; use to_bytes(role) and Session.from_bytes().";

    } // namespace

    void bind_bfv_session(nb::module_& m)
    {
        // --------------------------------------------------------------
        // SessionCiphertext
        // --------------------------------------------------------------

        nb::class_<SessionCipher> cipher(
            m, "SessionCiphertext",
            "A ciphertext of a Session. The operators +, -, * and their "
            "in-place forms work on ciphertexts of the same session, on "
            "integers (added to or multiplied with every slot) and on "
            "sequences or arrays of integers (one per slot, the other slots "
            "are 0), on both sides; the values are taken modulo the plaintext "
            "modulus. A product of two ciphertexts is relinearized. The "
            "ciphertext keeps the state of its session, so it stays usable "
            "after the Session object is deleted.");

        cipher
            .def(nb::init<>(),
                 "An empty ciphertext; only assigning to it is possible.")
            .def_prop_ro(
                "ciphertext",
                [](SessionCipher& self) -> Ciphertext&
                { return self.value.ciphertext(); },
                nb::rv_policy::reference_internal,
                "The underlying Ciphertext of the lower level API (a "
                "reference, not a copy).")
            .def_prop_ro(
                "stream",
                [](const SessionCipher& self)
                { return from_stream(self.value.stream()); },
                "The handle of the CUDA stream of the operations whose left "
                "operand this ciphertext is (see Session.on_stream).")
            .def_prop_ro(
                "plain_modulus",
                [](const SessionCipher& self) { return modulus_of(self); },
                "The plaintext modulus t of the session.")
            .def_prop_ro(
                "is_empty",
                [](const SessionCipher& self)
                { return self.plain_modulus == 0; },
                "Whether the ciphertext is empty (created with "
                "SessionCiphertext()).")
            .def(
                "rotate",
                [](const SessionCipher& self, int shift)
                {
                    std::uint64_t t = modulus_of(self);
                    nb::gil_scoped_release release;
                    return SessionCipher{self.value.rotate(shift), t};
                },
                "shift"_a,
                "Returns the ciphertext with both rows of slots rotated left "
                "by shift (right for a negative shift). The session generates "
                "the Galois key the first time a shift is used, if it has the "
                "secret key.")
            .def(
                "rotate_columns",
                [](const SessionCipher& self)
                {
                    std::uint64_t t = modulus_of(self);
                    nb::gil_scoped_release release;
                    return SessionCipher{self.value.rotate_columns(), t};
                },
                "Returns the ciphertext with the two rows of slots swapped.")
            .def(
                "rotate_hoisted",
                [](const SessionCipher& self, nb::handle shifts)
                {
                    std::uint64_t t = modulus_of(self);
                    std::vector<int> steps = to_int_vector(shifts, "shifts");
                    std::vector<SessionCipher> results;
                    {
                        nb::gil_scoped_release release;
                        std::vector<SessionCiphertextCpp> rotated =
                            self.value.rotate_hoisted(steps);
                        results.reserve(rotated.size());
                        for (SessionCiphertextCpp& value : rotated)
                        {
                            results.push_back({std::move(value), t});
                        }
                    }
                    return results;
                },
                "shifts"_a,
                "Returns a list with the ciphertext rotated by each shift, as "
                "rotate(shift) does; the rotations share the decomposition of "
                "the ciphertext.")
            .def(
                "sum_slots",
                [](const SessionCipher& self)
                {
                    std::uint64_t t = modulus_of(self);
                    nb::gil_scoped_release release;
                    return SessionCipher{self.value.sum_slots(), t};
                },
                "Returns a ciphertext whose slots all hold the sum of the "
                "slots of both rows (uses the rotations by powers of two and "
                "the column rotation).")
            .def(
                "inner_product",
                [](const SessionCipher& self, nb::handle other)
                {
                    std::uint64_t t = modulus_of(self);
                    if (nb::isinstance<SessionCipher>(other))
                    {
                        const SessionCipher& b =
                            nb::cast<const SessionCipher&>(other);
                        nb::gil_scoped_release release;
                        return SessionCipher{self.value.inner_product(b.value),
                                             t};
                    }
                    std::vector<std::uint64_t> values = slot_values(
                        other, t, self.value.ciphertext().ring_size());
                    nb::gil_scoped_release release;
                    return SessionCipher{self.value.inner_product(values), t};
                },
                "other"_a,
                "Returns a ciphertext whose slots all hold the inner product "
                "of the slots with those of other: a SessionCiphertext, an "
                "integer (in every slot) or a sequence of integers.")
            .def(
                "to_bytes",
                [](const SessionCipher& self)
                {
                    modulus_of(self);
                    std::string data = save_to_string(self.value);
                    return nb::bytes(data.data(), data.size());
                },
                "Serializes the ciphertext in the HEonGPU format; "
                "Session.ciphertext_from_bytes of a session with the same "
                "keys reads it.")
            .def(
                "save",
                [](const SessionCipher& self, nb::handle file)
                {
                    modulus_of(self);
                    write_file(file, save_to_string(self.value));
                },
                "file"_a,
                "Writes the ciphertext to a path or a binary file object; "
                "Session.load_ciphertext of a session with the same keys "
                "reads it.")
            .def(
                "copy",
                [](const SessionCipher& self)
                {
                    nb::gil_scoped_release release;
                    return SessionCipher(self);
                },
                "Returns a copy (the data is copied).")
            .def("__copy__",
                 [](const SessionCipher& self)
                 {
                     nb::gil_scoped_release release;
                     return SessionCipher(self);
                 })
            .def(
                "__deepcopy__",
                [](const SessionCipher& self, nb::handle)
                {
                    nb::gil_scoped_release release;
                    return SessionCipher(self);
                },
                "memo"_a)
            .def("__reduce_ex__",
                 [](const SessionCipher&, nb::handle) -> nb::object
                 { throw nb::type_error(ciphertext_pickle_message); })
            .def("__getstate__",
                 [](const SessionCipher&) -> nb::object
                 { throw nb::type_error(ciphertext_pickle_message); })
            .def("__repr__",
                 [](const SessionCipher& self)
                 {
                     if (self.plain_modulus == 0)
                     {
                         return std::string(
                             "<heongpu.bfv.SessionCiphertext empty>");
                     }
                     const Ciphertext& c = self.value.ciphertext();
                     std::ostringstream stream;
                     stream
                         << "<heongpu.bfv.SessionCiphertext ring_size="
                         << c.ring_size()
                         << " coeff_modulus_count=" << c.coeff_modulus_count()
                         << " plain_modulus=" << self.plain_modulus << ">";
                     return stream.str();
                 })
            .def("__add__", &binary_operator<BinaryOp::add>, nb::is_operator(),
                 nb::sig("def __add__(self, other: SessionCiphertext | int | "
                         "collections.abc.Sequence[int] | numpy.ndarray, /) -> "
                         "SessionCiphertext"))
            .def("__radd__", &binary_operator<BinaryOp::add>, nb::is_operator(),
                 nb::sig("def __radd__(self, other: SessionCiphertext | int | "
                         "collections.abc.Sequence[int] | numpy.ndarray, /) -> "
                         "SessionCiphertext"))
            .def("__sub__", &binary_operator<BinaryOp::sub>, nb::is_operator(),
                 nb::sig("def __sub__(self, other: SessionCiphertext | int | "
                         "collections.abc.Sequence[int] | numpy.ndarray, /) -> "
                         "SessionCiphertext"))
            .def("__rsub__", &binary_operator<BinaryOp::rsub>,
                 nb::is_operator(),
                 nb::sig("def __rsub__(self, other: SessionCiphertext | int | "
                         "collections.abc.Sequence[int] | numpy.ndarray, /) -> "
                         "SessionCiphertext"))
            .def("__mul__", &binary_operator<BinaryOp::multiply>,
                 nb::is_operator(),
                 nb::sig("def __mul__(self, other: SessionCiphertext | int | "
                         "collections.abc.Sequence[int] | numpy.ndarray, /) -> "
                         "SessionCiphertext"))
            .def("__rmul__", &binary_operator<BinaryOp::multiply>,
                 nb::is_operator(),
                 nb::sig("def __rmul__(self, other: SessionCiphertext | int | "
                         "collections.abc.Sequence[int] | numpy.ndarray, /) -> "
                         "SessionCiphertext"))
            .def("__iadd__", &inplace_operator<BinaryOp::add>,
                 nb::is_operator(),
                 nb::sig("def __iadd__(self, other: SessionCiphertext | int | "
                         "collections.abc.Sequence[int] | numpy.ndarray, /) -> "
                         "SessionCiphertext"))
            .def("__isub__", &inplace_operator<BinaryOp::sub>,
                 nb::is_operator(),
                 nb::sig("def __isub__(self, other: SessionCiphertext | int | "
                         "collections.abc.Sequence[int] | numpy.ndarray, /) -> "
                         "SessionCiphertext"))
            .def("__imul__", &inplace_operator<BinaryOp::multiply>,
                 nb::is_operator(),
                 nb::sig("def __imul__(self, other: SessionCiphertext | int | "
                         "collections.abc.Sequence[int] | numpy.ndarray, /) -> "
                         "SessionCiphertext"))
            .def(
                "__neg__",
                [](const SessionCipher& self)
                {
                    std::uint64_t t = modulus_of(self);
                    nb::gil_scoped_release release;
                    return SessionCipher{-self.value, t};
                },
                nb::is_operator())
            .def(
                "__pos__",
                [](const SessionCipher& self)
                {
                    nb::gil_scoped_release release;
                    return SessionCipher(self);
                },
                nb::is_operator());

        // NumPy arrays on the left of an operator call the reflected
        // operators of the ciphertext instead of broadcasting over it.
        cipher.attr("__array_ufunc__") = nb::none();

        // --------------------------------------------------------------
        // Session
        // --------------------------------------------------------------

        nb::class_<Session> session(
            m, "Session",
            "One party's BFV computation in one object: the context, the "
            "keys, the encoder, the encryptor, the decryptor and the "
            "operators. A session created from parameters is the owner of "
            "its keys; save(file, role) writes it with the keys of a role "
            "and load() reads it, for example for a server that computes "
            "without the secret key. Copies of a session share its state.");

        session
            .def(
                "__init__",
                [](Session* self, const Params& params, nb::handle rotations)
                {
                    std::vector<int> steps = to_rotations(rotations);
                    nb::gil_scoped_release release;
                    new (self) Session(params, steps);
                },
                "params"_a, "rotations"_a.none() = nb::none(),
                "Generates the context of params.generate(), the secret, "
                "public and relinearization keys, and the Galois keys for the "
                "shifts in rotations.")
            .def(
                "__init__",
                [](Session* self, const ParameterSet& parameters,
                   nb::handle rotations)
                {
                    std::vector<int> steps = to_rotations(rotations);
                    nb::gil_scoped_release release;
                    new (self) Session(parameters, steps);
                },
                "parameters"_a, "rotations"_a.none() = nb::none(),
                "Generates the context of the parameter set and the keys as "
                "Session(params, rotations) does.")
            .def_prop_ro("role", &Session::role,
                         "The keys of the session: OWNER has the secret key, "
                         "EVALUATOR the evaluation keys, ENCRYPTOR the public "
                         "key only.")
            .def_prop_ro("slot_count", &Session::slot_count,
                         "The number of slots, N.")
            .def_prop_ro(
                "row_size",
                [](const Session& self) { return self.slot_count() / 2; },
                "The number of slots of a row, N/2.")
            .def_prop_ro(
                "plain_modulus",
                [](const Session& self)
                { return std::uint64_t(self.plain_modulus()); },
                "The plaintext modulus t.")
            .def_prop_ro(
                "stream",
                [](const Session& self) { return from_stream(self.stream()); },
                "The handle of the CUDA stream of the session.")
            .def(
                "encrypt",
                [](const Session& self, const Plaintext& plain)
                {
                    require_generated(plain);
                    nb::gil_scoped_release release;
                    return wrap_result(self, self.encrypt(plain));
                },
                "plain"_a, "Returns the plaintext encrypted.")
            .def(
                "encrypt",
                [](const Session& self, nb::handle values)
                {
                    std::vector<std::uint64_t> message =
                        slot_values(values, self.plain_modulus(),
                                    std::size_t(self.slot_count()));
                    nb::gil_scoped_release release;
                    return wrap_result(self, self.encrypt(message));
                },
                "values"_a,
                "Returns the values encrypted: an integer (in every slot), or "
                "a sequence or an array of at most slot_count integers, one "
                "per slot (the other slots are 0). The values are taken "
                "modulo the plaintext modulus.")
            .def(
                "wrap",
                [](const Session& self, const Ciphertext& ciphertext)
                {
                    nb::gil_scoped_release release;
                    return wrap_result(self, self.wrap(ciphertext));
                },
                "ciphertext"_a,
                "Returns a copy of a Ciphertext as a ciphertext of the "
                "session. Raises ContextMismatchError unless it has the "
                "parameters of the session; it must also be encrypted with "
                "the keys of the session.")
            .def(
                "decrypt",
                [](const Session& self, const SessionCipher& input,
                   bool is_signed) -> nb::object
                {
                    if (is_signed)
                    {
                        std::vector<std::int64_t> values;
                        {
                            nb::gil_scoped_release release;
                            values = self.decrypt_signed(input.value);
                        }
                        return nb::cast(to_numpy(std::move(values)));
                    }
                    std::vector<std::uint64_t> values;
                    {
                        nb::gil_scoped_release release;
                        values = self.decrypt(input.value);
                    }
                    return nb::cast(to_numpy(std::move(values)));
                },
                "ciphertext"_a, nb::kw_only(), "signed"_a = false,
                "Returns the values of all slots as a uint64 NumPy array of "
                "values in [0, t), or with signed=True an int64 array of the "
                "centered values. Raises StateError without the secret key.")
            .def(
                "decrypt_signed",
                [](const Session& self, const SessionCipher& input)
                {
                    std::vector<std::int64_t> values;
                    {
                        nb::gil_scoped_release release;
                        values = self.decrypt_signed(input.value);
                    }
                    return to_numpy(std::move(values));
                },
                "ciphertext"_a,
                "Returns the values of all slots as an int64 NumPy array, "
                "from -(t - 1) / 2 to (t - 1) / 2.")
            .def(
                "noise_budget",
                [](const Session& self, const SessionCipher& input)
                {
                    nb::gil_scoped_release release;
                    return self.noise_budget(input.value);
                },
                "ciphertext"_a,
                "Returns the bits of noise budget the ciphertext has left; it "
                "decrypts correctly while the budget is above 0.")
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
                "key for yet. Raises KeyNotFoundError without the secret "
                "key.")
            .def(
                "on_stream",
                [](const Session& self, nb::handle stream)
                {
                    cudaStream_t s = to_stream(stream);
                    nb::gil_scoped_release release;
                    return self.on_stream(s);
                },
                "stream"_a.none(), nb::keep_alive<0, 2>(),
                "Returns a view of the session (sharing its keys) whose "
                "operations, and those of its ciphertexts, run on stream. "
                "Waits for the stream of this session first.")
            .def(
                "synchronize",
                [](const Session& self)
                {
                    nb::gil_scoped_release release;
                    self.synchronize();
                },
                "Waits for the operations queued on the stream of the "
                "session.")
            .def(
                "save",
                [](const Session& self, nb::handle file, Role role)
                { write_file(file, save_session(self, role)); },
                "file"_a, "role"_a,
                "Writes the session with the keys of role to a path or a "
                "binary file object: the context and the public key, for "
                "EVALUATOR also the relinearization and Galois keys, for "
                "OWNER also the secret key. Raises StateError if the session "
                "does not have them.")
            .def(
                "to_bytes",
                [](const Session& self, Role role)
                {
                    std::string data = save_session(self, role);
                    return nb::bytes(data.data(), data.size());
                },
                "role"_a,
                "Serializes the session with the keys of role, as save() "
                "does.")
            .def_static(
                "load",
                [](nb::handle file, std::optional<Role> expected_role)
                { return load_session(read_file(file), expected_role); },
                "file"_a, "expected_role"_a.none() = nb::none(),
                "Reads a session that save() wrote. With expected_role, raises "
                "StateError, before reading the keys, unless it was saved "
                "with that role.")
            .def_static(
                "from_bytes",
                [](nb::bytes data, std::optional<Role> expected_role)
                { return load_session(bytes_to_string(data), expected_role); },
                "data"_a, "expected_role"_a.none() = nb::none(),
                "Reads a session that to_bytes() wrote; see load().")
            .def(
                "load_ciphertext",
                [](const Session& self, nb::handle file)
                { return load_session_ciphertext(self, read_file(file)); },
                "file"_a,
                "Reads a ciphertext that SessionCiphertext.save or "
                "Ciphertext.save wrote, from a path or a binary file object; "
                "it must have the parameters of the session.")
            .def(
                "ciphertext_from_bytes",
                [](const Session& self, nb::bytes data) {
                    return load_session_ciphertext(self, bytes_to_string(data));
                },
                "data"_a,
                "Reads a ciphertext that SessionCiphertext.to_bytes or "
                "Ciphertext.to_bytes wrote; see load_ciphertext().")
            .def("context", &Session::context,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "The Context of the session.")
            .def("secret_key", &Session::secret_key,
                 nb::rv_policy::reference_internal,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "The secret key (a reference). Raises StateError without "
                 "it.")
            .def("public_key", &Session::public_key,
                 nb::rv_policy::reference_internal,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "The public key (a reference).")
            .def("relin_key", &Session::relin_key,
                 nb::rv_policy::reference_internal,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "The relinearization key (a reference). Raises "
                 "KeyNotFoundError without it.")
            .def("galois_key", &Session::galois_key,
                 nb::rv_policy::reference_internal,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "The Galois keys generated so far, with the column rotation "
                 "key (a reference). A rotation by a new shift adds its key "
                 "to it, so do not use it while other threads rotate.")
            .def("encoder", &Session::encoder,
                 nb::rv_policy::reference_internal,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "The Encoder of the session (a reference).")
            .def("encryptor", &Session::encryptor,
                 nb::rv_policy::reference_internal,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "The Encryptor of the session (a reference).")
            .def("decryptor", &Session::decryptor,
                 nb::rv_policy::reference_internal,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "The Decryptor of the session (a reference). Raises "
                 "StateError without the secret key.")
            .def("operators", &Session::operators,
                 nb::rv_policy::reference_internal,
                 nb::call_guard<nb::gil_scoped_release>(),
                 "The ArithmeticOperator of the session (a reference).")
            .def(
                "__copy__", [](const Session& self) { return Session(self); },
                "Returns a session sharing the state (keys and context) of "
                "this one.")
            .def("__reduce_ex__",
                 [](const Session&, nb::handle) -> nb::object
                 { throw nb::type_error(session_pickle_message); })
            .def("__repr__",
                 [](const Session& self)
                 {
                     std::ostringstream stream;
                     stream << "<heongpu.bfv.Session role="
                            << role_name(self.role())
                            << " slot_count=" << self.slot_count()
                            << " plain_modulus=" << self.plain_modulus() << ">";
                     return stream.str();
                 });

        // --------------------------------------------------------------
        // Logic gates on 0/1 slot values
        // --------------------------------------------------------------

        m.def(
            "NOT",
            [](const SessionCipher& a)
            {
                std::uint64_t t = modulus_of(a);
                nb::gil_scoped_release release;
                return SessionCipher{heongpu::NOT(a.value), t};
            },
            "a"_a,
            "Returns 1 - a, the NOT of the bits (0 or 1) in the slots of a.");
        m.def("AND", &logic_gate<&heongpu::AND<BFV>>, "a"_a, "b"_a,
              "Returns a * b, the AND of the bits in the slots.");
        m.def("OR", &logic_gate<&heongpu::OR<BFV>>, "a"_a, "b"_a,
              "Returns a + b - a * b, the OR of the bits in the slots.");
        m.def("XOR", &logic_gate<&heongpu::XOR<BFV>>, "a"_a, "b"_a,
              "Returns a + b - 2 a b, the XOR of the bits in the slots.");
        m.def("NAND", &logic_gate<&heongpu::NAND<BFV>>, "a"_a, "b"_a,
              "Returns 1 - a * b, the NAND of the bits in the slots.");
        m.def("NOR", &logic_gate<&heongpu::NOR<BFV>>, "a"_a, "b"_a,
              "Returns the NOR of the bits in the slots.");
        m.def("XNOR", &logic_gate<&heongpu::XNOR<BFV>>, "a"_a, "b"_a,
              "Returns the XNOR of the bits in the slots.");
    }

} // namespace heongpu_py
