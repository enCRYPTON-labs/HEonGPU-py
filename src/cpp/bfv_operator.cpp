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

        using ContextPtr = heongpu::HEContext<BFV>;
        using RelinKey = heongpu::Relinkey<BFV>;
        using GaloisKey = heongpu::Galoiskey<BFV>;
        using SwitchKey = heongpu::Switchkey<BFV>;
        using Plaintext = heongpu::Plaintext<BFV>;
        using Ciphertext = heongpu::Ciphertext<BFV>;
        using Encoder = heongpu::HEEncoder<BFV>;
        using Operator = heongpu::HEOperator<BFV>;
        using ArithmeticOperator = heongpu::HEArithmeticOperator<BFV>;
        using LogicOperator = heongpu::HELogicOperator<BFV>;

        /**
         * @brief Reads the protected context of an operator, which the
         * outputs of the operations without a value-returning C++ overload
         * are created with.
         */
        struct OperatorAccess : Operator
        {
            static const ContextPtr& context(const Operator& op)
            {
                return op.*(&OperatorAccess::context_);
            }
        };

        /**
         * @brief Binds NAME(cipher, cipher, relin_key) and NAME(cipher, plain)
         * with their _inplace variants for a gate of HELogicOperator. The
         * output is a copy of the first input the in-place gate is applied
         * to, as the C++ _inplace methods do.
         */
#define HEONGPU_PY_BFV_GATE(CLS, NAME, DOC)                                    \
    CLS.def(                                                                   \
           #NAME,                                                              \
           [](LogicOperator& self, const Ciphertext& a, const Ciphertext& b,   \
              const RelinKey& relin_key, nb::handle options)                   \
           {                                                                   \
               require_generated(relin_key);                                   \
               heongpu::ExecutionOptions opts = to_options(options);           \
               nb::gil_scoped_release release;                                 \
               Ciphertext output(a);                                           \
               self.NAME##_inplace(output, b, relin_key, opts);                \
               return output;                                                  \
           },                                                                  \
           "input1"_a, "input2"_a, "relin_key"_a, "options"_a = nb::none(),    \
           DOC)                                                                \
        .def(                                                                  \
            #NAME,                                                             \
            [](LogicOperator& self, const Ciphertext& a, const Plaintext& b,   \
               nb::handle options)                                             \
            {                                                                  \
                require_generated(b);                                          \
                heongpu::ExecutionOptions opts = to_options(options);          \
                nb::gil_scoped_release release;                                \
                Ciphertext output(a);                                          \
                self.NAME##_inplace(output, b, opts);                          \
                return output;                                                 \
            },                                                                 \
            "input1"_a, "input2"_a, "options"_a = nb::none())                  \
        .def(                                                                  \
            #NAME "_inplace",                                                  \
            [](LogicOperator& self, Ciphertext& a, const Ciphertext& b,        \
               const RelinKey& relin_key, nb::handle options)                  \
            {                                                                  \
                require_generated(relin_key);                                  \
                heongpu::ExecutionOptions opts = to_options(options);          \
                nb::gil_scoped_release release;                                \
                self.NAME##_inplace(a, b, relin_key, opts);                    \
            },                                                                 \
            "input1"_a, "input2"_a, "relin_key"_a, "options"_a = nb::none(),   \
            "In-place variant: the result replaces input1.")                   \
        .def(                                                                  \
            #NAME "_inplace",                                                  \
            [](LogicOperator& self, Ciphertext& a, const Plaintext& b,         \
               nb::handle options)                                             \
            {                                                                  \
                require_generated(b);                                          \
                heongpu::ExecutionOptions opts = to_options(options);          \
                nb::gil_scoped_release release;                                \
                self.NAME##_inplace(a, b, opts);                               \
            },                                                                 \
            "input1"_a, "input2"_a, "options"_a = nb::none())
    } // namespace

    void bind_bfv_operator(nb::module_& m)
    {
        // --------------------------------------------------------------
        // Arithmetic operator
        // --------------------------------------------------------------

        nb::class_<ArithmeticOperator> arithmetic(
            m, "ArithmeticOperator",
            "Homomorphic arithmetic on BFV ciphertexts: the slot-wise "
            "operations modulo the plaintext modulus, rotations and key "
            "switching. Methods return a new ciphertext; the _inplace "
            "variants replace their first argument.");
        arithmetic.def(nb::init<ContextPtr, Encoder&>(), "context"_a,
                       "encoder"_a);

        // Addition, subtraction and negation.
        arithmetic
            .def(
                "add",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const Ciphertext& b, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.add(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(),
                "Returns a + b.")
            .def(
                "add_inplace",
                [](ArithmeticOperator& self, Ciphertext& a, const Ciphertext& b,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.add_inplace(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(), "a += b.")
            .def(
                "sub",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const Ciphertext& b, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.sub(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(),
                "Returns a - b.")
            .def(
                "sub_inplace",
                [](ArithmeticOperator& self, Ciphertext& a, const Ciphertext& b,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.sub_inplace(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(), "a -= b.")
            .def(
                "negate",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.negate(a, opts);
                },
                "input"_a, "options"_a = nb::none(), "Returns -a.")
            .def(
                "negate_inplace",
                [](ArithmeticOperator& self, Ciphertext& a, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.negate_inplace(a, opts);
                },
                "input"_a, "options"_a = nb::none(), "a = -a.")
            .def(
                "add_plain",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const Plaintext& b, nb::handle options)
                {
                    require_generated(b);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.add_plain(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(),
                "Returns a + b for a plaintext b.")
            .def(
                "add_plain_inplace",
                [](ArithmeticOperator& self, Ciphertext& a, const Plaintext& b,
                   nb::handle options)
                {
                    require_generated(b);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.add_plain_inplace(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(),
                "a += b for a plaintext b.")
            .def(
                "sub_plain",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const Plaintext& b, nb::handle options)
                {
                    require_generated(b);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.sub_plain(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(),
                "Returns a - b for a plaintext b.")
            .def(
                "sub_plain_inplace",
                [](ArithmeticOperator& self, Ciphertext& a, const Plaintext& b,
                   nb::handle options)
                {
                    require_generated(b);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.sub_plain_inplace(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(),
                "a -= b for a plaintext b.");

        // Multiplication and relinearization.
        arithmetic
            .def(
                "multiply",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const Ciphertext& b, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.multiply(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(),
                "Returns a * b, a ciphertext of size 3 that needs "
                "relinearize() before decryption, the next multiplication or "
                "a rotation.")
            .def(
                "multiply_inplace",
                [](ArithmeticOperator& self, Ciphertext& a, const Ciphertext& b,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.multiply_inplace(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(), "a *= b.")
            .def(
                "multiply_plain",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const Plaintext& b, nb::handle options)
                {
                    require_generated(b);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.multiply_plain(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(),
                "Returns a * b for a plaintext b.")
            .def(
                "multiply_plain_inplace",
                [](ArithmeticOperator& self, Ciphertext& a, const Plaintext& b,
                   nb::handle options)
                {
                    require_generated(b);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.multiply_plain_inplace(a, b, opts);
                },
                "input1"_a, "input2"_a, "options"_a = nb::none(),
                "a *= b for a plaintext b.")
            .def(
                "relinearize",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const RelinKey& relin_key, nb::handle options)
                {
                    require_generated(relin_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    Ciphertext output(a);
                    self.relinearize_inplace(output, relin_key, opts);
                    return output;
                },
                "input"_a, "relin_key"_a, "options"_a = nb::none(),
                "Returns a relinearized copy of a ciphertext of size 3.")
            .def(
                "relinearize_inplace",
                [](ArithmeticOperator& self, Ciphertext& a,
                   const RelinKey& relin_key, nb::handle options)
                {
                    require_generated(relin_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.relinearize_inplace(a, relin_key, opts);
                },
                "input"_a, "relin_key"_a, "options"_a = nb::none(),
                "Brings a ciphertext of size 3 back to size 2.");

        // Rotations and key switching.
        arithmetic
            .def(
                "rotate_rows",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const GaloisKey& galois_key, int shift, nb::handle options)
                {
                    require_generated(galois_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.rotate_rows(a, galois_key, shift, opts);
                },
                "input"_a, "galois_key"_a, "shift"_a, "options"_a = nb::none(),
                "Rotates both rows of N/2 slots by shift to the left (to the "
                "right for a negative shift): slot i gets slot i + shift. "
                "Shifts without a key of their own are composed of the "
                "powers of two in the key.")
            .def(
                "rotate_rows_inplace",
                [](ArithmeticOperator& self, Ciphertext& a,
                   const GaloisKey& galois_key, int shift, nb::handle options)
                {
                    require_generated(galois_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.rotate_rows_inplace(a, galois_key, shift, opts);
                },
                "input"_a, "galois_key"_a, "shift"_a, "options"_a = nb::none())
            .def(
                "rotate_rows_hoisted",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const GaloisKey& galois_key, nb::handle shifts,
                   nb::handle options)
                {
                    require_generated(galois_key);
                    std::vector<int> steps = to_int_vector(shifts, "shifts");
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.rotate_rows_hoisted(a, galois_key, steps, opts);
                },
                "input"_a, "galois_key"_a, "shifts"_a, "options"_a = nb::none(),
                "Returns the row rotations of a by each shift, as a list. The "
                "decomposition of a is computed once and shared by the "
                "shifts that have a key of their own.")
            .def(
                "rotate_columns",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const GaloisKey& galois_key, nb::handle options)
                {
                    require_generated(galois_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.rotate_columns(a, galois_key, opts);
                },
                "input"_a, "galois_key"_a, "options"_a = nb::none(),
                "Swaps the two rows.")
            .def(
                "apply_galois",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const GaloisKey& galois_key, int galois_element,
                   nb::handle options)
                {
                    require_generated(galois_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    const ContextPtr& context = OperatorAccess::context(self);
                    nb::gil_scoped_release release;
                    Ciphertext output(context, opts);
                    self.apply_galois(a, output, galois_key, galois_element,
                                      opts);
                    return output;
                },
                "input"_a, "galois_key"_a, "galois_element"_a,
                "options"_a = nb::none(),
                "Applies the automorphism X -> X^galois_element; the key must "
                "hold that Galois element.")
            .def(
                "apply_galois_inplace",
                [](ArithmeticOperator& self, Ciphertext& a,
                   const GaloisKey& galois_key, int galois_element,
                   nb::handle options)
                {
                    require_generated(galois_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.apply_galois_inplace(a, galois_key, galois_element,
                                              opts);
                },
                "input"_a, "galois_key"_a, "galois_element"_a,
                "options"_a = nb::none())
            .def(
                "keyswitch",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const SwitchKey& switch_key, nb::handle options)
                {
                    require_generated(switch_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    const ContextPtr& context = OperatorAccess::context(self);
                    nb::gil_scoped_release release;
                    Ciphertext output(context, opts);
                    self.keyswitch(a, output, switch_key, opts);
                    return output;
                },
                "input"_a, "switch_key"_a, "options"_a = nb::none(),
                "Returns a under the new secret key of the switch key.")
            .def(
                "multiply_power_of_X",
                [](ArithmeticOperator& self, const Ciphertext& a, int index,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    const ContextPtr& context = OperatorAccess::context(self);
                    nb::gil_scoped_release release;
                    if (index == 0)
                    {
                        return Ciphertext(a);
                    }
                    Ciphertext output(context, opts);
                    self.multiply_power_of_X(a, output, index, opts);
                    return output;
                },
                "input"_a, "index"_a, "options"_a = nb::none(),
                "Multiplies the plaintext polynomial by X^index (a "
                "negacyclic shift of its coefficients).");

        // Slot sums.
        arithmetic
            .def(
                "sum_slots",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const GaloisKey& galois_key, nb::handle options)
                {
                    require_generated(galois_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.sum_slots(a, galois_key, opts);
                },
                "input"_a, "galois_key"_a, "options"_a = nb::none(),
                "Returns the sum of all slots in every slot. Needs the row "
                "rotations by the powers of two up to N/4, for example "
                "GaloisKey(context, max_shift=log2(N) - 2).")
            .def(
                "inner_product",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const Ciphertext& b, const RelinKey& relin_key,
                   const GaloisKey& galois_key, nb::handle options)
                {
                    require_generated(relin_key);
                    require_generated(galois_key);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.inner_product(a, b, relin_key, galois_key,
                                              opts);
                },
                "input1"_a, "input2"_a, "relin_key"_a, "galois_key"_a,
                "options"_a = nb::none(),
                "Returns the inner product of the slots of a and b in every "
                "slot (multiply, relinearize, sum_slots).")
            .def(
                "inner_product",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   const Plaintext& b, const GaloisKey& galois_key,
                   nb::handle options)
                {
                    require_generated(galois_key);
                    require_generated(b);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    return self.inner_product(a, b, galois_key, opts);
                },
                "input1"_a, "input2"_a, "galois_key"_a,
                "options"_a = nb::none(),
                "Returns the inner product of the slots of a and of a "
                "plaintext b in every slot.");

        // NTT domain.
        arithmetic
            .def(
                "transform_to_ntt",
                [](ArithmeticOperator& self, const Plaintext& a,
                   nb::handle options)
                {
                    require_generated(a);
                    heongpu::ExecutionOptions opts = to_options(options);
                    const ContextPtr& context = OperatorAccess::context(self);
                    nb::gil_scoped_release release;
                    Plaintext output(context, opts);
                    self.transform_to_ntt(a, output, opts);
                    return output;
                },
                "input"_a, "options"_a = nb::none(),
                "Returns the plaintext lifted to the ciphertext moduli in the "
                "NTT domain, which makes repeated multiply_plain() with it "
                "faster on ciphertexts in the NTT domain.")
            .def(
                "transform_to_ntt",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    const ContextPtr& context = OperatorAccess::context(self);
                    nb::gil_scoped_release release;
                    Ciphertext output(context, opts);
                    self.transform_to_ntt(a, output, opts);
                    return output;
                },
                "input"_a, "options"_a = nb::none(),
                "Returns the ciphertext in the NTT domain.")
            .def(
                "transform_to_ntt_inplace",
                [](ArithmeticOperator& self, Plaintext& a, nb::handle options)
                {
                    require_generated(a);
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.transform_to_ntt_inplace(a, opts);
                },
                "input"_a, "options"_a = nb::none())
            .def(
                "transform_to_ntt_inplace",
                [](ArithmeticOperator& self, Ciphertext& a, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.transform_to_ntt_inplace(a, opts);
                },
                "input"_a, "options"_a = nb::none())
            .def(
                "transform_from_ntt",
                [](ArithmeticOperator& self, const Ciphertext& a,
                   nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    const ContextPtr& context = OperatorAccess::context(self);
                    nb::gil_scoped_release release;
                    Ciphertext output(context, opts);
                    self.transform_from_ntt(a, output, opts);
                    return output;
                },
                "input"_a, "options"_a = nb::none(),
                "Returns the ciphertext back in the coefficient domain.")
            .def(
                "transform_from_ntt_inplace",
                [](ArithmeticOperator& self, Ciphertext& a, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.transform_from_ntt_inplace(a, opts);
                },
                "input"_a, "options"_a = nb::none());

        // --------------------------------------------------------------
        // Logic operator
        // --------------------------------------------------------------

        nb::class_<LogicOperator> logic(
            m, "LogicOperator",
            "Boolean gates on BFV ciphertexts whose slots hold 0 or 1. The "
            "gates of two ciphertexts multiply them and need the "
            "relinearization key; the gates with a plaintext do not.");
        logic.def(nb::init<ContextPtr, Encoder&>(), "context"_a, "encoder"_a)
            .def(
                "NOT",
                [](LogicOperator& self, const Ciphertext& a, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    Ciphertext output(a);
                    self.NOT_inplace(output, opts);
                    return output;
                },
                "input"_a, "options"_a = nb::none(), "Returns 1 - a.")
            .def(
                "NOT_inplace",
                [](LogicOperator& self, Ciphertext& a, nb::handle options)
                {
                    heongpu::ExecutionOptions opts = to_options(options);
                    nb::gil_scoped_release release;
                    self.NOT_inplace(a, opts);
                },
                "input"_a, "options"_a = nb::none(), "a = 1 - a.");

        HEONGPU_PY_BFV_GATE(logic, AND, "Returns a AND b (a * b).");
        HEONGPU_PY_BFV_GATE(logic, OR, "Returns a OR b (a + b - a * b).");
        HEONGPU_PY_BFV_GATE(logic, XOR, "Returns a XOR b (a + b - 2 * a * b).");
        HEONGPU_PY_BFV_GATE(logic, NAND, "Returns NOT (a AND b).");
        HEONGPU_PY_BFV_GATE(logic, NOR, "Returns NOT (a OR b).");
        HEONGPU_PY_BFV_GATE(logic, XNOR, "Returns NOT (a XOR b).");
    }

#undef HEONGPU_PY_BFV_GATE

} // namespace heongpu_py
