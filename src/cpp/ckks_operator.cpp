// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "common.h"

#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>

#include <complex>
#include <functional>

namespace heongpu_py
{
    namespace
    {
        constexpr auto CKKS = heongpu::Scheme::CKKS;

        using ContextPtr = heongpu::HEContext<CKKS>;
        using RelinKey = heongpu::Relinkey<CKKS>;
        using GaloisKey = heongpu::Galoiskey<CKKS>;
        using SwitchKey = heongpu::Switchkey<CKKS>;
        using Plaintext = heongpu::Plaintext<CKKS>;
        using Ciphertext = heongpu::Ciphertext<CKKS>;
        using Encoder = heongpu::HEEncoder<CKKS>;
        using BaseOperator = heongpu::HEOperator<CKKS>;
        using ArithmeticOperator = heongpu::HEArithmeticOperator<CKKS>;
        using LogicOperator = heongpu::HELogicOperator<CKKS>;
        using BootstrappingKeys = heongpu::CKKSBootstrappingKeys;
        using TransformConfig = heongpu::CKKSEncodingTransformConfig;
        using TransformContext = heongpu::CKKSEncodingTransformContext;
        using Options = heongpu::ExecutionOptions;

        /**
         * @brief Reads the context of an operator, which HEOperator keeps as
         * a protected member. The outputs of the operations without a C++
         * overload that returns them are created with it, as the C++
         * overloads do.
         */
        struct OperatorAccess : BaseOperator
        {
            static const ContextPtr& context(const BaseOperator& op)
            {
                return op.*(&OperatorAccess::context_);
            }
        };

        // HELogicOperator derives privately from HEOperator; an explicit
        // conversion to the base class is still valid.
        const ContextPtr& context_of(const ArithmeticOperator& op)
        {
            return OperatorAccess::context(op);
        }

        const ContextPtr& context_of(const LogicOperator& op)
        {
            return OperatorAccess::context((const BaseOperator&) op);
        }

        /**
         * @brief Converts the options and calls function(options) with the
         * GIL released.
         */
        template <typename Function>
        auto run(nb::handle options, Function&& function)
        {
            Options opts = to_options(options);
            nb::gil_scoped_release release;
            return function(opts);
        }

        Complex64 to_complex64(const std::complex<double>& value)
        {
            return Complex64(value.real(), value.imag());
        }

        /**
         * @brief Wraps a Python callable as a real function for the C++
         * code, which calls it with the GIL released.
         */
        std::function<double(double)> real_function(nb::handle function)
        {
            return [function](double x)
            {
                nb::gil_scoped_acquire acquire;
                nb::object result = function(x);
                // A result that is not a real number raises TypeError.
                double value = PyFloat_AsDouble(result.ptr());
                if (value == -1.0 && PyErr_Occurred())
                {
                    throw nb::python_error();
                }
                return value;
            };
        }

        /**
         * @brief Returns a copy of input for an operation on the stream of
         * opts. The copy is made on the stream of input; the stream of opts
         * waits for it.
         */
        Ciphertext copy_for(const Ciphertext& input, const Options& opts)
        {
            Ciphertext output(input);
            if (output.is_on_device() && input.stream() != opts.stream_)
            {
                cudaEvent_t event;
                HEONGPU_PY_CUDA_CHECK(
                    cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                HEONGPU_PY_CUDA_CHECK(cudaEventRecord(event, input.stream()));
                HEONGPU_PY_CUDA_CHECK(
                    cudaStreamWaitEvent(opts.stream_, event, 0));
                HEONGPU_PY_CUDA_CHECK(cudaEventDestroy(event));
                output.switch_stream(opts.stream_);
            }
            return output;
        }

        // --------------------------------------------------------------
        // Operations shared by ArithmeticOperator and LogicOperator
        // --------------------------------------------------------------

        template <typename Op, typename Class>
        void def_shared_operations(Class& cls)
        {
            cls.def(
                   "rotate",
                   [](Op& self, const Ciphertext& input,
                      const GaloisKey& galois_key, int shift,
                      nb::handle options)
                   {
                       return run(options,
                                  [&](const Options& opts) {
                                      return self.rotate(input, galois_key,
                                                         shift, opts);
                                  });
                   },
                   "input"_a, "galois_key"_a, "shift"_a,
                   "options"_a = nb::none(),
                   "Rotates the slots to the left by shift (to the right if "
                   "it is negative). The Galois key must have a key for the "
                   "shift or for the powers of two that sum to it.")
                .def(
                    "rotate_inplace",
                    [](Op& self, Ciphertext& input, const GaloisKey& galois_key,
                       int shift, nb::handle options)
                    {
                        run(options,
                            [&](const Options& opts) {
                                self.rotate_inplace(input, galois_key, shift,
                                                    opts);
                            });
                    },
                    "input"_a, "galois_key"_a, "shift"_a,
                    "options"_a = nb::none(), "Rotates input in place.")
                .def(
                    "apply_galois",
                    [](Op& self, const Ciphertext& input,
                       const GaloisKey& galois_key, int galois_element,
                       nb::handle options)
                    {
                        return run(
                            options,
                            [&](const Options& opts)
                            {
                                Ciphertext output(context_of(self), opts);
                                self.apply_galois(input, output, galois_key,
                                                  galois_element, opts);
                                return output;
                            });
                    },
                    "input"_a, "galois_key"_a, "galois_element"_a,
                    "options"_a = nb::none(),
                    "Applies the automorphism of a Galois element; the Galois "
                    "key must have a key for it.")
                .def(
                    "apply_galois_inplace",
                    [](Op& self, Ciphertext& input, const GaloisKey& galois_key,
                       int galois_element, nb::handle options)
                    {
                        run(options,
                            [&](const Options& opts) {
                                self.apply_galois_inplace(input, galois_key,
                                                          galois_element, opts);
                            });
                    },
                    "input"_a, "galois_key"_a, "galois_element"_a,
                    "options"_a = nb::none(),
                    "Applies the automorphism of a Galois element in place.")
                .def(
                    "keyswitch",
                    [](Op& self, const Ciphertext& input,
                       const SwitchKey& switch_key, nb::handle options)
                    {
                        return run(
                            options,
                            [&](const Options& opts)
                            {
                                Ciphertext output(context_of(self), opts);
                                self.keyswitch(input, output, switch_key, opts);
                                return output;
                            });
                    },
                    "input"_a, "switch_key"_a, "options"_a = nb::none(),
                    "Switches input to the secret key of the switch key: a "
                    "ciphertext of the old secret key becomes one of the new "
                    "secret key.")
                .def(
                    "mod_drop",
                    [](Op& self, const Ciphertext& input, nb::handle options)
                    {
                        return run(options, [&](const Options& opts)
                                   { return self.mod_drop(input, opts); });
                    },
                    "input"_a, "options"_a = nb::none(),
                    "Drops the last prime of the ciphertext without "
                    "rescaling: the level goes down by one and the scale is "
                    "kept (in the automatic rescale mode, the result has the "
                    "scale of the new level).")
                .def(
                    "mod_drop",
                    [](Op& self, const Plaintext& input, nb::handle options)
                    {
                        return run(options, [&](const Options& opts)
                                   { return self.mod_drop(input, opts); });
                    },
                    "input"_a, "options"_a = nb::none(),
                    "Drops the last prime of the plaintext.")
                .def(
                    "mod_drop_inplace",
                    [](Op& self, Ciphertext& input, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.mod_drop_inplace(input, opts); });
                    },
                    "input"_a, "options"_a = nb::none(),
                    "Drops the last prime of the ciphertext in place.")
                .def(
                    "mod_drop_inplace",
                    [](Op& self, Plaintext& input, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.mod_drop_inplace(input, opts); });
                    },
                    "input"_a, "options"_a = nb::none(),
                    "Drops the last prime of the plaintext in place.");
        }

        // --------------------------------------------------------------
        // Gates of LogicOperator
        // --------------------------------------------------------------

        using GateCipher = void (LogicOperator::*)(const Ciphertext&,
                                                   const Ciphertext&,
                                                   Ciphertext&, const RelinKey&,
                                                   const Options&);
        using GatePlain = void (LogicOperator::*)(const Ciphertext&,
                                                  const Plaintext&, Ciphertext&,
                                                  const Options&);
        using GateCipherInplace = void (LogicOperator::*)(Ciphertext&,
                                                          const Ciphertext&,
                                                          const RelinKey&,
                                                          const Options&);
        using GatePlainInplace = void (LogicOperator::*)(Ciphertext&,
                                                         const Plaintext&,
                                                         const Options&);
        using GateBootstrapping =
            Ciphertext (LogicOperator::*)(const Ciphertext&, const Ciphertext&,
                                          BootstrappingKeys&, const Options&);

        struct Gate
        {
            const char* name;
            const char* inplace_name;
            const char* bootstrapping_name;
            const char* doc;
            GateCipher cipher;
            GatePlain plain;
            GateCipherInplace cipher_inplace;
            GatePlainInplace plain_inplace;
            GateBootstrapping bootstrapping;
        };

        void def_gate(nb::class_<LogicOperator>& cls, const Gate& gate)
        {
            const GateCipher cipher = gate.cipher;
            const GatePlain plain = gate.plain;
            const GateCipherInplace cipher_inplace = gate.cipher_inplace;
            const GatePlainInplace plain_inplace = gate.plain_inplace;
            const GateBootstrapping bootstrapping = gate.bootstrapping;

            cls.def(
                   gate.name,
                   [cipher](LogicOperator& self, const Ciphertext& input1,
                            const Ciphertext& input2, const RelinKey& relin_key,
                            nb::handle options)
                   {
                       return run(options,
                                  [&](const Options& opts)
                                  {
                                      Ciphertext output(context_of(self), opts);
                                      (self.*cipher)(input1, input2, output,
                                                     relin_key, opts);
                                      return output;
                                  });
                   },
                   "input1"_a, "input2"_a, "relin_key"_a,
                   "options"_a = nb::none(), gate.doc)
                .def(
                    gate.name,
                    [plain](LogicOperator& self, const Ciphertext& input1,
                            const Plaintext& input2, nb::handle options)
                    {
                        return run(
                            options,
                            [&](const Options& opts)
                            {
                                Ciphertext output(context_of(self), opts);
                                (self.*plain)(input1, input2, output, opts);
                                return output;
                            });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "The gate with a plaintext bit vector as second input; "
                    "no relinearization key is needed.")
                .def(
                    gate.inplace_name,
                    [cipher_inplace](LogicOperator& self, Ciphertext& input1,
                                     const Ciphertext& input2,
                                     const RelinKey& relin_key,
                                     nb::handle options)
                    {
                        run(options,
                            [&](const Options& opts) {
                                (self.*cipher_inplace)(input1, input2,
                                                       relin_key, opts);
                            });
                    },
                    "input1"_a, "input2"_a, "relin_key"_a,
                    "options"_a = nb::none(),
                    "The gate in place: input1 gets the result.")
                .def(
                    gate.inplace_name,
                    [plain_inplace](LogicOperator& self, Ciphertext& input1,
                                    const Plaintext& input2, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { (self.*plain_inplace)(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "The gate in place with a plaintext second input.")
                .def(
                    gate.bootstrapping_name,
                    [bootstrapping](LogicOperator& self,
                                    const Ciphertext& input1,
                                    const Ciphertext& input2,
                                    BootstrappingKeys& keys, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts) {
                                       return (self.*bootstrapping)(
                                           input1, input2, keys, opts);
                                   });
                    },
                    "input1"_a, "input2"_a, "keys"_a, "options"_a = nb::none(),
                    "Evaluates the gate inside gate bootstrapping: the result "
                    "is a fresh bit at the output level of the bootstrapping. "
                    "Needs generate_bootstrapping_params() with a gate "
                    "configuration.");
        }

        // --------------------------------------------------------------
        // ArithmeticOperator
        // --------------------------------------------------------------

        void bind_arithmetic_operator(nb::module_& m)
        {
            nb::class_<ArithmeticOperator> cls(
                m, "ArithmeticOperator",
                "Homomorphic operations on CKKS ciphertexts: arithmetic, "
                "rescaling, rotations, key switching, polynomial evaluation, "
                "the encoding transforms and bootstrapping. In the automatic "
                "rescale mode the operations rescale and align levels and "
                "scales by themselves.");

            cls.def(nb::init<ContextPtr, Encoder&>(), "context"_a, "encoder"_a,
                    "Creates the operator. The encoder is only read here, it "
                    "does not need to outlive the operator.");

            // Ciphertext-ciphertext arithmetic -------------------------------

            cls.def(
                   "add",
                   [](ArithmeticOperator& self, const Ciphertext& input1,
                      const Ciphertext& input2, nb::handle options)
                   {
                       return run(options, [&](const Options& opts)
                                  { return self.add(input1, input2, opts); });
                   },
                   "input1"_a, "input2"_a, "options"_a = nb::none(),
                   "Returns input1 + input2.")
                .def(
                    "add_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       const Ciphertext& input2, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.add_inplace(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Adds input2 to input1.")
                .def(
                    "sub",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       const Ciphertext& input2, nb::handle options)
                    {
                        return run(options, [&](const Options& opts)
                                   { return self.sub(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Returns input1 - input2.")
                .def(
                    "sub_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       const Ciphertext& input2, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.sub_inplace(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Subtracts input2 from input1.")
                .def(
                    "negate",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       nb::handle options)
                    {
                        return run(options, [&](const Options& opts)
                                   { return self.negate(input, opts); });
                    },
                    "input"_a, "options"_a = nb::none(), "Returns -input.")
                .def(
                    "negate_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input,
                       nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.negate_inplace(input, opts); });
                    },
                    "input"_a, "options"_a = nb::none(), "Negates input.")
                .def(
                    "multiply",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       const Ciphertext& input2, nb::handle options)
                    {
                        return run(
                            options, [&](const Options& opts)
                            { return self.multiply(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Returns input1 * input2, with three polynomials: "
                    "relinearize it before the next multiplication and "
                    "rescale it (the automatic rescale mode rescales by "
                    "itself).")
                .def(
                    "multiply_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       const Ciphertext& input2, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.multiply_inplace(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Multiplies input1 by input2.");

            // Plaintext and constant arithmetic ------------------------------

            cls.def(
                   "add_plain",
                   [](ArithmeticOperator& self, const Ciphertext& input1,
                      const Plaintext& input2, nb::handle options)
                   {
                       return run(
                           options, [&](const Options& opts)
                           { return self.add_plain(input1, input2, opts); });
                   },
                   "input1"_a, "input2"_a, "options"_a = nb::none(),
                   "Returns input1 + input2 for a plaintext input2.")
                .def(
                    "add_plain",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       double input2, nb::handle options)
                    {
                        return run(
                            options, [&](const Options& opts)
                            { return self.add_plain(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Adds a real constant to every slot.")
                .def(
                    "add_plain",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       std::complex<double> input2, nb::handle options)
                    {
                        Complex64 value = to_complex64(input2);
                        return run(options,
                                   [&](const Options& opts)
                                   {
                                       Ciphertext output(context_of(self),
                                                         opts);
                                       self.add_plain_complex(input1, value,
                                                              output, opts);
                                       return output;
                                   });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Adds a complex constant to every slot.")
                .def(
                    "add_plain_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       const Plaintext& input2, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.add_plain_inplace(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Adds a plaintext to input1.")
                .def(
                    "add_plain_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       double input2, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.add_plain_inplace(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Adds a real constant to every slot of input1.")
                .def(
                    "add_plain_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       std::complex<double> input2, nb::handle options)
                    {
                        Complex64 value = to_complex64(input2);
                        run(options,
                            [&](const Options& opts) {
                                self.add_plain_complex(input1, value, input1,
                                                       opts);
                            });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Adds a complex constant to every slot of input1.")
                .def(
                    "sub_plain",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       const Plaintext& input2, nb::handle options)
                    {
                        return run(
                            options, [&](const Options& opts)
                            { return self.sub_plain(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Returns input1 - input2 for a plaintext input2.")
                .def(
                    "sub_plain",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       double input2, nb::handle options)
                    {
                        return run(
                            options, [&](const Options& opts)
                            { return self.sub_plain(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Subtracts a real constant from every slot.")
                .def(
                    "sub_plain",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       std::complex<double> input2, nb::handle options)
                    {
                        Complex64 value = to_complex64(-input2);
                        return run(options,
                                   [&](const Options& opts)
                                   {
                                       Ciphertext output(context_of(self),
                                                         opts);
                                       self.add_plain_complex(input1, value,
                                                              output, opts);
                                       return output;
                                   });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Subtracts a complex constant from every slot.")
                .def(
                    "sub_plain_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       const Plaintext& input2, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.sub_plain_inplace(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Subtracts a plaintext from input1.")
                .def(
                    "sub_plain_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       double input2, nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.sub_plain_inplace(input1, input2, opts); });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Subtracts a real constant from every slot of input1.")
                .def(
                    "sub_plain_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       std::complex<double> input2, nb::handle options)
                    {
                        Complex64 value = to_complex64(-input2);
                        run(options,
                            [&](const Options& opts) {
                                self.add_plain_complex(input1, value, input1,
                                                       opts);
                            });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Subtracts a complex constant from every slot of input1.")
                .def(
                    "multiply_plain",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       const Plaintext& input2, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.multiply_plain(input1,
                                                                  input2, opts);
                                   });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Returns input1 * input2 for a plaintext input2; rescale "
                    "the result (the automatic rescale mode rescales by "
                    "itself).")
                .def(
                    "multiply_plain",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       double input2, std::optional<double> scale,
                       nb::handle options)
                    {
                        double s = scale ? *scale : input1.scale();
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.multiply_plain(
                                           input1, input2, s, opts);
                                   });
                    },
                    "input1"_a, "input2"_a, "scale"_a = nb::none(),
                    "options"_a = nb::none(),
                    "Multiplies every slot by a real constant encoded at "
                    "scale (default: the scale of input1); the result needs a "
                    "rescaling. In the automatic rescale mode the scale of "
                    "the level is used and scale is ignored.")
                .def(
                    "multiply_plain",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       std::complex<double> input2, nb::handle options)
                    {
                        Complex64 value = to_complex64(input2);
                        return run(options,
                                   [&](const Options& opts)
                                   {
                                       Ciphertext output(context_of(self),
                                                         opts);
                                       self.multiply_plain_complex(
                                           input1, value, output, opts);
                                       return output;
                                   });
                    },
                    "input1"_a, "input2"_a, nb::kw_only(),
                    "options"_a = nb::none(),
                    "Multiplies every slot by a complex constant. A Gaussian "
                    "integer keeps the scale; otherwise the constant is "
                    "encoded at the prime of the level, which multiplies the "
                    "scale by it.")
                .def(
                    "multiply_plain_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       const Plaintext& input2, nb::handle options)
                    {
                        run(options,
                            [&](const Options& opts) {
                                self.multiply_plain_inplace(input1, input2,
                                                            opts);
                            });
                    },
                    "input1"_a, "input2"_a, "options"_a = nb::none(),
                    "Multiplies input1 by a plaintext.")
                .def(
                    "multiply_plain_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       double input2, std::optional<double> scale,
                       nb::handle options)
                    {
                        double s = scale ? *scale : input1.scale();
                        run(options,
                            [&](const Options& opts) {
                                self.multiply_plain_inplace(input1, input2, s,
                                                            opts);
                            });
                    },
                    "input1"_a, "input2"_a, "scale"_a = nb::none(),
                    "options"_a = nb::none(),
                    "Multiplies every slot of input1 by a real constant (see "
                    "multiply_plain).")
                .def(
                    "multiply_plain_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       std::complex<double> input2, nb::handle options)
                    {
                        Complex64 value = to_complex64(input2);
                        run(options,
                            [&](const Options& opts) {
                                self.multiply_plain_complex(input1, value,
                                                            input1, opts);
                            });
                    },
                    "input1"_a, "input2"_a, nb::kw_only(),
                    "options"_a = nb::none(),
                    "Multiplies every slot of input1 by a complex constant.")
                .def(
                    "scale_up",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       double scale, nb::handle options)
                    {
                        return run(
                            options,
                            [&](const Options& opts)
                            {
                                Ciphertext output(context_of(self), opts);
                                self.scale_up(input, scale, output, opts);
                                return output;
                            });
                    },
                    "input"_a, "scale"_a, "options"_a = nb::none(),
                    "Multiplies the data and the scale of input by an "
                    "integer factor; the values do not change.")
                .def(
                    "mult_i",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts)
                                   {
                                       Ciphertext output(context_of(self),
                                                         opts);
                                       self.mult_i(input, output, opts);
                                       return output;
                                   });
                    },
                    "input"_a, "options"_a = nb::none(),
                    "Returns input * i (the imaginary unit), without using a "
                    "level.")
                .def(
                    "div_i",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts)
                                   {
                                       Ciphertext output(context_of(self),
                                                         opts);
                                       self.div_i(input, output, opts);
                                       return output;
                                   });
                    },
                    "input"_a, "options"_a = nb::none(),
                    "Returns input / i, without using a level.");

            // Relinearization and rescaling ----------------------------------

            cls.def(
                   "relinearize",
                   [](ArithmeticOperator& self, const Ciphertext& input,
                      const RelinKey& relin_key, nb::handle options)
                   {
                       return run(options,
                                  [&](const Options& opts)
                                  {
                                      Ciphertext output = copy_for(input, opts);
                                      self.relinearize_inplace(output,
                                                               relin_key, opts);
                                      return output;
                                  });
                   },
                   "input"_a, "relin_key"_a, "options"_a = nb::none(),
                   "Returns input relinearized: a product of three "
                   "polynomials brought back to two.")
                .def(
                    "relinearize_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input,
                       const RelinKey& relin_key, nb::handle options)
                    {
                        run(options,
                            [&](const Options& opts) {
                                self.relinearize_inplace(input, relin_key,
                                                         opts);
                            });
                    },
                    "input"_a, "relin_key"_a, "options"_a = nb::none(),
                    "Relinearizes input in place.")
                .def(
                    "rescale",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts)
                                   {
                                       Ciphertext output =
                                           copy_for(input, opts);
                                       self.rescale_inplace(output, opts);
                                       return output;
                                   });
                    },
                    "input"_a, "options"_a = nb::none(),
                    "Returns input divided by the last prime: the level goes "
                    "down by one and the scale is divided by the prime.")
                .def(
                    "rescale_inplace",
                    [](ArithmeticOperator& self, Ciphertext& input,
                       nb::handle options)
                    {
                        run(options, [&](const Options& opts)
                            { self.rescale_inplace(input, opts); });
                    },
                    "input"_a, "options"_a = nb::none(),
                    "Rescales input in place.");

            // Rotations and key switching ------------------------------------

            def_shared_operations<ArithmeticOperator>(cls);

            cls.def(
                   "rotate_hoisted",
                   [](ArithmeticOperator& self, const Ciphertext& input,
                      const GaloisKey& galois_key, nb::handle shifts,
                      nb::handle options)
                   {
                       std::vector<int> steps = to_int_vector(shifts, "shifts");
                       return run(options,
                                  [&](const Options& opts) {
                                      return self.rotate_hoisted(
                                          input, galois_key, steps, opts);
                                  });
                   },
                   "input"_a, "galois_key"_a, "shifts"_a,
                   "options"_a = nb::none(),
                   "Returns the rotations of input by each of the shifts, "
                   "sharing one decomposition of input (faster than separate "
                   "rotations). The Galois key must have a key for every "
                   "nonzero shift.")
                .def(
                    "conjugate",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       const GaloisKey& galois_key, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.conjugate(input, galois_key,
                                                             opts);
                                   });
                    },
                    "input"_a, "galois_key"_a, "options"_a = nb::none(),
                    "Returns the complex conjugate of every slot. Every "
                    "Galois key has the conjugation key.")
                .def(
                    "sum_slots",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       const GaloisKey& galois_key, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.sum_slots(input, galois_key,
                                                             opts);
                                   });
                    },
                    "input"_a, "galois_key"_a, "options"_a = nb::none(),
                    "Returns a ciphertext whose slots all hold the sum of the "
                    "slots of input, with rotations by the powers of two.")
                .def(
                    "inner_product",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       const Ciphertext& input2, const RelinKey& relin_key,
                       const GaloisKey& galois_key, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.inner_product(
                                           input1, input2, relin_key,
                                           galois_key, opts);
                                   });
                    },
                    "input1"_a, "input2"_a, "relin_key"_a, "galois_key"_a,
                    "options"_a = nb::none(),
                    "Returns a ciphertext whose slots all hold the inner "
                    "product of the slots of input1 and input2.")
                .def(
                    "inner_product",
                    [](ArithmeticOperator& self, const Ciphertext& input1,
                       const Plaintext& input2, const GaloisKey& galois_key,
                       nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.inner_product(
                                           input1, input2, galois_key, opts);
                                   });
                    },
                    "input1"_a, "input2"_a, "galois_key"_a,
                    "options"_a = nb::none(),
                    "The inner product with a plaintext; no relinearization "
                    "key is needed.");

            // Polynomial evaluation ------------------------------------------

            cls.def(
                   "evaluate_polynomial",
                   [](ArithmeticOperator& self, const Ciphertext& input,
                      nb::handle coefficients, const RelinKey& relin_key,
                      nb::handle options)
                   {
                       std::vector<double> values =
                           to_double_vector(coefficients, "coefficients");
                       return run(options,
                                  [&](const Options& opts) {
                                      return self.evaluate_polynomial(
                                          input, values, relin_key, opts);
                                  });
                   },
                   "input"_a, "coefficients"_a, "relin_key"_a,
                   "options"_a = nb::none(),
                   "Evaluates sum(coefficients[k] * input^k) in the monomial "
                   "basis. It uses polynomial_depth(degree) levels and the "
                   "result keeps the scale of input.")
                .def(
                    "evaluate_chebyshev",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       nb::callable function, double a, double b, int degree,
                       const RelinKey& relin_key, nb::handle options)
                    {
                        std::function<double(double)> f =
                            real_function(function);
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.evaluate_chebyshev(
                                           input, f, a, b, degree, relin_key,
                                           opts);
                                   });
                    },
                    "input"_a, "function"_a, "a"_a, "b"_a, "degree"_a,
                    "relin_key"_a, "options"_a = nb::none(),
                    "Evaluates the Chebyshev interpolant of degree degree of "
                    "a real function (a Python callable of one float) on the "
                    "interval [a, b]; the slots must be in [a, b].")
                .def(
                    "evaluate_chebyshev",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       nb::handle coefficients, double a, double b,
                       const RelinKey& relin_key, nb::handle options)
                    {
                        std::vector<double> values =
                            to_double_vector(coefficients, "coefficients");
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.evaluate_chebyshev(
                                           input, values, a, b, relin_key,
                                           opts);
                                   });
                    },
                    "input"_a, "coefficients"_a, "a"_a, "b"_a, "relin_key"_a,
                    "options"_a = nb::none(),
                    "Evaluates sum(coefficients[k] * T_k(y)) with the "
                    "Chebyshev polynomials T_k of y = (2 * input - a - b) / "
                    "(b - a), for slots in [a, b].")
                .def_static("polynomial_depth",
                            &ArithmeticOperator::polynomial_depth, "degree"_a,
                            "chebyshev"_a = false, "a"_a = -1.0, "b"_a = 1.0,
                            "Returns the number of levels a polynomial of "
                            "this degree uses (a Chebyshev series on [a, b] "
                            "if chebyshev is True).");

            // Encoding transforms ---------------------------------------------

            cls.def(
                   "generate_encoding_transform_context",
                   [](ArithmeticOperator& self,
                      std::optional<TransformConfig> config)
                   {
                       TransformConfig used =
                           config ? *config : TransformConfig();
                       TransformContext context;
                       nb::gil_scoped_release release;
                       self.generate_encoding_transform_context(context, used);
                       return context;
                   },
                   "config"_a = nb::none(),
                   "Precomputes the CoeffToSlot and SlotToCoeff matrices of "
                   "config (default: EncodingTransformConfig()). The Galois "
                   "key of the transforms needs the shifts of the result's "
                   "key_indices. Manual rescale mode only.")
                .def(
                    "coeff_to_slot",
                    [](ArithmeticOperator& self, Ciphertext& input,
                       const GaloisKey& galois_key,
                       TransformContext* transform_context, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts)
                                   {
                                       if (transform_context)
                                       {
                                           return self.coeff_to_slot(
                                               input, galois_key,
                                               *transform_context, opts);
                                       }
                                       return self.coeff_to_slot(
                                           input, galois_key, opts);
                                   });
                    },
                    "input"_a, "galois_key"_a,
                    "transform_context"_a = nb::none(),
                    "options"_a = nb::none(),
                    "Moves the coefficients of a coefficient-encoded "
                    "ciphertext into the slots of two slot-encoded "
                    "ciphertexts. Without a transform context, the "
                    "matrices of generate_bootstrapping_params() are used. "
                    "Manual rescale mode only.")
                .def(
                    "slot_to_coeff",
                    [](ArithmeticOperator& self, Ciphertext& input1,
                       Ciphertext& input2, const GaloisKey& galois_key,
                       TransformContext* transform_context, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts)
                                   {
                                       if (transform_context)
                                       {
                                           return self.slot_to_coeff(
                                               input1, input2, galois_key,
                                               *transform_context, opts);
                                       }
                                       return self.slot_to_coeff(
                                           input1, input2, galois_key, opts);
                                   });
                    },
                    "input1"_a, "input2"_a, "galois_key"_a,
                    "transform_context"_a = nb::none(),
                    "options"_a = nb::none(),
                    "The inverse of coeff_to_slot: returns the "
                    "coefficient-encoded ciphertext of the two slot-encoded "
                    "ciphertexts. Manual rescale mode only.");

            // Bootstrapping ---------------------------------------------------

            cls.def(
                   "generate_bootstrapping_params",
                   [](ArithmeticOperator& self, double scale,
                      const heongpu::BootstrappingConfig& config,
                      heongpu::arithmetic_bootstrapping_type boot_type)
                   {
                       nb::gil_scoped_release release;
                       self.generate_bootstrapping_params(scale, config,
                                                          boot_type);
                   },
                   "scale"_a, "config"_a,
                   "boot_type"_a =
                       heongpu::arithmetic_bootstrapping_type::regular,
                   "Precomputes the matrices and polynomials of regular or "
                   "slim bootstrapping with this configuration.")
                .def(
                    "generate_bootstrapping_params",
                    [](ArithmeticOperator& self,
                       const heongpu::CKKSBootstrappingPreset& preset)
                    {
                        nb::gil_scoped_release release;
                        self.generate_bootstrapping_params(preset);
                    },
                    "preset"_a,
                    "Precomputes bootstrapping for a regular or slim preset "
                    "(the context must have been created with it). Raises "
                    "ParameterError for a logic preset.")
                .def("bootstrapping_key_indices",
                     &ArithmeticOperator::bootstrapping_key_indices,
                     "Returns the rotation steps the Galois key of "
                     "bootstrapping needs (for "
                     "KeyGenerator.bootstrapping_keys()). Raises ValueError "
                     "before generate_bootstrapping_params().")
                .def(
                    "regular_bootstrapping",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       BootstrappingKeys& keys, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.regular_bootstrapping(
                                           input, keys, opts);
                                   });
                    },
                    "input"_a, "keys"_a, "options"_a = nb::none(),
                    "Refreshes the levels of a ciphertext of real or complex "
                    "values (ModRaise, CoeffToSlot, EvalMod, SlotToCoeff). "
                    "An input above the input level is brought down to it.")
                .def(
                    "slim_bootstrapping",
                    [](ArithmeticOperator& self, const Ciphertext& input,
                       BootstrappingKeys& keys, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.slim_bootstrapping(
                                           input, keys, opts);
                                   });
                    },
                    "input"_a, "keys"_a, "options"_a = nb::none(),
                    "Refreshes the levels of a ciphertext of real values "
                    "(SlotToCoeff, ModRaise, CoeffToSlot, EvalMod). The input "
                    "must be at the SlotToCoeff start level or above; a lower "
                    "one raises LevelMismatchError.");
        }

        // --------------------------------------------------------------
        // LogicOperator
        // --------------------------------------------------------------

        void bind_logic_operator(nb::module_& m)
        {
            nb::class_<LogicOperator> cls(
                m, "LogicOperator",
                "Boolean gates on CKKS ciphertexts whose slots hold bits (0 "
                "or 1), and bit and gate bootstrapping. The gates without "
                "bootstrapping work in the manual rescale mode only.");

            cls.def(nb::init<ContextPtr, Encoder&, double>(), "context"_a,
                    "encoder"_a, "scale"_a,
                    "Creates the operator; scale is the scale of the "
                    "ciphertexts. The encoder is only read here.");

            cls.def(
                   "NOT",
                   [](LogicOperator& self, const Ciphertext& input,
                      nb::handle options)
                   {
                       return run(options,
                                  [&](const Options& opts)
                                  {
                                      Ciphertext output(context_of(self), opts);
                                      self.NOT(input, output, opts);
                                      return output;
                                  });
                   },
                   "input"_a, "options"_a = nb::none(), "Returns 1 - input.")
                .def(
                    "NOT_inplace",
                    [](LogicOperator& self, Ciphertext& input,
                       nb::handle options) {
                        run(options, [&](const Options& opts)
                            { self.NOT_inplace(input, opts); });
                    },
                    "input"_a, "options"_a = nb::none(),
                    "Replaces input by 1 - input.");

            const Gate gates[] = {
                {"AND", "AND_inplace", "AND_bootstrapping",
                 "Returns input1 AND input2 (input1 * input2); uses a level.",
                 &LogicOperator::AND, &LogicOperator::AND,
                 &LogicOperator::AND_inplace, &LogicOperator::AND_inplace,
                 &LogicOperator::AND_bootstrapping},
                {"OR", "OR_inplace", "OR_bootstrapping",
                 "Returns input1 OR input2 (input1 + input2 - input1 * "
                 "input2); uses a level.",
                 &LogicOperator::OR, &LogicOperator::OR,
                 &LogicOperator::OR_inplace, &LogicOperator::OR_inplace,
                 &LogicOperator::OR_bootstrapping},
                {"XOR", "XOR_inplace", "XOR_bootstrapping",
                 "Returns input1 XOR input2 (input1 + input2 - 2 * input1 * "
                 "input2); uses a level.",
                 &LogicOperator::XOR, &LogicOperator::XOR,
                 &LogicOperator::XOR_inplace, &LogicOperator::XOR_inplace,
                 &LogicOperator::XOR_bootstrapping},
                {"NAND", "NAND_inplace", "NAND_bootstrapping",
                 "Returns NOT (input1 AND input2); uses a level.",
                 &LogicOperator::NAND, &LogicOperator::NAND,
                 &LogicOperator::NAND_inplace, &LogicOperator::NAND_inplace,
                 &LogicOperator::NAND_bootstrapping},
                {"NOR", "NOR_inplace", "NOR_bootstrapping",
                 "Returns NOT (input1 OR input2); uses a level.",
                 &LogicOperator::NOR, &LogicOperator::NOR,
                 &LogicOperator::NOR_inplace, &LogicOperator::NOR_inplace,
                 &LogicOperator::NOR_bootstrapping},
                {"XNOR", "XNOR_inplace", "XNOR_bootstrapping",
                 "Returns NOT (input1 XOR input2); uses a level.",
                 &LogicOperator::XNOR, &LogicOperator::XNOR,
                 &LogicOperator::XNOR_inplace, &LogicOperator::XNOR_inplace,
                 &LogicOperator::XNOR_bootstrapping},
            };
            for (const Gate& gate : gates)
            {
                def_gate(cls, gate);
            }

            def_shared_operations<LogicOperator>(cls);

            cls.def(
                   "generate_bootstrapping_params",
                   [](LogicOperator& self, double scale,
                      const heongpu::BootstrappingConfig& config,
                      heongpu::logic_bootstrapping_type boot_type)
                   {
                       nb::gil_scoped_release release;
                       self.generate_bootstrapping_params(scale, config,
                                                          boot_type);
                   },
                   "scale"_a, "config"_a, "boot_type"_a,
                   "Precomputes the matrices and polynomials of bit or gate "
                   "bootstrapping with this configuration.")
                .def(
                    "generate_bootstrapping_params",
                    [](LogicOperator& self,
                       const heongpu::CKKSBootstrappingPreset& preset)
                    {
                        nb::gil_scoped_release release;
                        self.generate_bootstrapping_params(preset);
                    },
                    "preset"_a,
                    "Precomputes bootstrapping for a bit or gate preset (the "
                    "context must have been created with it). Raises "
                    "ParameterError for an arithmetic preset.")
                .def("bootstrapping_key_indices",
                     &LogicOperator::bootstrapping_key_indices,
                     "Returns the rotation steps the Galois key of "
                     "bootstrapping needs. Raises ValueError before "
                     "generate_bootstrapping_params().")
                .def(
                    "bit_bootstrapping",
                    [](LogicOperator& self, const Ciphertext& input,
                       BootstrappingKeys& keys, nb::handle options)
                    {
                        return run(options,
                                   [&](const Options& opts) {
                                       return self.bit_bootstrapping(
                                           input, keys, opts);
                                   });
                    },
                    "input"_a, "keys"_a, "options"_a = nb::none(),
                    "Refreshes a ciphertext of bits: the result holds the "
                    "same bits, cleaned of noise, at the output level. Needs "
                    "generate_bootstrapping_params() with a bit "
                    "configuration.");
        }

        // --------------------------------------------------------------
        // Encoding transforms, precision and Chebyshev helpers
        // --------------------------------------------------------------

        void bind_transform(nb::module_& m)
        {
            nb::class_<TransformConfig>(
                m, "EncodingTransformConfig",
                "Configuration of the stand-alone CoeffToSlot and SlotToCoeff "
                "transforms: the number of pieces (levels, 2 to 5) of each "
                "and the depth they start at (0 is the top level; -1 "
                "chooses 0 for CoeffToSlot and cts_piece for SlotToCoeff).")
                .def(nb::init<int, int, int, int>(), "cts_piece"_a = 3,
                     "stc_piece"_a = 3, "cts_start_level"_a = -1,
                     "stc_start_level"_a = -1)
                .def_rw("cts_piece", &TransformConfig::CtoS_piece_)
                .def_rw("stc_piece", &TransformConfig::StoC_piece_)
                .def_rw("cts_start_level", &TransformConfig::CtoS_start_level_)
                .def_rw("stc_start_level", &TransformConfig::StoC_start_level_)
                .def("__repr__",
                     [](const TransformConfig& self)
                     {
                         std::ostringstream stream;
                         stream
                             << "EncodingTransformConfig(cts_piece="
                             << self.CtoS_piece_
                             << ", stc_piece=" << self.StoC_piece_
                             << ", cts_start_level=" << self.CtoS_start_level_
                             << ", stc_start_level=" << self.StoC_start_level_
                             << ")";
                         return stream.str();
                     });

            nb::class_<TransformContext>(
                m, "EncodingTransformContext",
                "The precomputed matrices of the stand-alone CoeffToSlot and "
                "SlotToCoeff transforms, from "
                "ArithmeticOperator.generate_encoding_transform_context().")
                .def(nb::init<>(),
                     "An empty context; generate it with "
                     "ArithmeticOperator.generate_encoding_transform_context()"
                     ".")
                .def_prop_ro("generated", [](const TransformContext& self)
                             { return self.generated_; })
                .def_prop_ro("cts_piece", [](const TransformContext& self)
                             { return self.CtoS_piece_; })
                .def_prop_ro("stc_piece", [](const TransformContext& self)
                             { return self.StoC_piece_; })
                .def_prop_ro(
                    "cts_level",
                    [](const TransformContext& self)
                    { return self.CtoS_level_; },
                    "The depth CoeffToSlot starts at.")
                .def_prop_ro(
                    "stc_level",
                    [](const TransformContext& self)
                    { return self.StoC_level_; },
                    "The depth SlotToCoeff starts at.")
                .def_prop_ro(
                    "key_indices",
                    [](const TransformContext& self)
                    { return self.key_indexs_; },
                    "The rotation steps the Galois key of the transforms "
                    "needs.");
        }

        void bind_precision(nb::module_& m)
        {
            nb::class_<heongpu::Stats>(
                m, "Stats",
                "A statistic of the real parts, the imaginary parts and the "
                "L2 norm.")
                .def(nb::init<>())
                .def(nb::init<double, double, double>(), "real"_a, "imag"_a,
                     "l2"_a)
                .def_rw("real", &heongpu::Stats::real)
                .def_rw("imag", &heongpu::Stats::imag)
                .def_rw("l2", &heongpu::Stats::l2)
                .def("__repr__",
                     [](const heongpu::Stats& self)
                     {
                         std::ostringstream stream;
                         stream << "Stats(real=" << self.real
                                << ", imag=" << self.imag << ", l2=" << self.l2
                                << ")";
                         return stream.str();
                     });

            nb::class_<heongpu::DistEntry>(
                m, "DistEntry",
                "A point of a precision distribution: a precision in bits "
                "and the number of values at it.")
                .def(nb::init<>())
                .def_rw("prec", &heongpu::DistEntry::prec)
                .def_rw("count", &heongpu::DistEntry::count)
                .def("__repr__",
                     [](const heongpu::DistEntry& self)
                     {
                         std::ostringstream stream;
                         stream << "DistEntry(prec=" << self.prec
                                << ", count=" << self.count << ")";
                         return stream.str();
                     });

            using PrecisionStats = heongpu::PrecisionStats;
            nb::class_<PrecisionStats>(
                m, "PrecisionStats",
                "Precision of approximate values against the expected ones: "
                "the deltas (absolute errors) and the precisions in bits "
                "(log2 of 1 / delta), from get_precision_stats().")
                .def(nb::init<>())
                .def_rw("max_delta", &PrecisionStats::max_delta)
                .def_rw("min_delta", &PrecisionStats::min_delta)
                .def_rw("max_precision", &PrecisionStats::max_precision)
                .def_rw("min_precision", &PrecisionStats::min_precision)
                .def_rw("mean_delta", &PrecisionStats::mean_delta)
                .def_rw("mean_precision", &PrecisionStats::mean_precision)
                .def_rw("median_delta", &PrecisionStats::median_delta)
                .def_rw("median_precision", &PrecisionStats::median_precision)
                .def_rw("real_dist", &PrecisionStats::real_dist)
                .def_rw("imag_dist", &PrecisionStats::imag_dist)
                .def_rw("l2_dist", &PrecisionStats::l2_dist)
                .def_rw("cdf_resol", &PrecisionStats::cdf_resol,
                        "Resolution of the distributions.")
                .def("__str__", &PrecisionStats::to_string)
                .def("__repr__",
                     [](const PrecisionStats& self)
                     {
                         std::ostringstream stream;
                         stream << "<heongpu.ckks.PrecisionStats min="
                                << self.min_precision.real
                                << " mean=" << self.mean_precision.real
                                << " bits>";
                         return stream.str();
                     });

            m.def(
                "get_precision_stats",
                [](nb::handle values_want, nb::handle values_test)
                {
                    if (is_complex(values_want))
                    {
                        std::vector<Complex64> want =
                            to_complex_vector(values_want, "values_want");
                        std::vector<Complex64> test =
                            to_complex_vector(values_test, "values_test");
                        nb::gil_scoped_release release;
                        return heongpu::get_precision_stats(want, test);
                    }
                    std::vector<double> want =
                        to_double_vector(values_want, "values_want");
                    if (is_complex(values_test))
                    {
                        std::vector<Complex64> test =
                            to_complex_vector(values_test, "values_test");
                        nb::gil_scoped_release release;
                        return heongpu::get_precision_stats(want, test);
                    }
                    std::vector<double> test =
                        to_double_vector(values_test, "values_test");
                    nb::gil_scoped_release release;
                    return heongpu::get_precision_stats(want, test);
                },
                "values_want"_a, "values_test"_a,
                "Compares values_test (for example decrypted values) with the "
                "expected values_want, real or complex array-likes of the "
                "same length. Raises ValueError for different lengths.");

            m.def(
                "chebyshev_coefficients",
                [](nb::callable function, double a, double b, int degree)
                {
                    std::function<double(double)> f = real_function(function);
                    std::vector<double> coefficients;
                    {
                        nb::gil_scoped_release release;
                        coefficients =
                            heongpu::chebyshev_coefficients(f, a, b, degree);
                    }
                    return to_numpy(std::move(coefficients));
                },
                "function"_a, "a"_a, "b"_a, "degree"_a,
                "Returns the coefficients of the Chebyshev interpolant of "
                "degree degree of a real function on [a, b], for "
                "ArithmeticOperator.evaluate_chebyshev().");

            m.def(
                "chebyshev_nodes",
                [](int n, double a, double b)
                { return to_numpy(heongpu::chebyshev_nodes(n, a, b)); },
                "n"_a, "a"_a, "b"_a,
                "Returns the n Chebyshev nodes on [a, b].");
        }
    } // namespace

    void bind_ckks_operator(nb::module_& m)
    {
        bind_transform(m);
        bind_arithmetic_operator(m);
        bind_logic_operator(m);
        bind_precision(m);
    }

} // namespace heongpu_py
