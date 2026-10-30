# Contributing to HEonGPU-py

Thank you for your interest in contributing to HEonGPU-py! Bug reports,
feature requests, documentation improvements and code changes are welcome.

## Reporting bugs and suggesting features

Open an issue in the GitHub repository. For a bug, include the steps to
reproduce it, the expected and the actual behavior, the versions
(`heongpu.__version__`, `heongpu.heongpu_version`, Python, CUDA) and the GPU
model.

## Development build

Install HEonGPU 2.1 once (see its README), then build the extension module
with CMake and run the tests against the build:

```bash
python -m pip install nanobind numpy pytest

cmake -S . -B build -G Ninja \
    -D Python_EXECUTABLE="$(which python)" \
    -D CMAKE_PREFIX_PATH=/path/to/heongpu/install \
    -D HEONGPU_PY_HEONGPU_PROVIDER=SYSTEM
cmake --build build

# A package directory with the sources and the built module.
mkdir -p build/stage/heongpu
ln -sf "$PWD"/src/heongpu/* build/stage/heongpu/
ln -sf "$PWD"/build/_heongpu*.so build/stage/heongpu/

PYTHONPATH=build/stage python -m pytest
```

Alternatively, `pip install --no-build-isolation -e .` installs an editable
package that is rebuilt by `pip install` again.

The bindings are in `src/cpp`: one file per part (`core.cpp`,
`ckks_core.cpp`, `ckks_operator.cpp`, `ckks_session.cpp`, `ckks_mpc.cpp`, the
same for BFV, and `tfhe.cpp`), with the shared helpers in `common.h`. The
Python package is in `src/heongpu`.

## Code changes

- Work in a branch with a descriptive name (for example `fix-bug-456`).
- Follow the conventions of the existing bindings: class names without the
  scheme prefix inside the scheme modules, output parameters as return
  values, `options=None` as the last argument, the GIL released during GPU
  work.
- Add tests for your changes; mark long tests with `@pytest.mark.slow`.
- Format C++ code with `clang-format` (the `.clang-format` file) and Python
  code with `black`.
- Write clear commit messages that explain what the change does and why.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the license of the project.
