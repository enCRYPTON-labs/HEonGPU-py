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
python -m pip install "scikit-build-core>=0.11" "nanobind>=2.9" numpy pytest

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

Alternatively, install an editable package (run the command again after
changing the C++ sources):

```bash
CMAKE_PREFIX_PATH=/path/to/heongpu/install HEONGPU_PY_HEONGPU_PROVIDER=SYSTEM \
    python -m pip install --no-build-isolation -e .
```

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

## Releases

The wheels are built by [cibuildwheel](https://cibuildwheel.pypa.io) in a
`manylinux_2_28` container: `ci/manylinux_before_all.sh` installs CUDA 13.0
and the libraries of HEonGPU and builds HEonGPU once, and the settings are in
`[tool.cibuildwheel]` of `pyproject.toml`. To build them locally (with
Docker):

```bash
python -m pip install cibuildwheel
cibuildwheel --platform linux
```

The machines that build the wheels have no GPU, so test the wheels on one
before a release: install a wheel of `wheelhouse/` in a new virtual
environment and run the tests and the examples.

To release a version:

1. Set the version in `pyproject.toml` and update `CHANGELOG.md`.
2. Run the `Release` workflow by hand; it publishes the source distribution
   and the wheels to TestPyPI. Install them from there and test them.
3. Push a tag `v<version>` (for example `v0.1.0`); the workflow publishes the
   same files to PyPI.

PyPI does not accept a file again under a version it already has, so a
broken release needs a new version.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the license of the project.
