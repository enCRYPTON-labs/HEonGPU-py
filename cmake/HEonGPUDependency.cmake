# Copyright 2026 enCRYPTON-Labs
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

# Finds an installed HEonGPU or builds it from source, and sets
#   HEONGPU_PY_HEONGPU_TARGET   the HEonGPU target to link
#   HEONGPU_PY_HEONGPU_VERSION  its version
#   HEONGPU_PY_HEONGPU_FETCHED  whether it is built from source
# and the functions heongpu_py_configure_device_link() and
# heongpu_py_install_runtime_libraries().

set(_heongpu_py_required_version 2.1.1)

string(TOUPPER "${HEONGPU_PY_HEONGPU_PROVIDER}" _heongpu_py_provider)
if(NOT _heongpu_py_provider MATCHES "^(AUTO|SYSTEM|FETCH)$")
    message(FATAL_ERROR
        "HEONGPU_PY_HEONGPU_PROVIDER must be AUTO, SYSTEM or FETCH, not "
        "'${HEONGPU_PY_HEONGPU_PROVIDER}'.")
endif()

set(HEONGPU_PY_HEONGPU_FETCHED OFF)
if(NOT _heongpu_py_provider STREQUAL "FETCH")
    find_package(HEonGPU ${_heongpu_py_required_version} CONFIG QUIET)
    if(HEonGPU_FOUND)
        message(STATUS "Using the installed HEonGPU ${HEonGPU_VERSION} (${HEonGPU_DIR})")
        set(HEONGPU_PY_HEONGPU_TARGET HEonGPU::heongpu)
        set(HEONGPU_PY_HEONGPU_VERSION ${HEonGPU_VERSION})
    elseif(_heongpu_py_provider STREQUAL "SYSTEM")
        message(FATAL_ERROR
            "HEONGPU_PY_HEONGPU_PROVIDER is SYSTEM, but no installed HEonGPU "
            "${_heongpu_py_required_version} was found. Pass its install "
            "prefix with CMAKE_PREFIX_PATH (or its config directory with "
            "HEonGPU_DIR), or use AUTO or FETCH to build HEonGPU from source.")
    endif()
endif()

if(NOT HEONGPU_PY_HEONGPU_TARGET)
    # HEonGPU is built as a static library and linked into the extension
    # module. FETCHCONTENT_SOURCE_DIR_HEONGPU selects a local source tree
    # instead of the repository.
    message(STATUS
        "Building HEonGPU ${HEONGPU_PY_HEONGPU_TAG} from source "
        "(${HEONGPU_PY_HEONGPU_REPOSITORY})")
    include(FetchContent)
    set(HEonGPU_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(HEonGPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(HEonGPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HEonGPU_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        heongpu
        GIT_REPOSITORY ${HEONGPU_PY_HEONGPU_REPOSITORY}
        GIT_TAG        ${HEONGPU_PY_HEONGPU_TAG}
        GIT_SHALLOW    ON
        GIT_SUBMODULES_RECURSE ON
    )
    FetchContent_MakeAvailable(heongpu)
    set(HEONGPU_PY_HEONGPU_TARGET heongpu)
    # The version of HEonGPU, from its target.
    get_target_property(HEONGPU_PY_HEONGPU_VERSION heongpu VERSION)
    set(HEONGPU_PY_HEONGPU_FETCHED ON)
endif()

find_package(CUDAToolkit REQUIRED)

get_target_property(_heongpu_py_heongpu_type ${HEONGPU_PY_HEONGPU_TARGET} TYPE)
if(_heongpu_py_heongpu_type STREQUAL "STATIC_LIBRARY")
    # With a static HEonGPU, the module performs the CUDA device link.
    enable_language(CUDA)
endif()

# Performs the CUDA device link of a static HEonGPU in the given target.
function(heongpu_py_configure_device_link target)
    if(NOT _heongpu_py_heongpu_type STREQUAL "STATIC_LIBRARY")
        return()
    endif()
    if(HEONGPU_PY_HEONGPU_FETCHED)
        get_target_property(_architectures heongpu CUDA_ARCHITECTURES)
    elseif(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
        set(_architectures ${CMAKE_CUDA_ARCHITECTURES})
    else()
        # Without CMAKE_CUDA_ARCHITECTURES (or with the nvcc default 52),
        # build for the GPU of this machine. Pass CMAKE_CUDA_ARCHITECTURES to
        # build for other GPUs.
        set(_architectures native)
    endif()
    message(STATUS "CUDA architectures of the device link: ${_architectures}")
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_ARCHITECTURES "${_architectures}")
endfunction()

# Installs the shared libraries HEonGPU needs at run time (RMM and its logger)
# next to the module, or points the module at an installed HEonGPU.
function(heongpu_py_install_runtime_libraries)
    if(HEONGPU_PY_HEONGPU_FETCHED)
        set_target_properties(_heongpu PROPERTIES INSTALL_RPATH "$ORIGIN")
        install(FILES $<TARGET_FILE:rmm::rmm> $<TARGET_FILE:rapids_logger::rapids_logger>
            DESTINATION heongpu
            COMPONENT ${HEONGPU_PY_COMPONENT})
    else()
        # The libraries stay where HEonGPU installed them.
        set_target_properties(_heongpu PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH ON)
    endif()
endfunction()
