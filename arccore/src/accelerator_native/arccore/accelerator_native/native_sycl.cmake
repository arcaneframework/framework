# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Backend Arccore for SYCL

set(ARCCORE_SOURCES
  SyclAccelerator.cc
  SyclAccelerator.h
)

# For compatibility with existing code
if (DEFINED ARCANE_CXX_SYCL_FLAGS AND NOT DEFINED ARCCORE_CXX_SYCL_FLAGS)
  set(ARCCORE_CXX_SYCL_FLAGS "${ARCANE_CXX_SYCL_FLAGS}")
endif ()

# Created an interface target to propagate compilation options
# common for SYCL compilation

add_library(arccore_sycl_compile_flags INTERFACE)

target_compile_options(arccore_sycl_compile_flags INTERFACE
  # No specific option for now
)
if (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
  target_link_options(arccore_sycl_compile_flags INTERFACE "-lsycl")
endif ()

install(TARGETS arccore_sycl_compile_flags EXPORT ArccoreTargets)

arccore_add_component_library(accelerator_native
  LIB_NAME arccore_accelerator_sycl
  FILES ${ARCCORE_SOURCES}
)
target_compile_options(arccore_accelerator_sycl PRIVATE "${ARCCORE_CXX_SYCL_FLAGS}")

target_link_libraries(arccore_accelerator_sycl PUBLIC
  arccore_base
  arccore_sycl_compile_flags
)
target_link_options(arccore_accelerator_sycl PUBLIC "${ARCCORE_CXX_SYCL_FLAGS}")

# Detects oneDPL if using DPC++
# We only perform detection and assume it is consistent with the
# compiler. We should not add the 'oneDPL' target to 'arccore_accelerator_sycl'
# because this adds compilation flags (notably related to OpenMP) which
# can cause compilation errors. Furthermore, there may be
# inconsistencies with the TBB version used elsewhere in Arccore.
if (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
  find_package(oneDPL CONFIG)
  message(STATUS "[Sycl] oneDPL found?=${oneDPL_FOUND} Version=${oneDPL_VERSION}")
  if (oneDPL_FOUND)
    target_compile_definitions(arccore_accelerator_sycl PUBLIC ARCCORE_HAS_ONEDPL ARCANE_HAS_ONEDPL)
  endif ()
endif ()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
