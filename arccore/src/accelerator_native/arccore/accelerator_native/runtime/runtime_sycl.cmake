# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Runtime Arcane pour SYCL

message(STATUS "Adding Arccore Runtime for Sycl")

set(ARCCORE_SOURCES
  SyclAcceleratorRuntime.cc
)

arccore_add_library(arccore_accelerator_sycl_runtime
  INPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/runtime
  RELATIVE_PATH .
  FILES ${ARCCORE_SOURCES}
)

target_compile_options(arccore_accelerator_sycl_runtime PRIVATE
  ${ARCCORE_CXX_SYCL_FLAGS}
)

target_link_libraries(arccore_accelerator_sycl_runtime PRIVATE
  arccore_common
  arccore_accelerator_sycl
  arccore_sycl_compile_flags
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
