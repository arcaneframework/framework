# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Runtime Arcane pour HIP

set(ARCANE_SOURCES
  SyclAcceleratorRuntime.cc
  )

arcane_add_library(arcane_accelerator_sycl_runtime
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator/sycl/runtime
  FILES ${ARCANE_SOURCES}
  )

target_compile_options(arcane_accelerator_sycl_runtime PRIVATE "-fsycl")

target_link_libraries(arcane_accelerator_sycl_runtime PUBLIC
  arcane_accelerator arcane_sycl_compile_flags
  )

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_executable(arcane_accelerator_sycl_test TestMain.cc)
target_link_libraries(arcane_accelerator_sycl_test PUBLIC arcane_accelerator_sycl_runtime)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

arcane_register_library(arcane_accelerator_sycl_runtime OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
