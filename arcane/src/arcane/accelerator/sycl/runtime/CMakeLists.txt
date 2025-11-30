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

target_compile_options(arcane_accelerator_sycl_runtime PRIVATE ${ARCANE_CXX_SYCL_FLAGS})

target_link_libraries(arcane_accelerator_sycl_runtime PUBLIC
  Arccore::arccore_accelerator_sycl
  arccore_sycl_compile_flags
  )

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(arcane_accelerator_sycl_testlib Test.sycl.cc TestCpp.cc)
arcane_accelerator_add_source_files(Test.sycl.cc)
target_link_libraries(arcane_accelerator_sycl_testlib PUBLIC arcane_accelerator arccore_sycl_compile_flags)
target_link_libraries(arcane_accelerator_sycl_testlib PUBLIC ${ARCANE_CXX_SYCL_FLAGS})
add_executable(arcane_accelerator_sycl_test TestMain.cc)
target_link_libraries(arcane_accelerator_sycl_test PUBLIC arcane_accelerator_sycl_testlib arcane_accelerator_sycl_runtime)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

arcane_register_library(arcane_accelerator_sycl_runtime OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
