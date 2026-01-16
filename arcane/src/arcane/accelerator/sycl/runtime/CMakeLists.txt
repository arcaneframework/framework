# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(arcane_accelerator_sycl_testlib Test.sycl.cc TestCpp.cc)
arcane_accelerator_add_source_files(Test.sycl.cc)
target_link_libraries(arcane_accelerator_sycl_testlib PUBLIC arcane_accelerator arccore_sycl_compile_flags)
target_link_libraries(arcane_accelerator_sycl_testlib PUBLIC ${ARCCORE_CXX_SYCL_FLAGS})
add_executable(arcane_accelerator_sycl_test TestMain.cc)
target_link_libraries(arcane_accelerator_sycl_test PUBLIC arcane_accelerator_sycl_testlib arccore_accelerator_sycl_runtime)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
