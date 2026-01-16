# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(arcane_accelerator_cuda_testlib Test.cu Test2.cc TestCpp.cc)
arcane_accelerator_add_source_files(Test2.cc)
set_target_properties(arcane_accelerator_cuda_testlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(arcane_accelerator_cuda_testlib PUBLIC arcane_accelerator arcane_core arccore_cuda_compile_flags)

add_executable(arcane_accelerator_cuda_test TestMain.cc)
target_link_libraries(arcane_accelerator_cuda_test PUBLIC arcane_accelerator_cuda_testlib arccore_accelerator_cuda_runtime)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
