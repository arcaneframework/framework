# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

add_library(arcane_accelerator_hip_testlib Test.cu.cc TestCpp.cc)
set_source_files_properties(Test.cu.cc PROPERTIES LANGUAGE HIP)
target_link_libraries(arcane_accelerator_hip_testlib PUBLIC arcane_accelerator arcane_core arccore_hip_compile_flags)

add_executable(arcane_accelerator_hip_test TestMain.cc)
target_link_libraries(arcane_accelerator_hip_test PUBLIC arcane_accelerator_hip_testlib arccore_accelerator_hip_runtime)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
