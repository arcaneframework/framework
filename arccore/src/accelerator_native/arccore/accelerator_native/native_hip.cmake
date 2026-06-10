# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Arccore Backend for ROCM/HIP

set(ARCCORE_SOURCES
  HipAccelerator.cc
  HipAccelerator.h
)

find_package(Hip REQUIRED)

# Create an interface target to propagate the compilation options
# common for HIP compilation

add_library(arccore_hip_compile_flags INTERFACE)
# Normally there shouldn't be a need to add this line, but if we do
# not with 'ROCM 4.3', then it doesn't find 'libclang_rt.builtins-x86_64.a'
target_link_directories(arccore_hip_compile_flags INTERFACE ${HIP_CLANG_INCLUDE_PATH}/lib/linux)

target_compile_options(arccore_hip_compile_flags INTERFACE
  # No specific option for now
)
install(TARGETS arccore_hip_compile_flags EXPORT ArccoreTargets)

arccore_add_component_library(accelerator_native
  LIB_NAME arccore_accelerator_hip
  FILES ${ARCCORE_SOURCES}
)

target_link_libraries(arccore_accelerator_hip PUBLIC
  arccore_base
  arccore_hip_compile_flags
  hip::host
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
