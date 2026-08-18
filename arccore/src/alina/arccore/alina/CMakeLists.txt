include (srcs.cmake)

arccore_add_component_library(alina
  FILES ${SOURCES}
)

target_compile_definitions(arccore_alina PRIVATE ARCANE_COMPONENT_arccore_alina)

target_link_libraries(arccore_alina PUBLIC Arcane::arcane_utils arccore_accelerator)
arcane_register_library(arccore_alina OPTIONAL)

if (ARCCORE_CXX_COMPILER_IS_GNU_OR_CLANG_BASED)
  target_compile_options(arccore_alina PUBLIC -Wno-conversion -Wno-float-conversion)
endif()

#----------------------------------------------------------------------------
# Add Arccore MPI
# In the future, it will not be needed. Only 'arccore_message_passing' will be needed
target_link_libraries(arccore_alina PUBLIC Arccore::arccore_message_passing_mpi)

#----------------------------------------------------------------------------
# Find CUDA
#----------------------------------------------------------------------------
if (ARCCORE_HAS_CUDA)
  find_package(CUDAToolkit REQUIRED)
  if (OPENMP_FOUND)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
  endif()

  add_library(cuda_target INTERFACE)
  target_link_libraries(cuda_target INTERFACE ${CUDA_cusparse_LIBRARY})
endif()

#----------------------------------------------------------------------------
# Find Metis/Parmetis
#----------------------------------------------------------------------------
find_package(Metis QUIET)

arcane_find_package(Parmetis)
if(Parmetis_FOUND)
  message(STATUS "Found Parmetis for Alina")
  target_compile_definitions(arccore_alina PUBLIC ARCCORE_ALINA_HAVE_PARMETIS ARCCORE_ALINA_HAVE_PARMETIS)
  target_link_libraries(arccore_alina PUBLIC arccon::Parmetis)
endif()

#----------------------------------------------------------------------------
# Check compilation of tests and samples
#----------------------------------------------------------------------------

option(ARCCORE_ALINA_BUILD_TESTS "Build Alina tests" ON)
option(ARCCORE_ALINA_BUILD_SAMPLES "Build Alina samples" ON)

if (ARCCORE_ALINA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if (ARCCORE_ALINA_BUILD_SAMPLES)
  add_subdirectory(samples)
  add_subdirectory(tutorial)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
