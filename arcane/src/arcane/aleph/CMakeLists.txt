include(srcs.cmake)

arcane_add_library(arcane_aleph
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph
  FILES ${ARCANE_SOURCES}
  )

target_compile_definitions(arcane_aleph PRIVATE ARCANE_COMPONENT_arcane_aleph)

target_link_libraries(arcane_aleph PUBLIC arcane_core)
arcane_register_library(arcane_aleph OPTIONAL)

if(MPI_FOUND)
  add_subdirectory(kappa)
  add_subdirectory(hypre)
  if(NOT WIN32)
    if(NOT DISABLE_TRILINOS_ALEPH)
      add_subdirectory(trilinos)
    endif()
    add_subdirectory(petsc)
    #  add_subdirectory(cuda)
  endif()
  add_subdirectory(tests)
endif()

# A supprimer mais en attendant permet de compiler Sloop avec IntelMPI
if (ARCANE_WANT_MPI_DISTRIB MATCHES "intelmpi")
  set(DISABLE_SLOOP_AND_TRILINOS YES)
  include(cmake/FindItac.cmake)
endif ()
if(ARCANE_USE_MPC)
  set(DISABLE_SLOOP_AND_TRILINOS YES)
endif()

if(NOT WIN32)
  if(NOT DISABLE_SLOOP_AND_TRILINOS)
    include(${CMAKE_CURRENT_LIST_DIR}/FindSuperLU.cmake)
    add_subdirectory(sloop)
  endif()
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
