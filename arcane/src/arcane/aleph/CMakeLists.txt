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
    add_subdirectory(trilinos)
    add_subdirectory(petsc)
    #  add_subdirectory(cuda)
  endif()
  add_subdirectory(tests)
endif()
