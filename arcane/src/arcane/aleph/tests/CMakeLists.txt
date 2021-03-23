include(srcs.cmake)

arcane_add_library(arcane_aleph_tests
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph/tests
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_aleph_tests PRIVATE ARCANE_COMPONENT_arcane_aleph_tests)

arcane_add_axl_files(arcane_aleph_tests
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph/tests
  FILES ${AXL_FILES}
)

if(TARGET arcane_aleph_hypre)
  target_link_libraries(arcane_aleph_tests PUBLIC arcane_aleph_hypre)
endif()
if(TARGET arcane_aleph_trilinos)
  target_link_libraries(arcane_aleph_tests PUBLIC arcane_aleph_trilinos)
endif()
if(TARGET arcane_aleph_petsc)
  target_link_libraries(arcane_aleph_tests PUBLIC arcane_aleph_petsc)
endif()

target_link_libraries(arcane_aleph_tests PUBLIC arcane_aleph_kappa arcane_aleph ${ARCANE_BASE_LIBRARIES})
