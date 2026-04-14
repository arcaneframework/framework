include(srcs.cmake)

if (ARCANE_HAS_ACCELERATOR_API)
  list(APPEND ARCANE_SOURCES
    ${ARCANE_MATERIAL_SOURCES}
  )
  arcane_accelerator_add_source_files(${ARCANE_ACCELERATOR_SOURCES})
endif()

arcane_add_library(arcane_cea_tests
  INPUT_PATH ${ARCANECEA_SRC_PATH}
  RELATIVE_PATH arcane/tests
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
)

target_compile_definitions(arcane_cea_tests PRIVATE ARCANE_COMPONENT_arcane_cea_tests)

if (TARGET arcane_materials)
  target_link_libraries(arcane_cea_tests PUBLIC arcane_materials)
endif()
target_link_libraries(arcane_cea_tests PUBLIC arcane_cea_geometric arcane_cea ${ARCANE_BASE_LIBRARIES})
if(TARGET arcane_aleph)
  target_link_libraries(arcane_cea_tests PUBLIC arcane_aleph)
endif()
if(TARGET arcane_aleph_sloop)
  target_link_libraries(arcane_cea_tests PUBLIC arcane_aleph_sloop)
endif()
target_link_libraries(arcane_cea_tests PRIVATE $<TARGET_NAME_IF_EXISTS:arcane_accelerator>)

# ----------------------------------------------------------------------------

# Indique que la bibliothèque des tests a besoin de celle-ci
target_link_libraries(arcane_tests_lib PUBLIC arcane_cea_tests)

arcane_register_library(arcane_cea_tests OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
