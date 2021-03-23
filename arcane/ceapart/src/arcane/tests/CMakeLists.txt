include(srcs.cmake)

arcane_add_library(arcane_cea_tests
  INPUT_PATH ${ARCANECEA_SRC_PATH}
  RELATIVE_PATH arcane/tests
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_cea_tests PRIVATE ARCANE_COMPONENT_arcane_cea_tests)

arcane_add_axl_files(arcane_cea_tests
  INPUT_PATH ${ARCANECEA_SRC_PATH}
  RELATIVE_PATH arcane/tests
  FILES ${AXL_FILES}
)

target_link_libraries(arcane_cea_tests PUBLIC arcane_materials arcane_cea_geometric arcane_cea ${ARCANE_BASE_LIBRARIES})
if(TARGET arcane_aleph)
  target_link_libraries(arcane_cea_tests PUBLIC arcane_aleph)
endif()
if(TARGET arcane_aleph_sloop)
  target_link_libraries(arcane_cea_tests PUBLIC arcane_aleph_sloop)
endif()

# Indique que la bibliothèque des tests a besoin de celle-ci
target_link_libraries(arcane_tests_lib PUBLIC arcane_cea_tests)

# Nécessaire pour les tests Simd sur les matériaux
if (ARCANE_USE_CXX14)
  target_compile_features(arcane_cea_tests PRIVATE cxx_generic_lambdas)
endif()

arcane_register_library(arcane_cea_tests OPTIONAL)

include(TestDotNet.cmake)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
