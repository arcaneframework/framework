include(srcs.cmake)

arcane_add_library(arcane_aleph_kappa
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/aleph/kappa
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_aleph_kappa PRIVATE ARCANE_COMPONENT_arcane_aleph_kappa)
target_link_libraries(arcane_aleph_kappa PUBLIC ${ARCANE_BASE_LIBRARIES})
target_link_libraries(arcane_aleph_kappa PRIVATE arcane_aleph)
arcane_register_library(arcane_aleph_kappa OPTIONAL)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
