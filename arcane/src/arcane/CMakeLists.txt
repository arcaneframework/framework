find_package(LibXml2 REQUIRED)

include(srcs.cmake)

arcane_add_library(arcane_core
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane
  FILES ${ARCANE_SOURCES}
)

arcane_add_arccon_packages(arcane_core PRIVATE LibXml2)

target_compile_definitions(arcane_core PRIVATE ARCANE_COMPONENT_arcane_core)

target_link_libraries(arcane_core PUBLIC arcane_accelerator_core arcane_utils)
target_link_libraries(arcane_core PUBLIC Arccore::arccore_message_passing Arccore::arccore_serialize)

arcane_register_library(arcane_core)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
