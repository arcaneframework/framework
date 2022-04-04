include(srcs.cmake)

arcane_add_library(arcane_accelerator_core
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/accelerator/core
  FILES ${ARCANE_SOURCES}
  )

target_link_libraries(arcane_accelerator_core PUBLIC
  arcane_utils
  )

arcane_register_library(arcane_accelerator_core)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
