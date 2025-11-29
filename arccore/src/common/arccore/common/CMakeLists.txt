include(srcs.cmake)

arccore_add_component_library(common
  FILES ${SOURCES}
)

target_link_libraries(arccore_common PUBLIC arccore_trace arccore_base)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
