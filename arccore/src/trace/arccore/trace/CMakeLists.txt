include(srcs.cmake)

arccore_add_component_library(trace
  FILES ${SOURCES}
)

target_link_libraries(arccore_trace PUBLIC arccore_base)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
