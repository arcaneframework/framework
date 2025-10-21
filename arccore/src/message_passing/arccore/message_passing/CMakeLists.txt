include(srcs.cmake)

arccore_add_component_library(message_passing
  FILES ${SOURCES}
)

target_link_libraries(arccore_message_passing PUBLIC arccore_serialize arccore_collections arccore_trace arccore_base)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
