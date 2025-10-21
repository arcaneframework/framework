if(ARCCORE_ENABLE_GLIB)
  find_package(Glib REQUIRED)
endif()

include(srcs.cmake)

if(ARCCORE_ENABLE_GLIB)
  list(APPEND SOURCES
    GlibThreadImplementation.cc
    GlibThreadImplementation.h
    GlibAdapter.h
    GlibAdapter.cc
    ThreadPrivate.h
    ThreadPrivate.cc
  )
endif()

arccore_add_component_library(concurrency
  FILES ${SOURCES}
)

if(ARCCORE_ENABLE_GLIB)
  target_link_libraries(arccore_concurrency PRIVATE arcconpkg_Glib)
  target_compile_definitions(arccore_concurrency PRIVATE ARCCORE_HAS_GLIB)
endif()

target_link_libraries(arccore_concurrency PUBLIC arccore_base)

find_package(Threads)
if (TARGET Threads::Threads)
  target_link_libraries(arccore_concurrency PRIVATE Threads::Threads)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
