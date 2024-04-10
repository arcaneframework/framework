if(ARCCORE_ENABLE_GLIB)
  find_package(Glib REQUIRED)
endif()

set(SOURCES
  ConcurrencyGlobal.h
  ConcurrencyGlobal.cc
  IThreadImplementation.h
  IThreadBarrier.h
  Mutex.h
  Mutex.cc
  NullThreadImplementation.h
  SpinLock.h
  SpinLock.cc
)

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
  target_compile_definitions(arccore_base PRIVATE ARCCORE_HAS_GLIB)
endif()

target_link_libraries(arccore_concurrency PUBLIC arccore_base)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
