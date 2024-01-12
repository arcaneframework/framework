find_package(Glib REQUIRED)

set(SOURCES
  ConcurrencyGlobal.h
  ConcurrencyGlobal.cc
  IThreadImplementation.h
  IThreadBarrier.h
  GlibThreadImplementation.cc
  GlibThreadImplementation.h
  GlibAdapter.h
  GlibAdapter.cc
  Mutex.h
  Mutex.cc
  NullThreadImplementation.h
  SpinLock.h
  SpinLock.cc
  ThreadPrivate.h
  ThreadPrivate.cc
)

arccore_add_component_library(concurrency
  FILES ${SOURCES}
)

target_link_libraries(arccore_concurrency PRIVATE arcconpkg_Glib)
target_link_libraries(arccore_concurrency PUBLIC arccore_base)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
