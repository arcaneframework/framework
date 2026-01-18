if (ARCCORE_ENABLE_GLIB)
  find_package(Glib REQUIRED)
endif()
if (ARCCORE_ENABLE_TBB)
  find_package(TBB)
  if(TBB_VERSION)
    # Si TBB_VERSION est définie, on est sur que TBB est présent.
    # Normalement cette variable est définie si on utilise une version de TBB
    # qui possède un fichier de configuration CMake
    message(STATUS "[arccore_concurrency] TBB_VERSION=${TBB_VERSION}")
    if (TBB_VERSION VERSION_GREATER_EQUAL "2021")
      message(STATUS "[arcane_thread] Using OneTBB 2021+ implementation")
      set(ARCANE_HAS_ONETBB TRUE)
    else()
      message(FATAL_ERROR "[arcane_thread] Your version '${TBB_VERSION} of TBB is too old. Version 2020+ is required."
        " Install a recent version of TBB or disable TBB support using -DARCCORE_ENABLE_TBB=FALSE when running cmake command")
    endif()
  elseif(TBB_FOUND)
    message(FATAL_ERROR
      "[arccore_concurrency] Can not find TBB_VERSION. This is probably because your TBB version is too old."
      "Version 2021+ is required."
      "Install a recent version of TBB or disable TBB support using -DARCCORE_ENABLE_TBB=FALSE when running cmake command"
    )
  else()
    message(FATAL_ERROR
      "[arccore_concurrency] Can not find OneTBB."
      "Install a recent version of TBB or disable TBB support using -DARCCORE_ENABLE_TBB=FALSE when running cmake command"
    )
  endif()
endif()

include(srcs.cmake)

if (ARCCORE_ENABLE_GLIB)
  list(APPEND SOURCES
    GlibThreadImplementation.cc
    GlibThreadImplementation.h
    GlibAdapter.h
    GlibAdapter.cc
    ThreadPrivate.h
    ThreadPrivate.cc
  )
endif()
if (ARCCORE_ENABLE_TBB)
  list(APPEND SOURCES
    TBBTaskImplementation.cc
  )
endif()

arccore_add_component_library(concurrency
  FILES ${SOURCES}
)

if (ARCCORE_ENABLE_GLIB)
  target_link_libraries(arccore_concurrency PRIVATE arcconpkg_Glib)
  target_compile_definitions(arccore_concurrency PRIVATE ARCCORE_HAS_GLIB)
endif()
if (ARCCORE_ENABLE_TBB)
  target_link_libraries(arccore_concurrency PRIVATE arccon::TBB)
endif()

target_link_libraries(arccore_concurrency PUBLIC arccore_trace arccore_base)

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
