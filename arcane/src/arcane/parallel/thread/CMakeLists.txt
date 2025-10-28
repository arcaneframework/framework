set(PKGS Glib TBB)
arcane_find_package(TBB)

include(srcs.cmake)

set(ARCANE_TBB_SOURCES TBBThreadImplementation.cc TBBTaskImplementation.cc)
set(ARCANE_HAS_ONETBB FALSE)
set(ARCANE_HAS_TBBIMPL FALSE)
if(TBB_VERSION)
  # Si TBB_VERSION est définie, on est sur que TBB est présent.
  # Normalement cette variable est définie si on utilise une version de TBB
  # qui possède un fichier de configuration CMake
  message(STATUS "[arcane_thread] TBB_VERSION=${TBB_VERSION}")
  set(ARCANE_HAS_TBBIMPL TRUE)
  if (TBB_VERSION VERSION_GREATER_EQUAL "2021")
    message(STATUS "[arcane_thread] Using OneTBB 2021+ implementation")
    set(ARCANE_HAS_ONETBB TRUE)
  else()
    message(FATAL_ERROR "[arcane_thread] Your version '${TBB_VERSION} is too old. Version 2021+ is required."
    "Install a recent version of TBB or disable TBB support using -DCMAKE_DISABLE_FIND_PACKAGE_TBB=TRUE when running cmake command")
  endif()
elseif(TBB_FOUND)
  message(FATAL_ERROR
    "[arcane_thread] Can not find TBB_VERSION. This is probably because your TBB version is too old."
    "Version 2021+ is required."
    "Install a recent version of TBB or disable TBB support using -DCMAKE_DISABLE_FIND_PACKAGE_TBB=TRUE when running cmake command"
  )
endif()

# Si les TBB sont disponibles, indique qu'on a le support des tâches
if (ARCANE_HAS_TBBIMPL)
  list(APPEND ARCANE_SOURCES ${ARCANE_TBB_SOURCES})
  set(ARCANE_HAS_TASKS TRUE CACHE STRING "Support for tasks" FORCE)
endif()

arcane_add_library(arcane_thread
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/parallel/thread
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_thread PRIVATE ARCANE_COMPONENT_arcane_thread)
if (ARCANE_HAS_ONETBB)
  target_compile_definitions(arcane_thread PRIVATE ARCANE_USE_ONETBB)
endif()

arcane_add_arccon_packages(arcane_thread PRIVATE ${PKGS})

target_link_libraries(arcane_thread PUBLIC arcane_impl arcane_core arcane_utils)

arcane_register_library(arcane_thread)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
