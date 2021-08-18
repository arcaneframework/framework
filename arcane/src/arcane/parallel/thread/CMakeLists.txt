set(PKGS Glib TBB)
arcane_find_package(TBB)

include(srcs.cmake)

set(ARCANE_TBB_SOURCES ${ARCANE_SOURCES} TBBThreadImplementation.cc TBBTaskImplementation.cc)
set(ARCANE_ONETBB_SOURCES ${ARCANE_SOURCES} TBBThreadImplementation.cc OneTBBTaskImplementation.cc)

if(TBB_VERSION)
  # Si TBB_VERSION est définie, on est sur que TBB est présent.
  # Normalement cette variable est définie si on utilise une version de TBB
  # qui possède un fichier de configuration CMake
  message(STATUS "[arcane_thread] TBB_VERSION=${TBB_VERSION}")
  if (TBB_VERSION VERSION_GREATER_EQUAL "2021")
    message(STATUS "[arcane_thread] Using OneTBB 2021+ implementation")
    list(APPEND ARCANE_SOURCES ${ARCANE_ONETBB_SOURCES})
  else()
    message(STATUS "[arcane_thread] Using Legacy (2018-2020) TBB implementation")
    list(APPEND ARCANE_SOURCES ${ARCANE_TBB_SOURCES})
  endif()
elseif(TBB_FOUND)
  # Il faut une version récente de TBB (2018+).
  # Pour cela, on regarde si le fichier 'tbb_stddef.h' existe car ce dernier a été
  # ajouté à partir de la version 2018. De plus, il contient des macros permettant de
  # savoir quelle version de TBB on utilise.
  find_path(_TBB_STDDEF_HEADER NAMES tbb/tbb_stddef.h PATHS ${TBB_INCLUDE_DIRS})
  if (_TBB_STDDEF_HEADER)
    message(STATUS "[arcane_thread] Adding TBB implementation")
    list(APPEND ARCANE_SOURCES ${ARCANE_TBB_SOURCES})
  else()
    message(FATAL_ERROR
      "[arcane_thread] Your version of TBB is too old (need version 2018 or superior)"
      "To disable TBB, pass the flag -DCMAKE_DISABLE_FIND_PACKAGE_TBB=TRUE to 'cmake' command"
      )
  endif()
endif()

arcane_add_library(arcane_thread
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/parallel/thread
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_thread PRIVATE ARCANE_COMPONENT_arcane_thread)

arcane_add_arccon_packages(arcane_thread PRIVATE ${PKGS})

target_link_libraries(arcane_thread PUBLIC arcane_impl arcane_core arcane_utils)

arcane_register_library(arcane_thread)
