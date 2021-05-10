set(PKGS Glib TBB)
arcane_find_package(TBB)

include(srcs.cmake)
if(TBB_FOUND)
  # Il faut une version récente de TBB (2018+).
  # Pour cela, on regarde si le fichier 'tbb_stddef.h' existe car ce dernier a été
  # ajouté à partir de la version 2018. De plus, il contient des macros permettant de
  # savoir quelle version de TBB on utilise.
  find_path(_TBB_STDDEF_HEADER NAMES tbb/tbb_stddef.h PATHS ${TBB_INCLUDE_DIRS})
  if (_TBB_STDDEF_HEADER)
    message(STATUS "[arcane_thread] Adding TBB implementation")
    set(ARCANE_SOURCES ${ARCANE_SOURCES} TBBThreadImplementation.cc TBBTaskImplementation.cc)
  else()
    message(STATUS "[arcane_thread] Remove TBB support because your version of TBB is too old (doesn't have 'tbb_stddef.h' header)")
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
