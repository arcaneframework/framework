message(STATUS "Parsing ${CMAKE_CURRENT_SOURCE_DIR}")
if(NOT TARGET arcane_hyoda)
  return()
endif()

include(srcs.cmake)

arcane_add_library(arcane_hyoda_plugins
  INPUT_PATH ${ARCANECEA_SRC_PATH}
  RELATIVE_PATH arcane/hyoda
  FILES ${ARCANE_SOURCES}
)

target_include_directories(arcane_hyoda_plugins PUBLIC $<BUILD_INTERFACE:${ARCANECEA_SRC_PATH}> $<INSTALL_INTERFACE:include>)
target_link_libraries(arcane_hyoda_plugins arcane_core)

arcane_register_library(arcane_hyoda_plugins OPTIONAL)
