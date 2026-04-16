message(STATUS "Parsing ${CMAKE_CURRENT_SOURCE_DIR}")

include(srcs.cmake)

#arcane_add_library(arcane_cea_geometric
#  INPUT_PATH ${ARCANECEA_SRC_PATH}
#  RELATIVE_PATH arcane/geometric
#  FILES ${ARCANE_SOURCES}
#  )

foreach (isource ${ARCANE_SOURCES})
  set(_FILE ${ARCANECEA_SRC_PATH}/arcane/geometric/${isource})
  string(REGEX MATCH "\.(h|H)$" _OUT_HEADER ${isource})
  if (_OUT_HEADER)
    get_filename_component(_HEADER_RELATIVE_PATH ${isource} DIRECTORY)
    install(FILES ${_FILE} DESTINATION include/arcane/geometric/${_HEADER_RELATIVE_PATH})
  endif ()
endforeach ()

#target_compile_definitions(arcane_cea_geometric PRIVATE ARCANE_COMPONENT_arcane_cea_geometric)
#
#target_include_directories(arcane_cea_geometric PUBLIC $<BUILD_INTERFACE:${ARCANECEA_SRC_PATH}> $<INSTALL_INTERFACE:include>)
#target_link_libraries(arcane_cea_geometric PUBLIC arcane_geometry arcane_core)
#
#arcane_register_library(arcane_cea_geometric)
