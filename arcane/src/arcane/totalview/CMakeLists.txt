# Totalview n'est pas disponible sous Windows
if (WIN32)
  return()
endif()
include(srcs.cmake)

arcane_add_library(arcane_totalview
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/totalview
  FILES ${ARCANE_SOURCES}
)

target_compile_definitions(arcane_totalview PRIVATE ARCANE_COMPONENT_arcane_totalview)
target_link_libraries(arcane_totalview PUBLIC arcane_core)

arcane_register_library(arcane_totalview OPTIONAL)
