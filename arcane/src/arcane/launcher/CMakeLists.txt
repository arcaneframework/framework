include(srcs.cmake)

arcane_add_library(arcane_launcher
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/launcher
  FILES ${ARCANE_SOURCES}
  )

target_compile_definitions(arcane_launcher PRIVATE ARCANE_COMPONENT_arcane_launcher)

target_link_libraries(arcane_launcher PUBLIC arcane_impl arcane_core)

arcane_register_library(arcane_launcher)
