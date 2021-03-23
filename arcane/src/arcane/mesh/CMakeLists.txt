include(srcs.cmake)

arcane_add_library(arcane_mesh
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/mesh
  FILES ${ARCANE_SOURCES}
  AXL_FILES ${AXL_FILES}
  )

target_compile_definitions(arcane_mesh PRIVATE ARCANE_COMPONENT_arcane_mesh)

arcane_add_arccon_packages(arcane_mesh PRIVATE ${PKGS})

target_link_libraries(arcane_mesh PUBLIC arcane_core)

arcane_register_library(arcane_mesh)
