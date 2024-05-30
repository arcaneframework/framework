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

find_package(Neo QUIET)
if(Neo_FOUND)
  target_link_libraries(arcane_mesh PRIVATE Neo::Neo)
  target_compile_definitions(arcane_mesh PRIVATE ARCANE_HAS_POLYHEDRAL_MESH_TOOLS)
endif()

if(vtkIOXML_FOUND) # this dependency to vtk is temporary, will be in Neo then
  target_compile_definitions(arcane_mesh PRIVATE ARCANE_HAS_VTKIO)
  arcane_add_arccon_packages(arcane_mesh PRIVATE vtkIOXML)
endif()

if(REMOVE_UID_ON_DETACH)
  target_compile_definitions(arcane_mesh PRIVATE REMOVE_UID_ON_DETACH)
endif()

option(USE_GRAPH_CONNECTIVITY_POLICY "Option to activate graph connectivity")
if (USE_GRAPH_CONNECTIVITY_POLICY)
  target_compile_definitions(arcane_mesh PRIVATE USE_GRAPH_CONNECTIVITY_POLICY)
endif ()
arcane_register_library(arcane_mesh)
