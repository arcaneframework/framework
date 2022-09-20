include(srcs.cmake)
set(PKGS vtkIOXML)
arcane_find_package(vtkIOXML)

if (vtkIOXML_FOUND) # this dependency to vtk is temporary, will be in Neo then
  list(APPEND ARCANE_SOURCES
        PolyhedralMeshTools.cc
      )
endif()

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
if (Neo_FOUND)
  target_link_libraries(arcane_mesh PRIVATE Neo::Neo)
  target_compile_definitions(arcane_mesh PRIVATE ARCANE_HAS_CUSTOM_MESH_TOOLS)
endif()

if (vtkIOXML_FOUND) # this dependency to vtk is temporary, will be in Neo then
  target_compile_definitions(arcane_mesh PRIVATE ARCANE_HAS_VTKIO)
  #arcane_add_arccon_packages(arcane_mesh PRIVATE VTK::IOXML)
  #target_link_libraries(arcane_mesh PRIVATE VTK::IOLegacy)
endif()

arcane_register_library(arcane_mesh)
