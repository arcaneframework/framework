set(PKGS vtkIOXML vtkIOXdmf2)
arcane_find_package(vtkIOXML)
arcane_find_package(vtkIOXdmf2)

include(srcs.cmake)

if(TARGET arccon::vtkIOXdmf2)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} XmfMeshWriter.cc XmfMeshReader.cc)
endif()
if(TARGET arccon::vtkIOXML)
  set( ARCANE_SOURCES ${ARCANE_SOURCES} VtuMeshReader.cc VtuMeshWriter.cc)
endif()

arcane_add_library(arcane_ios
  INPUT_PATH ${Arcane_SOURCE_DIR}/src
  RELATIVE_PATH arcane/ios
  FILES ${ARCANE_SOURCES}
)

arcane_add_arccon_packages(arcane_ios PRIVATE ${PKGS})

target_compile_definitions(arcane_ios PRIVATE ARCANE_COMPONENT_arcane_ios)

target_link_libraries(arcane_ios PUBLIC arcane_core)

arcane_register_library(arcane_ios OPTIONAL)
