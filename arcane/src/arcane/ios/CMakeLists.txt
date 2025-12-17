arcane_find_package(VTKVersionHelper) # Pour obtenir VTK_PACKAGE_PREFIX qui varie en fonction des versions de VTK
set(PKGS ${VTK_PACKAGE_PREFIX}IOXML ${VTK_PACKAGE_PREFIX}IOXdmf2)
arcane_find_package(${VTK_PACKAGE_PREFIX}IOXML)
arcane_find_package(${VTK_PACKAGE_PREFIX}IOXdmf2)

include(srcs.cmake)

if(${VTK_PACKAGE_PREFIX}IOXdmf2_FOUND)
  if (${VTK_VERSION} VERSION_LESS_EQUAL "8.90.0")
    list(APPEND ARCANE_SOURCES
      XmfMeshWriter.cc
      XmfMeshReader.cc
    )
    set (VTK_HAS_XDMF TRUE PARENT_SCOPE)
  endif ()
endif()

if(${VTK_PACKAGE_PREFIX}IOXML_FOUND)
  list(APPEND ARCANE_SOURCES
    VtuMeshReader.cc
    VtuMeshWriter.cc
    )
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
