# VTK peut se trouver soit directement, soit via Paraview.
# Pour Arcane, vtkIOXdmf2 est utilisé pour la gestion des Xdmf
# Voir fichier 'FindvtkIOXML' pour plus d'infos
if (TARGET arccon::vtkIOXdmf2)
  return()
endif()

find_package(ParaView QUIET)
find_package(VTK QUIET COMPONENTS vtkIOXdmf2)

if (VTK_FOUND)
  message(STATUS "VTK (for vtkIOXdmf2) version ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
endif()

if(TARGET vtkIOXdmf2)
  set(vtkIOXdmf2_FOUND TRUE)
  message(STATUS "vtkIOXdmf2_INCLUDE_DIRS = ${vtkIOXdmf2_INCLUDE_DIRS}")
  set(vtkIOXdmf2_LIBRARIES)
  arcane_add_package_library(vtkIOXdmf2 vtkIOXdmf2)
  arcane_vtkutils_add_depend_lib_to_list(vtkIOXdmf2)
  message(STATUS "vtkIOXML LIBS=${_ALLLIBS}")
  target_link_libraries(arcanepkg_vtkIOXdmf2 INTERFACE ${_ALLLIBS})
endif()
