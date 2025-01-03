# VTK peut se trouver soit directement, soit via Paraview.
# Pour Arcane, vtkIOLegacy est utilisé pour la lecture des maillages polyédriques 
# Voir fichier 'FindvtkIOXML' pour plus d'infos
arccon_return_if_package_found(vtkIOLegacy)

find_package(ParaView QUIET)
find_package(VTK QUIET COMPONENTS vtkIOLegacy)

if (VTK_FOUND)
  message(STATUS "VTK (for vtkIOLegacy) version ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
endif()

if(TARGET vtkIOLegacy)
  set(vtkIOLegacy_FOUND TRUE)
  message(STATUS "vtkIOLegacy_INCLUDE_DIRS = ${vtkIOLegacy_INCLUDE_DIRS}")
  arcane_vtkutils_add_depend_lib_to_list(vtkIOLegacy)
  message(STATUS "vtkIOLegacy LIBS=${_ALLLIBS}")
  set(vtkIOLegacy_LIBRARIES "${_ALLLIBS}")
  arccon_register_package_library(vtkIOLegacy vtkIOLegacy)
  # Pour compatibilité avec l'existant (septembre 2022)
  add_library(arcane::vtkIOLegacy ALIAS arcconpkg_vtkIOLegacy)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End: