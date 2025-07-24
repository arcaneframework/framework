# VTK peut se trouver soit directement, soit via Paraview.
# Pour Arcane, IOLegacy est utilisé pour la lecture des maillages polyédriques
# Voir fichier 'FindIOXML' pour plus d'infos
# Ce finder et utilisé pour des versions de vtk >= 9. Auparavant le package s'appelait vtkIOLegacy
arccon_return_if_package_found(IOLegacy)

find_package(ParaView QUIET)
find_package(VTK QUIET COMPONENTS IOLegacy)

if (VTK_FOUND)
  message(STATUS "VTK (for IOLegacy) version ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
endif()

if(TARGET VTK::IOLegacy)
  set(IOLegacy_FOUND TRUE)
  get_target_property(IOLegacy_INCLUDE_DIRS VTK::IOLegacy INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "IOLegacy_INCLUDE_DIRS = ${IOLegacy_INCLUDE_DIRS}")
  arcane_vtkutils_add_depend_lib_to_list_v2(VTK::IOLegacy)
  message(STATUS "IOLegacy LIBS=${_ALLLIBS}")
  set(IOLegacy_LIBRARIES "${_ALLLIBS}")
  arccon_register_package_library(IOLegacy IOLegacy)
  # Pour compatibilité avec l'existant (septembre 2022)
  add_library(arcane::IOLegacy ALIAS arcconpkg_IOLegacy)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End: