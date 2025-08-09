# VTK peut se trouver soit directement, soit via Paraview.
# Pour Arcane, le package 'vtkIOXML'  sert uniquement pour la gestion des fichiers 'vtu'

# VTK/Paraview utilise les fichiers *Config.cmake et définissent des cibles
# importées pour vtkIOXML et les autres packages. Cependant, on ne peut pas utiliser
# directement ces cibles tant qu'on génère le fichier 'pkglist.xml' car sinon
# ce dernier contiendra des noms de cible VTK.
#
# On fabrique donc directement le package 'arccon::vtkIOXML' et se basant
# sur le fait que vtk fournit pour chaque package 'x' une variable 'x_DEPENDS' qui
# contient les dépendances.
#
# Ce finder et utilisé pour des versions de vtk >= 9. Auparavant le package s'appelait vtkIOXML

arccon_return_if_package_found(IOXML)

find_package(ParaView QUIET)
find_package(VTK QUIET COMPONENTS IOXML)

if (VTK_FOUND)
  message(STATUS "VTK (for IOXML) version ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
endif()

# Pour la définition de 'arcane_vtkutils_add_depend_lib_to_list'
include(${CMAKE_CURRENT_LIST_DIR}/VtkUtils.cmake)

if(TARGET VTK::IOXML)
  set(IOXML_FOUND TRUE)
  get_target_property(IOXML_INCLUDE_DIRS VTK::IOXML INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "IOXML_INCLUDE_DIRS = ${IOXML_INCLUDE_DIRS}")
  # On supprime la valeur de vtkIOXML_LIBRARIES car les bibliothèques dépendantes
  # seront positionnées via add_depend_lib_to_list
  arcane_vtkutils_add_depend_lib_to_list_v2(VTK::IOXML)
  message(STATUS "IOXML LIBRARIES=${_ALLLIBS}")
  set(IOXML_LIBRARIES "${_ALLLIBS}")
  arccon_register_package_library(IOXML IOXML)
  # Pour compatibilité avec l'existant (septembre 2022)
  add_library(arcane::IOXML ALIAS arcconpkg_IOXML)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
