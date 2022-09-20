﻿# VTK peut se trouver soit directement, soit via Paraview.
# Pour Arcane, le package 'vtkIOXML'  sert uniquement pour la gestion des fichiers 'vtu'

# VTK/Paraview utilise les fichiers *Config.cmake et définissent des cibles
# importées pour vtkIOXML et les autres packages. Cependant, on ne peut pas utiliser
# directement ces cibles tant qu'on génère le fichier 'pkglist.xml' car sinon
# ce dernier contiendra des noms de cible VTK.
#
# On fabrique donc directement le package 'arccon::vtkIOXML' et se basant
# sur le fait que vtk fournit pour chaque package 'x' une variable 'x_DEPENDS' qui
# contient les dépendances.

arccon_return_if_package_found(vtkIOXML)

find_package(ParaView QUIET)

find_package(VTK COMPONENTS IOXML)
if (VTK_FOUND)
  message(STATUS "VTK (for IOXML) version ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
endif()
if(TARGET VTK::IOXML)
  set(vtkIOXML_FOUND TRUE)
  message(STATUS "FOUND TARGET VTK::IOXML")
#  arccon_register_cmake_config_target(vtkIOXML CONFIG_TARGET_NAME "VTK::IOXML;VTK::IOLegacy" PACKAGE_NAME vktIOXML)
  arccon_register_cmake_multiple_config_target(vtkIOXML CONFIG_TARGET_NAMES VTK::IOXML VTK::IOLegacy)
  return()
endif()

find_package(VTK COMPONENTS vtkIOXML)

if (VTK_FOUND)
  message(STATUS "VTK (for vtkIOXML) version ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
endif()

# Pour la définition de 'arcane_vtkutils_add_depend_lib_to_list'
include(${CMAKE_CURRENT_LIST_DIR}/VtkUtils.cmake)

if(TARGET vtkIOXML)
  set(vtkIOXML_FOUND TRUE)
  message(STATUS "vtkIOXML_INCLUDE_DIRS = ${vtkIOXML_INCLUDE_DIRS}")
  # On supprime la valeur de vtkIOXML_LIBRARIES car les bibliothèques dépendantes
  # seront positionnées via add_depend_lib_to_list
  arcane_vtkutils_add_depend_lib_to_list(vtkIOXML)
  message(STATUS "vtkIOXML LIBRARIES=${_ALLLIBS}")
  set(vtkIOXML_LIBRARIES "${_ALLLIBS}")
  arccon_register_package_library(vtkIOXML vtkIOXML)
  # Pour compatibilité avec l'existant (septembre 2022)
  add_library(arcane::vtkIOXML ALIAS arcconpkg_vtkIOXML)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
