#
# Cherche les composants de Trilinos nécessaires pour Aleph.
#
# Afin de ne pas interférer avec la package Trilinos, on nomme ce
# package 'TrilinosAleph'. Il n'est utilisé que pour Aleph
#
# À la date de novembre 2023, L'utilisation de CMake
# avec Trilinos est assez compliqué car Trilinos fournit
# des fichiers de configuration CMake pour chaque composant
# mais ces derniers ne sont pas facilement utilisables pour deux raisons:
# - le nom de la cible peut varier suivant l'installation.
#   Par exemple, pour un composant 'ifpack', le nom
#   de la cible peut être 'ifpack' (si compilé avec Spack)
#   ou 'trilinos_ifpack' (si package système ubuntu)
# - la cible importée ne contient pas la liste des fichiers
#   d'en-tête nécessaires.
#
# Le seul moyen qui semble identique quelles que soient les
# installations est d'utilier les variables '<pkg>_INCLUDE_DIRS'
# et '<pkg>_LIBRARIES'. On utilise donc ce mécanisme.
#
arccon_return_if_package_found(TrilinosAleph)

set(Trilinos_TARGETS_IMPORTED 1)
find_package(AztecOO QUIET)
find_package(ML QUIET)
find_package(Ifpack QUIET)
find_package(Epetra QUIET)

message(STATUS "AztecOO_FOUND " ${AztecOO_FOUND})
message(STATUS "ML_FOUND " ${ML_FOUND})
message(STATUS "Ifpack_FOUND " ${Ifpack_FOUND})
message(STATUS "Epetra_FOUND " ${Epetra_FOUND})

set(TrilinosAleph_FOUND "NO")

if (AztecOO_FOUND AND ML_FOUND AND Ifpack_FOUND AND Epetra_FOUND)
  message(STATUS "Found Trilinos components for Aleph")
  SET(TrilinosAleph_INCLUDE_DIRS "${AztecOO_INCLUDE_DIRS}" "${ML_INCLUDE_DIRS}" "${Ifpack_INCLUDE_DIRS}" "${Epetra_INCLUDE_DIRS}")
  SET(TrilinosAleph_LIBRARIES "${AztecOO_LIBRARIES}" "${ML_LIBRARIES}" "${Ifpack_LIBRARIES}" "${Epetra_LIBRARIES}")
  if (TrilinosAleph_INCLUDE_DIRS AND TrilinosAleph_LIBRARIES)
    set(TrilinosAleph_FOUND YES)
  endif()
endif()

# Remove duplicate libraries, keeping the last (for linking)
if (TrilinosAleph_FOUND)
  set(TrilinosAleph_FOUND TRUE)
  LIST(REVERSE TrilinosAleph_LIBRARIES)
  LIST(REMOVE_DUPLICATES TrilinosAleph_LIBRARIES)
  LIST(REVERSE TrilinosAleph_LIBRARIES)
  arccon_register_package_library(TrilinosAleph TrilinosAleph)
endif ()

message(STATUS "TrilinosAleph_LIBRARIES = ${TrilinosAleph_LIBRARIES}")
message(STATUS "TrilinosAleph_INCLUDE_DIRS = '${TrilinosAleph_INCLUDE_DIRS}'")
message(STATUS "TrilinosAleph_FOUND = '${TrilinosAleph_FOUND}'")

if (NOT TrilinosAleph_FOUND)
  unset(TrilinosAleph_INCLUDE_DIRS)
  unset(TrilinosAleph_LIBRARIES)
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
