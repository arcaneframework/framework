# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Pour tester l'utilisation d'un cible globale pour le C#
option(ARCANE_USE_GLOBAL_CSHARP "True if we use global C# project (experimental)" OFF)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(ARCANE_USE_GLOBAL_CSHARP)
  set(ARCANE_DOTNET_PUBLISH_PATH "${CMAKE_BINARY_DIR}/lib")
  # Fichier pour savoir si on a déjà générer la cible
  set(ARCANE_DOTNET_PUBLISH_TIMESTAMP "${ARCANE_DOTNET_PUBLISH_PATH}/.dotnet_stamp")
  # Fichier pour savoir si on a déjà restaurer la cible
  set(ARCANE_DOTNET_RESTORE_TIMESTAMP "${ARCANE_DOTNET_PUBLISH_PATH}/.dotnet_restore_stamp")

  add_custom_target(arcane_global_csharp_target ALL DEPENDS "${ARCANE_DOTNET_PUBLISH_TIMESTAMP}")
  add_custom_target(arcane_global_csharp_restore_target ALL DEPENDS "${ARCANE_DOTNET_RESTORE_TIMESTAMP}")

  # TODO: Ajouter cible pour le pack
  # TODO: Ajouter cible pour forcer la compilation
endif()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#
# Fonction pour créér une cible 'CMake' à partir d'un projet ou solution C#
# La compilation de toutes les projets générés par cette fonction sera
# effectuée en une fois
#
# Usage:
#
#  arcane_add_global_csharp_target(target_name
#    BUILD_DIR [target_path]
#    ASSEMBLY_NAME [assembly_name]
#    PROJECT_PATH [project_path]
#    PROJECT_NAME [project_name]
#    MSBUILD_ARGS [msbuild_args]
#    DEPENDS [depends]
#    DOTNET_TARGET_DEPENDS [dotnet_target_depends]
#    [PACK]
#  )
#
function(arcane_add_global_csharp_target target_name)
  # Appelle l'ancienne méthode tant que la nouvelle n'est pas finalisée
  if (NOT ARCANE_USE_GLOBAL_CSHARP)
    arccon_add_csharp_target(${target_name} DOTNET_RUNTIME coreclr ${ARGN})
    return()
  endif()
  set(_func_name "arcane_add_global_csharp_target")
  set(options PACK)
  set(oneValueArgs BUILD_DIR TARGET_PATH ASSEMBLY_NAME PROJECT_NAME PROJECT_PATH)
  set(multiValueArgs MSBUILD_ARGS DEPENDS DOTNET_TARGET_DEPENDS)
  set(ARGS_DOTNET_RUNTIME "coreclr")
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("In ${_func_name}: unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()
  if (NOT target_name)
    logFatalError("In ${_func_name}: no 'target_name' specified")
  endif()
  if (NOT ARGS_BUILD_DIR)
    logFatalError("In ${_func_name}: BUILD_DIR not specified")
  endif()
  if (NOT ARGS_ASSEMBLY_NAME)
    logFatalError("In ${_func_name}: ASSEMBLY_NAME not specified")
  endif()
  if (NOT ARGS_PROJECT_PATH)
    logFatalError("In ${_func_name}: PROJECT_PATH not specified")
  endif()
  if (NOT ARGS_PROJECT_NAME)
    logFatalError("In ${_func_name}: PROJECT_NAME not specified")
  endif()
  set(assembly_name ${ARGS_ASSEMBLY_NAME})
  set(build_proj_path ${ARGS_PROJECT_PATH}/${ARGS_PROJECT_NAME})
  set(output_assembly_path ${ARGS_BUILD_DIR}/${assembly_name})
  set(_DOTNET_TARGET_DLL_DEPENDS)
  if (ARGS_DOTNET_TARGET_DEPENDS)
    #message(STATUS "ARGS_DOTNET_TARGET_DEPENDS=${ARGS_DOTNET_TARGET_DEPENDS}")
    foreach(_dtarget ${ARGS_DOTNET_TARGET_DEPENDS})
      get_target_property(_dtarget_file ${_dtarget} DOTNET_DLL_NAME)
      #message(STATUS "TARGET ${_dtarget} dll_file=${_dtarget_file}")
      list(APPEND _DOTNET_TARGET_DLL_DEPENDS ${_dtarget_file})
    endforeach()
  endif()
  message(STATUS "_DOTNET_TARGET_DLL_DEPENDS=${_DOTNET_TARGET_DLL_DEPENDS}")
  set(_ALL_DEPENDS ${build_proj_path} ${ARGS_DEPENDS} ${_DOTNET_TARGET_DLL_DEPENDS} ${ARGS_DOTNET_TARGET_DEPENDS})

  # Ajoute les fichiers à la dépendance de la cible globale
  # Cela est utilisé ensuite dans 'GlobalCSharpCommand.cmake' pour générer la liste de dépendances
  set_property(TARGET arcane_global_csharp_target
    APPEND PROPERTY
    DEPENDS ${_ALL_DEPENDS}
  )

  add_custom_target(${target_name})
  add_dependencies(arcane_global_csharp_target ${target_name})

  # Indique que la cible génère la dll '${output_assembly_path}'
  set_target_properties(${target_name} PROPERTIES DOTNET_DLL_NAME ${output_assembly_path})
endfunction()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
