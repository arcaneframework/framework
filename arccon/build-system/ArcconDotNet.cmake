# Positionne les variables suivantes:
# ARCCON_DOTNET_FOUND                (TRUE|FALSE) si le framework .NET est trouvé.
# ARCCON_DOTNET_CSC                  chemin du compilateur C#
# ARCCON_DOTNET_CSC_DEBUG_EXTENSION  extension pour les symboles de débug (mdb|pdb)
# ARCCON_DOTNET_MSBUILD              outils de build (msbuild|xbuild)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Gestion du C#.
# Arguments communs à l'utilisation avec 'dotnet' ou 'mono'.
# La variable '${ARCCON_MSBUILD_RESTORE_ARGS}' peut être définie par l'utilisateur
# pour spécifier des arguments pour la restauration, comme par exemple
# le chemin des packages NuGet si on n'a pas accès à internet.
set(ARCCON_MSBUILD_COMMON_ARGS /nodeReuse:false ${ARCCON_MSBUILD_RESTORE_ARGS})

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Regarde si le framework 'dotnet' est disponible.
# Si c'est le cas, alors 'ARCCON_HAS_DOTNET' est mis à TRUE.
#
find_program(DOTNET_EXEC NAMES dotnet)
message(STATUS "[.Net] DOTNET exe: ${DOTNET_EXEC}")
if (DOTNET_EXEC)
  set(ARCCON_DOTNET_HAS_RUNTIME_coreclr TRUE)
  set(ARCCON_MSBUILD_EXEC_coreclr ${DOTNET_EXEC})
  # Pour les options, voir le CMakeLists.txt de 'axlstar'
  set(ARCCON_MSBUILD_ARGS_coreclr publish /p:UseSharedCompilation=false ${ARCCON_MSBUILD_COMMON_ARGS})
  # Arguments pour fabriquer les packages NuGet
  set(ARCCON_DOTNET_PACK_ARGS_coreclr pack --no-build --no-restore --no-dependencies /nodeReuse:false)

  # Récupère le numéro de version 'dotnet'
  execute_process(COMMAND ${DOTNET_EXEC} "--version" OUTPUT_VARIABLE CORECLR_EXEC_VERSION_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "([0-9]+)\.([0-9]+)\.(.*)" CORECLR_VERSION_REGEX_MATCH ${CORECLR_EXEC_VERSION_OUTPUT})
  set(CORECLR_VERSION ${CMAKE_MATCH_1}.${CMAKE_MATCH_2})
  set(CORECLR_VERSION_FULL ${CORECLR_VERSION}.${CMAKE_MATCH_3})
  message(STATUS "[.Net]: CORECLR_VERSION = ${CORECLR_VERSION} (full=${CORECLR_VERSION_FULL})")
else()
  message(STATUS "[.Net]: no 'dotnet' exec found")
  set(ARCCON_DOTNET_HAS_RUNTIME_coreclr FALSE)
endif()

# Mono is no longer supported (09/2024)
set(ARCCON_DOTNET_HAS_RUNTIME_mono FALSE)

if (ARCCON_DOTNET_HAS_RUNTIME_coreclr)
  set(ARCCON_HAS_DOTNET TRUE)
endif()
set(ARCCON_HAS_DOTNET ${ARCCON_HAS_DOTNET} CACHE BOOL "True if .NET environment ('coreclr' or 'mono') is found" FORCE)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Fonction pour installer une assembly .NET
# Les fichiers sont installés dans le chemin relatif 'rel_path'
# Obsolète: ne plus utiliser. Utiliser 'arccon_add_csharp_target' à la place
# avec 'arccon_dotnet_install_publish_directory'
function(arccon_install_clr assembly_name rel_path)
  set(full_output_name "${CMAKE_BINARY_DIR}/${rel_path}/${assembly_name}")
  install(FILES ${full_output_name} DESTINATION ${rel_path})
  # Installe les fichiers de debug s'ils existent
  # Si l'extension est 'mdb', alors elle s'ajoute au nom de l'assembly.
  # Si l'extension est 'pdb', alors elle la remplace.
  # Par exemple, pour 'toto.dll', alors c'est 'toto.dll.mdb' ou 'toto.pdb'
  if (CSC_DEBUG_EXTENSION STREQUAL "mdb")
    install(FILES ${full_output_name}.mdb DESTINATION ${rel_path} OPTIONAL)
  endif()
  if (CSC_DEBUG_EXTENSION STREQUAL "pdb")
    string(REGEX REPLACE ".(dll|exe)$" "" _name_without_extension ${full_output_name})
    install(FILES ${_name_without_extension}.pdb DESTINATION ${rel_path} OPTIONAL)
  endif()
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Fonction pour appeler 'mkbundle' sur un exe.
#
# NOTE: Avec l'utilisation de 'coreclr', cette function ne doit plus être
# utilisée.
#
# 'mkbundle' est un outil de 'mono' qui permet de transformer un exe .NET en
# en executable C classique qui est autonome et n'a pas besoin d'avoir 'mono'
# installé pour fonctionner.
#
# Usage:
#   arccon_dotnet_mkbundle(
#     BUNDLE out_exe
#     EXE    dotnet_exe
#     DLLs   dotnet_dlls_to_embed
#   )
function(arccon_dotnet_mkbundle)
  set(options)
  set(oneValueArgs BUNDLE EXE)
  set(multiValueArgs DLLs)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()
  get_filename_component(File ${ARGS_BUNDLE} NAME)
  if(WIN32)
    add_custom_command(
      OUTPUT  ${ARGS_BUNDLE}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ARGS_EXE} ${ARGS_BUNDLE}
      DEPENDS ${ARGS_EXE} ${ARGS_DLLs}
      )
  else()
    # mkbundle ne peut être lancé en parallèle dans un même répertoire
    # on utilise un répertoire temporaire et WORKING_DIRECTORY de add_custom_command
    # L'option '--skip-scan' permet à mkbundle de ne pas lever d'exceptions si une des assembly
    # n'est pas lisible.
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/dotnet/tmp/${File})
    add_custom_command(
      OUTPUT  ${ARGS_BUNDLE}
      COMMAND ${Mkbundle_EXEC} ${ARGS_EXE} ${ARGS_DLLs} -o ${ARGS_BUNDLE} --deps --static --skip-scan
      DEPENDS ${ARGS_EXE} ${ARGS_DLLs} ${Mkbundle_EXEC}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/dotnet/tmp/${File}
      )
  endif()
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#
# Copie à l'installation un répertoire où ont été publiés des
# fichier '.NET'.
#
# Usage:
#
#  arccon_dotnet_install_publish_directory(
#    DIRECTORY published_directory
#    DESTINATION destination_dir
#  )
#
function(arccon_dotnet_install_publish_directory)
  set(_func_name "arccon_dotnet_install_publish_directory")
  set(options)
  set(oneValueArgs DIRECTORY DESTINATION)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("In ${_func_name}: unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()
  if (NOT ARGS_DIRECTORY)
    logFatalError("In ${_func_name}: DIRECTORY not specified")
  endif()
  if (NOT ARGS_DESTINATION)
    logFatalError("In ${_func_name}: DESTINATION not specified")
  endif()
  arccon_install_directory(NAMES ${ARGS_DIRECTORY} DESTINATION ${ARGS_DESTINATION})
endfunction()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Propriété 'DOTNET_DLL_NAME' contenant le nom de la 'DLL' '.Net' générée
# par la cible
define_property(TARGET
  PROPERTY DOTNET_DLL_NAME
  BRIEF_DOCS "Name of generated 'DLL' by this target"
  FULL_DOCS "Name of generated 'DLL' by this target"
  )

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#
# Fonction pour créér une cible 'CMake' à partir d'un projet ou solution C#
#
# Usage:
#
#  arccon_add_csharp_target(target_name
#    DOTNET_RUNTIME [mono|coreclr]
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
function(arccon_add_csharp_target target_name)
  set(_func_name "arccon_add_csharp_target")
  set(options PACK)
  set(oneValueArgs BUILD_DIR TARGET_PATH ASSEMBLY_NAME PROJECT_NAME PROJECT_PATH DOTNET_RUNTIME)
  set(multiValueArgs MSBUILD_ARGS DEPENDS DOTNET_TARGET_DEPENDS)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(ARGS_UNPARSED_ARGUMENTS)
    logFatalError("In ${_func_name}: unparsed arguments '${ARGS_UNPARSED_ARGUMENTS}'")
  endif()
  if (NOT target_name)
    logFatalError("In ${_func_name}: no 'target_name' specified")
  endif()
  if (NOT ARGS_DOTNET_RUNTIME)
    logFatalError("In ${_func_name}: DOTNET_RUNTIME not specified")
  endif()
  if (NOT ${ARGS_DOTNET_RUNTIME} MATCHES "coreclr|mono")
    logFatalError("In ${_func_name}: Invalid value '${ARGS_DOTNET_RUNTIME}' for DOTNET_RUNTIME. Valid values are 'coreclr' or 'mono'")
  endif()
  if (NOT ARCCON_DOTNET_HAS_RUNTIME_${ARGS_DOTNET_RUNTIME})
    logFatalError("In ${_func_name}: Requested runtime '${ARGS_DOTNET_RUNTIME}' is not available")
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
  set(_msbuild_exe ${ARCCON_MSBUILD_EXEC_${ARGS_DOTNET_RUNTIME}})
  if (NOT _msbuild_exe)
    logFatalError("In ${_func_name}: 'msbuild' command for runtime '${ARGS_DOTNET_RUNTIME}' is not available.")
  endif()
  set(_BUILD_ARGS ${ARCCON_MSBUILD_ARGS_${ARGS_DOTNET_RUNTIME}} ${ARGS_PROJECT_NAME} /t:Publish /p:PublishDir=${ARGS_BUILD_DIR}/ ${ARGS_MSBUILD_ARGS})
  # Comme 'cmake' ne propage pas les dépendances de fichiers entre les 'add_custom_command'
  # et 'add_custom_target', il faut le faire manuellement. Pour cela, on utilise
  # notre propriété 'DOTNET_DLL_NAME' définie sur les cibles '.Net' et on
  # ajoute explicitement aux dépendences ce fichier.
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
  #message(STATUS "_ALL_DEPENDS=${_ALL_DEPENDS}")

  if (ARGS_PACK)
    set(_DO_PACK TRUE)
  endif()
  if (_DO_PACK)
    set(_PACK_DIR ${CMAKE_BINARY_DIR}/nupkgs)
    file(MAKE_DIRECTORY ${_PACK_DIR})
    set(_PACK_ARGS ${ARCCON_DOTNET_PACK_ARGS_${ARGS_DOTNET_RUNTIME}} /p:PackageOutputPath=${_PACK_DIR} /p:IncludeSymbols=true ${ARGS_PROJECT_NAME} ${ARGS_MSBUILD_ARGS})
  endif()
  if (_DO_PACK)
    add_custom_command(OUTPUT ${output_assembly_path}
      WORKING_DIRECTORY ${ARGS_PROJECT_PATH}
      COMMAND ${_msbuild_exe} ${_BUILD_ARGS}
      COMMAND ${_msbuild_exe} ${_PACK_ARGS}
      DEPENDS ${_ALL_DEPENDS}
      COMMENT "Building and packing 'C#' target '${target_name}' (expected output '${output_assembly_path}')"
      )
    else()
    add_custom_command(OUTPUT ${output_assembly_path}
      WORKING_DIRECTORY ${ARGS_PROJECT_PATH}
      COMMAND ${_msbuild_exe} ${_BUILD_ARGS}
      DEPENDS ${_ALL_DEPENDS}
      COMMENT "Building 'C#' target '${target_name}' (expected output '${output_assembly_path}')"
      )
  endif()
  add_custom_target(${target_name} ALL DEPENDS ${output_assembly_path} ${ARGS_DOTNET_TARGET_DEPENDS})
  # Indique que la cible génère la dll '${output_assembly_path}'
  set_target_properties(${target_name} PROPERTIES DOTNET_DLL_NAME ${output_assembly_path})

  # Cible pour forcer la compilation du projet
  if (_DO_PACK)
    add_custom_target(force_${target_name}
      WORKING_DIRECTORY ${ARGS_PROJECT_PATH}
      COMMAND ${_msbuild_exe} ${_BUILD_ARGS}
      COMMAND ${_msbuild_exe} ${_PACK_ARGS}
      COMMENT "Force Building and packing 'C#' target '${target_name}'"
      )
  else()
    add_custom_target(force_${target_name}
      WORKING_DIRECTORY ${ARGS_PROJECT_PATH}
      COMMAND ${_msbuild_exe} ${_BUILD_ARGS}
      COMMENT "Force Building 'C#' target '${target_name}'"
      )
  endif()

endfunction()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
