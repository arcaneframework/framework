# Fonctions utilitaires pour compiler les exemples

# Ajoute une cible pour compiler un projet 'C#'
function(arcane_sample_add_csharp_target)

  set(options        )
  set(oneValueArgs   TARGET_NAME PROJECT_NAME WORKING_DIRECTORY PUBLISH_DIRECTORY)
  set(multiValueArgs DEPENDS)

  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT ARGS_TARGET_NAME)
    message(FATAL_ERROR "TARGET_NAME not set")
  endif()
  if (NOT ARGS_PROJECT_NAME)
    message(FATAL_ERROR "PROJECT_NAME not set")
  endif()
  if (NOT ARGS_WORKING_DIRECTORY)
    message(FATAL_ERROR "WORKING_DIRECTORY not set")
  endif()
  if (NOT ARGS_PUBLISH_DIRECTORY)
    message(FATAL_ERROR "PUBLISH_DIRECTORY not set")
  endif()

  set(OUTPUT_DLL ${ARGS_PUBLISH_DIRECTORY}/${ARGS_PROJECT_NAME}.dll)
  # Ajoute une commande pour compiler le C# généré par 'swig'
  add_custom_command(OUTPUT ${OUTPUT_DLL}
    WORKING_DIRECTORY ${ARGS_WORKING_DIRECTORY}
    COMMAND ${ARCANE_MSBUILD_EXEC} restore ${ARGS_PROJECT_NAME}.csproj --packages ${ARGS_WORKING_DIRECTORY}
    COMMAND ${ARCANE_MSBUILD_EXEC} publish -o ${ARGS_PUBLISH_DIRECTORY} --no-restore ${ARGS_PROJECT_NAME}.csproj
    COMMENT "Compiling C# extension for '${ARGS_PROJECT_NAME}'"
    DEPENDS ${ARGS_WORKING_DIRECTORY}/${ARGS_PROJECT_NAME}.csproj ${ARGS_DEPENDS}
    )

  # Ajoute une cible pour la commande de compilation du C# généré
  add_custom_target(${ARGS_TARGET_NAME} ALL DEPENDS ${OUTPUT_DLL})
endfunction()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
