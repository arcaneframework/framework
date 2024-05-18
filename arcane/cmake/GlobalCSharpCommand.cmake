# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(ARCANE_USE_GLOBAL_CSHARP)
  set(ARGS_DOTNET_RUNTIME "coreclr")
  set(_msbuild_exe ${ARCCON_MSBUILD_EXEC_${ARGS_DOTNET_RUNTIME}})
  if (NOT _msbuild_exe)
    logFatalError("In ${_func_name}: 'msbuild' command for runtime '${ARGS_DOTNET_RUNTIME}' is not available.")
  endif()
  # TODO: Faire la restauration avant
  set(_BUILD_ARGS publish --no-restore BuildAllCSharp.proj /t:Publish /p:PublishDir=${ARCANE_DOTNET_PUBLISH_PATH}/ ${ARGS_MSBUILD_ARGS})

  get_property(_ALL_DEPEND_LIST TARGET arcane_global_csharp_target
    PROPERTY DEPENDS
  )
  message(STATUS "GLOBAL_CSHARP_DEPENDS=${_ALL_DEPEND_LIST}")

  # Commande de restauration des packages nuget
  add_custom_command(OUTPUT "${ARCANE_DOTNET_RESTORE_TIMESTAMP}"
    WORKING_DIRECTORY "${ARCANE_CSHARP_PROJECT_PATH}"
    COMMAND ${_msbuild_exe} build BuildAllCSharp.proj /t:Restore
    COMMAND ${CMAKE_COMMAND} -E touch ${ARCANE_DOTNET_RESTORE_TIMESTAMP}
    COMMENT "Restoring global 'C#' target"
  )
  message(STATUS "Add Custom command for C#")
  # Commande de compilation de la cible
  add_custom_command(OUTPUT "${ARCANE_DOTNET_PUBLISH_TIMESTAMP}"
    WORKING_DIRECTORY "${ARCANE_CSHARP_PROJECT_PATH}"
    COMMAND ${_msbuild_exe} ${_BUILD_ARGS}
    COMMAND ${CMAKE_COMMAND} -E touch ${ARCANE_DOTNET_PUBLISH_TIMESTAMP}
    DEPENDS arcane_global_csharp_restore_target ${_ALL_DEPEND_LIST}
    COMMENT "Building global 'C#' target"
  )

  # TODO: Ajouter cible pour le pack
  # TODO: Ajouter cible pour forcer la compilation
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
