# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if(ARCANE_USE_GLOBAL_CSHARP)
  set(ARGS_DOTNET_RUNTIME "coreclr")
  set(_msbuild_exe ${ARCCON_MSBUILD_EXEC_${ARGS_DOTNET_RUNTIME}})
  if (NOT _msbuild_exe)
    logFatalError("In ${_func_name}: 'msbuild' command for runtime '${ARGS_DOTNET_RUNTIME}' is not available.")
  endif()
  # TODO: Faire la restauration avant
  set(_BUILD_ARGS publish --no-restore BuildAllCSharp.proj /t:PublishAndPack /p:PublishDir=${ARCANE_DOTNET_PUBLISH_PATH}/ ${ARGS_MSBUILD_ARGS})

  get_property(_ALL_DEPEND_LIST TARGET arcane_global_csharp_target
    PROPERTY DEPENDS
  )
  set(_PACK_DIR ${CMAKE_BINARY_DIR}/nupkgs)
  file(MAKE_DIRECTORY ${_PACK_DIR})
  set(_PACK_ARGS /p:PackageOutputPath=${_PACK_DIR} /p:IncludeSymbols=true)

  if (ARCANE_HAS_DOTNET_WRAPPER)
    # Commande de restauration des packages nuget
    add_custom_command(OUTPUT "${ARCANE_DOTNET_RESTORE_TIMESTAMP}"
      WORKING_DIRECTORY "${ARCANE_CSHARP_PROJECT_PATH}"
      COMMAND ${_msbuild_exe} build BuildAllCSharp.proj /t:Restore
      COMMAND ${CMAKE_COMMAND} -E touch ${ARCANE_DOTNET_RESTORE_TIMESTAMP}
      COMMENT "Restoring global 'C#' target"
    )

    message(STATUS "Adding Custom command for C#")

    # Commande de compilation de la cible
    add_custom_command(OUTPUT "${ARCANE_DOTNET_PUBLISH_TIMESTAMP}"
      WORKING_DIRECTORY "${ARCANE_CSHARP_PROJECT_PATH}"
      COMMAND ${_msbuild_exe} ${_BUILD_ARGS} ${_PACK_ARGS}
      COMMAND ${CMAKE_COMMAND} -E touch ${ARCANE_DOTNET_PUBLISH_TIMESTAMP}
      DEPENDS arcane_global_csharp_restore_target ${_ALL_DEPEND_LIST}
      COMMENT "Building and packing global 'C#' target"
    )
  else()
    # Si le wrapper n'est pas actif, on créé une commande qui ne fait rien
    add_custom_command(OUTPUT "${ARCANE_DOTNET_RESTORE_TIMESTAMP}"
      WORKING_DIRECTORY "${ARCANE_CSHARP_PROJECT_PATH}"
      COMMAND ${CMAKE_COMMAND} -E touch ${ARCANE_DOTNET_RESTORE_TIMESTAMP}
    )
    add_custom_command(OUTPUT "${ARCANE_DOTNET_PUBLISH_TIMESTAMP}"
      WORKING_DIRECTORY "${ARCANE_CSHARP_PROJECT_PATH}"
      COMMAND ${CMAKE_COMMAND} -E touch ${ARCANE_DOTNET_PUBLISH_TIMESTAMP}
    )
  endif()

  # TODO: Ajouter cible pour forcer la compilation
endif()

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
