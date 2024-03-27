# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Wrapping C# du support de la composante 'launcher'.

set(ARCANE_SWIG_LAUNCHER_FILES
  ArcaneSwigLauncher.i
)
if (ARCANE_ENABLE_DOTNET_PYTHON_WRAPPER)
  # Cette variable est utilisée lors de la génération du fichier '.csproj'
  set(_HAS_PYTHON_WRAPPER "true")
endif()

arcane_wrapper_add_swig_target(NAME launcher SOURCE ArcaneSwigLauncher.i NAMESPACE_NAME Arcane
  DLL_NAME Arcane.Launcher
  CSHARP_SOURCES ArcaneDefaultCaseFunctionProvider
  SWIG_TARGET_DEPENDS core services)
target_link_libraries(arcane_dotnet_wrapper_launcher PUBLIC arcane_launcher arcane_std)

# Indique les .i dont dépend 'ArcaneSwigLauncher.i'
set_property(TARGET arcane_dotnet_wrapper_launcher PROPERTY SWIG_DEPENDS ${ARCANE_SWIG_LAUNCHER_FILES})

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
