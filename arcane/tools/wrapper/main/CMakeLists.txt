set(ARCANE_SWIG_MAIN_CSHARP_FILES
  ArcaneMainExec
)

arcane_wrapper_add_csharp_target(TARGET_NAME dotnet_wrapper_main
  PROJECT_NAME Arcane.Main
  CSHARP_SOURCES ${ARCANE_SWIG_MAIN_CSHARP_FILES}
  DOTNET_TARGET_DEPENDS dotnet_wrapper_launcher
  EXECUTABLE
)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
