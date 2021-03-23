# ----------------------------------------------------------------------------
# Exemple d'exécutable C# pour lancer un code avec un 'main' en C#.

configure_file(EOS.Main.csproj.in ${CMAKE_CURRENT_BINARY_DIR}/EOS.Main.csproj @ONLY)

arcane_sample_add_csharp_target(TARGET_NAME dotnet_eos_csharp_main
  PROJECT_NAME EOS.Main
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  PUBLISH_DIRECTORY ${EOS_BINARY_DIR}
  DEPENDS dotnet_eos_csharp_perfectgas dotnet_eos_csharp_wrapper
  )

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
