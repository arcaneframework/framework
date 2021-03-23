# ----------------------------------------------------------------------------
# Exemple d'implémentation avec un service de gaz parfait.

configure_file(PerfectGas.csproj.in ${CMAKE_CURRENT_BINARY_DIR}/PerfectGas.csproj @ONLY)

arcane_sample_add_csharp_target(TARGET_NAME dotnet_eos_csharp_perfectgas
  PROJECT_NAME PerfectGas
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  PUBLISH_DIRECTORY ${EOS_BINARY_DIR}
  DEPENDS dotnet_eos_csharp_wrapper
  )

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
