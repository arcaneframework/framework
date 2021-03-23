# Wrapping C# des services.

arcane_wrapper_add_swig_target(NAME services SOURCE ArcaneSwigServices.i
  NAMESPACE_NAME Arcane
  DLL_NAME Arcane.Services
  SWIG_TARGET_DEPENDS core)

target_link_libraries(arcane_dotnet_wrapper_services PUBLIC arcane_std)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
