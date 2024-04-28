# Wrapping C# pour le python

arcane_wrapper_add_swig_target(NAME python SOURCE ArcaneSwigPython.i
  NAMESPACE_NAME Arcane
  DLL_NAME Arcane.Python
  SWIG_TARGET_DEPENDS core services)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
