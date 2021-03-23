# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Liste des fichiers C# compilés qui sont aussi utilisés pour la gestion des
# dépendances (si l'un de ces fichiers est modifié, 'cmake' relance la
# génération de la cible
set(ARCANE_SWIG_CORE_TESTS_CSHARP_FILES
  Test1
  )

# ----------------------------------------------------------------------------
# Liste des fichiers utilisés qu'il n'est pas nécessaire d'installer
set(ARCANE_SWIG_CORE_TESTS_FILES
  ArcaneSwigCoreTests.i
  )

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Wrapping C# des classes de base (Core)

arcane_wrapper_add_swig_target(NAME core_tests SOURCE ArcaneSwigCoreTests.i
  NAMESPACE_NAME Arcane
  DLL_NAME Arcane.Core.Tests
  SWIG_TARGET_DEPENDS core
  CSHARP_SOURCES ${ARCANE_SWIG_CORE_TESTS_CSHARP_FILES}
  )

# Indique les .i dont dépend 'ArcaneSwigCoreTests.i'
set_property(TARGET arcane_dotnet_wrapper_core_tests PROPERTY SWIG_DEPENDS ${ARCANE_SWIG_CORE_TESTS_FILES})

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
