# ----------------------------------------------------------------------------
# Liste des fichiers utilisés par ce wrapping à installer
set(ARCANE_SWIG_CORE_INSTALL_FILES
  ArcaneSwigCore.i
  Collection.i
  ItemPairGroup.i
  Interface.i
  InterfacesDeclaration.i
  InterfacesInclude.i
  Mesh.i
  Parallel.i
  Ref.i
  StringView.i
  String.i
  CaseOption.i
  ItemGroup.i
  Numeric.i
  Service.i
  Variables.i
  DataVisitor.i
  ArrayView.i
  MemoryView.i
  DotNetObject.i
  )

# ----------------------------------------------------------------------------
# Liste des fichiers C# compilés qui sont aussi utilisés pour la gestion des
# dépendances (si l'un de ces fichiers est modifié, 'cmake' relance la
# génération de la cible
set(ARCANE_SWIG_CORE_CSHARP_FILES
  ArcaneMain
  ArcaneSimpleExecutor
  AssemblyLoaderHelper
  CaseOptionMultiEnumT
  CaseFunction
  Debug
  ExternalRef
  EntryPoint
  ItemInternal
  Item
  ItemLocalIdListContainerView
  ItemIndexArrayView
  ItemVectorViewT
  Math
  CaseOptionServiceContainer
  CaseOptionEnumT
  CaseOptionServiceT
  CaseOptionMultiServiceT
  IObserver
  ItemItemEnumerator
  Variable
  BasicModule
  CaseOptionExtendedT
  ItemEnumerator
  ModuleAttribute
  ServiceAttribute
  ServiceFactory
  ITraceMng
  TraceAccessor
  )

# ----------------------------------------------------------------------------
# Liste des fichiers utilisés qu'il n'est pas nécessaire d'installer
set(ARCANE_SWIG_CORE_FILES
  ${ARCANE_SWIG_CORE_INSTALL_FILES}
  ArcaneSwigCoreInclude.h
  )

# ----------------------------------------------------------------------------
# Installe les .cs et les .i (temporaire) pour ceux qui souhaitent wrapper Arcane.

install(FILES
  ${ARCANE_SWIG_CORE_INSTALL_FILES}
  DESTINATION share/wrapper/swig/core
  )

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Wrapping C# des classes de base (Core)

arcane_wrapper_add_swig_target(NAME core SOURCE ArcaneSwigCore.i NAMESPACE_NAME Arcane
  DLL_NAME Arcane.Core
  CSHARP_SOURCES ${ARCANE_SWIG_CORE_CSHARP_FILES}
  )

# Indique qu'il faut avoir compiler 'Arcane.Utils' avant de compiler les sources
# générés par le wrapper.
add_dependencies(dotnet_wrapper_core dotnet_arcane_utils)

# Indique les cibles dont dépend 'arcane_dotnet_wrapper_core'
target_link_libraries(arcane_dotnet_wrapper_core PUBLIC arcane_impl arcane_core)

# Indique les .i dont dépend 'ArcaneSwigCore.i'
set_property(TARGET arcane_dotnet_wrapper_core PROPERTY SWIG_DEPENDS ${ARCANE_SWIG_CORE_FILES})

add_subdirectory(tests)

# ----------------------------------------------------------------------------
# Local Variables:
# tab-width: 2
# indent-tabs-mode: nil
# coding: utf-8-with-signature
# End:
