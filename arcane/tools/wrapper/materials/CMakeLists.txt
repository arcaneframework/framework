
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Wrapping C# du support des matériaux.

set(ARCANE_SWIG_CEA_MATERIALS_FILES
  ArcaneSwigCeaMaterials.i
  ComponentItemVector.i
  MeshMaterialVariable.i
)

set(ARCANE_SWIG_CEA_MATERIALS_CSHARP_FILES
  Component
  ComponentItem
  ComponentItemEnumerator
  ComponentItemInternal
  ComponentItemVectorView
  MatVarIndex
)

# ----------------------------------------------------------------------------

install(FILES
  ${ARCANE_SWIG_CEA_MATERIALS_FILES}
  DESTINATION share/wrapper/swig/materials
  )

arcane_wrapper_add_swig_target(NAME cea_materials
  SOURCE ArcaneSwigCeaMaterials.i
  NAMESPACE_NAME Arcane.Materials
  DLL_NAME Arcane.Cea.Materials
  SWIG_TARGET_DEPENDS core
  CSHARP_SOURCES ${ARCANE_SWIG_CEA_MATERIALS_CSHARP_FILES}
  )

target_link_libraries(arcane_dotnet_wrapper_cea_materials PUBLIC arcane_core)

# Indique les .i dont dépend ce wrapper
set_property(TARGET arcane_dotnet_wrapper_cea_materials PROPERTY SWIG_DEPENDS ${ARCANE_SWIG_CEA_MATERIALS_FILES})

# Indique qu'on dépend du wrapping de Arcane.Core
arcane_wrapper_add_swig_dependency(cea_materials core)
