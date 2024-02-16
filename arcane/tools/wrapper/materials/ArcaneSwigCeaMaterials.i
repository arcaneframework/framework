%module(directors="1") ArcaneCeaMaterials

%import core/ArcaneSwigCore.i

%{
#define ARCANE_DOTNET
#include "ArcaneSwigUtils.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshBlock.h"
#include "arcane/core/materials/IMeshEnvironment.h"
#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/ComponentPartItemVectorView.h"
#include "arcane/core/materials/ComponentItemVectorView.h"
#include "arcane/core/materials/CellToAllEnvCellConverter.h"
#include "arcane/core/materials/MeshMaterialVariableRef.h"

namespace Arcane::Materials
{
  class ConstituentItemLocalIdListViewPOD
  {
   public:
    ComponentItemSharedInfo* m_component_shared_info;
    ConstArrayView<Int32> m_ids;
    ConstArrayView<ComponentItemInternal*> m_items_internal;
  };

  // Cette classe sert de type de retour pour wrapper la classe 'ComponentItemVectorView'
  class ComponentItemVectorViewPOD
  {
   public:
    ConstArrayView<MatVarIndex> m_matvar_indexes_view;
    ConstituentItemLocalIdListViewPOD m_items_internal_main_view;
    ConstArrayView<Int32> m_items_local_id_view;
    IMeshComponent* m_component;
  };
}

using namespace Arcane;
using namespace Arcane::Materials;

namespace
{
  ComponentItemVectorViewPOD _createComponentItemVectorViewPOD(const ComponentItemVectorView& view)
  {
    ComponentItemVectorViewPOD pod;
    size_t size1 = sizeof(ComponentItemVectorViewPOD);
    size_t size2 = sizeof(ComponentItemVectorView);
    if (size1!=size2)
      ARCANE_FATAL("Bad size for POD copy size1={0} size2={1}",size1,size2);
    std::memcpy(&pod,&view,size1);
    return pod;
  }
}

%}

#define ARCANE_DOTNET

// Supprime temporairement ces méthodes car elles ne sont pas bien wrappées
%ignore Arcane::Materials::IMeshMaterialMng::view;
%ignore Arcane::Materials::IMeshMaterialMng::cellToAllEnvCellConverter;

// Supprime cette méthode obsolète
%ignore Arcane::Materials::IMeshMaterialMng::allEnvCells;

// Supprime temporairement ces méthodes car elles ne sont pas bien wrappées
%ignore Arcane::Materials::IMeshComponent::findComponentCell;
%ignore Arcane::Materials::IMeshMaterial::findMatCell;
%ignore Arcane::Materials::IMeshEnvironment::findEnvCell;
%ignore Arcane::Materials::IMeshBlock::view;
%rename("$ignore", regextarget=1, fullname=1) "Arcane::Materials::CellMaterialVariableScalarRef<.*>::matValue$";
%rename("$ignore", regextarget=1, fullname=1) "Arcane::Materials::CellMaterialVariableScalarRef<.*>::envValue$";

// Supprime ces méthodes car elles ne sont pas utiles pour le C#.
%ignore Arcane::Materials::MeshMaterialVariableIndexer::changeLocalIds;
%ignore Arcane::Materials::MeshMaterialVariableIndexer::endUpdate;
%ignore Arcane::Materials::MeshMaterialVariableIndexer::endUpdateAdd;
%ignore Arcane::Materials::MeshMaterialVariableIndexer::endUpdateRemove;
%ignore Arcane::Materials::MeshMaterialVariableIndexer::matvarIndexesArray;
%ignore Arcane::Materials::MeshMaterialVariableIndexer::transformCells;
%ignore Arcane::Materials::MeshMaterialVariableIndexer::TransformCellsArgs;
%ignore Arcane::Materials::IMeshMaterialVariable::dumpValues;
%ignore Arcane::Materials::IMeshMaterialMng::getReference;

%include arcane/core/materials/MaterialsCoreGlobal.h

#undef ARCANE_MATERIALS_EXPORT
#define ARCANE_MATERIALS_EXPORT

%include ComponentItemVector.i

ARCANE_STD_EXHANDLER
%include arcane/core/materials/IMeshBlock.h
%include arcane/core/materials/IMeshComponent.h
%include arcane/core/materials/IMeshMaterial.h
%include arcane/core/materials/IMeshEnvironment.h
%include arcane/core/materials/IMeshMaterialMng.h
%include arcane/core/materials/CellToAllEnvCellConverter.h
%include MeshMaterialVariable.i
%exception;

%template(IMeshMaterialMng_Ref) Arcane::Ref<Arcane::Materials::IMeshMaterialMng>;
