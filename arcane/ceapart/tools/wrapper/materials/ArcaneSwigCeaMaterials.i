%module(directors="1") ArcaneCeaMaterials

%import core/ArcaneSwigCore.i

%{
#include "ArcaneSwigUtils.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshEnvironment.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MatItem.h"
#include "arcane/materials/ComponentPartItemVectorView.h"
#include "arcane/materials/ComponentItemVectorView.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MeshMaterialVariableRef.h"

using namespace Arcane;
using namespace Arcane::Materials;
%}

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
%ignore Arcane::Materials::IMeshMaterialVariable::dumpValues;
%ignore Arcane::Materials::IMeshMaterialMng::getReference;

%include arcane/core/materials/MaterialsCoreGlobal.h
%include arcane/materials/MaterialsGlobal.h
#undef ARCANE_MATERIALS_EXPORT
#define ARCANE_MATERIALS_EXPORT

%include ComponentItemVector.i

ARCANE_STD_EXHANDLER
%include arcane/core/materials/MeshMaterialVariableIndexer.h
%include arcane/core/materials/IMeshBlock.h
%include arcane/core/materials/IMeshComponent.h
%include arcane/core/materials/IMeshMaterial.h
%include arcane/core/materials/IMeshEnvironment.h
%include arcane/core/materials/IMeshMaterialMng.h
%include arcane/core/materials/CellToAllEnvCellConverter.h
%include MeshMaterialVariable.i
%exception;

//%include arcane/materials/IMeshMaterialMng.h

%template(IMeshMaterialMng_Ref) Arcane::Ref<Arcane::Materials::IMeshMaterialMng>;
