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
#include "arcane/core/materials/internal/IMeshComponentInternal.h"

using namespace Arcane;
using namespace Arcane::Materials;
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
%include arcane/core/materials/MeshMaterialVariableIndexer.h
%include arcane/core/materials/IMeshBlock.h
%include arcane/core/materials/IMeshComponent.h
%include arcane/core/materials/IMeshMaterial.h
%include arcane/core/materials/IMeshEnvironment.h
%include arcane/core/materials/IMeshMaterialMng.h
%include arcane/core/materials/CellToAllEnvCellConverter.h
%include arcane/core/materials/internal/IMeshComponentInternal.h
%include MeshMaterialVariable.i
%exception;

%template(IMeshMaterialMng_Ref) Arcane::Ref<Arcane::Materials::IMeshMaterialMng>;
