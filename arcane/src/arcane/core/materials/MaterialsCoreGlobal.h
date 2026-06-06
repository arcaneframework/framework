// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialsCoreGlobal.h                                       (C) 2000-2026 */
/*                                                                           */
/* General declarations for Arcane materials.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_MATERIALSCOREGLOBAL_H
#define ARCANE_CORE_MATERIALS_MATERIALSCOREGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arccore/base/RefDeclarations.h"
#include "arccore/collections/ArrayTraits.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define MATERIALS_BEGIN_NAMESPACE \
  namespace Materials \
  {
#define MATERIALS_END_NAMESPACE }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemBase;
}
namespace Arcane::Materials::matimpl
{
using ::Arcane::ItemBase;
class ConstituentItemBase;
}

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllCellToAllEnvCellContainer;
class AllEnvCellVectorView;
class ComponentItemInternalData;
class ConstituentItem;
using ComponentCell = ConstituentItem;
using ConstituentCell = ConstituentItem;
class ConstituentItemVectorImpl;
class ConstituentItemLocalIdList;
class IMeshBlock;
class IMeshComponent;
class IMeshMaterial;
class IMeshMaterialMng;
class IMeshEnvironment;
class MatVarIndex;
class MeshMaterialVariableIndexer;
class MeshMaterialInfo;
class MatCell;
class EnvCell;
class MatItemVectorView;
class EnvItemVectorView;
class MatPurePartItemVectorView;
class MatImpurePartItemVectorView;
class MatPartItemVectorView;
class EnvPurePartItemVectorView;
class EnvImpurePartItemVectorView;
class EnvPartItemVectorView;
class ComponentItemInternal;
class AllEnvCell;
class ComponentItemVectorView;
class ComponentPartItemVectorView;
class ComponentPurePartItemVectorView;
class ComponentImpurePartItemVectorView;
class ConstituentItemLocalIdListView;
class MeshComponentPartData;
class MatCellEnumerator;
class ComponentItemVector;
class IMeshMaterialVariableFactoryMng;
class IMeshMaterialVariableFactory;
class IMeshMaterialVariable;
class MaterialVariableBuildInfo;
class MaterialVariableTypeInfo;
class MeshMaterialVariableRef;
class ConstituentItemVectorBuildInfo;
class EnvAndGlobalCell
{};
class MatAndGlobalCell
{};
class IMeshMaterialMngInternal;
class MeshEnvironmentBuildInfo;
class MeshBlockBuildInfo;
class IMeshMaterialMngInternal;
class MeshMaterialModifierImpl;
class CellToAllEnvCellConverter;
class IMeshMaterialVariableSynchronizer;
class AllCellToAllEnvCell;
class ConstituentItemIndex;
class IConstituentItemVectorImpl;
class ConstituentItemSharedInfo;
using ComponentItemSharedInfo = ConstituentItemSharedInfo;
class ConstituentItemLocalId;
class MatItemLocalId;
class EnvItemLocalId;
using ComponentItemLocalId = ConstituentItemLocalId;

class AllEnvData;
class MeshMaterialMng;
class MeshEnvironment;
class MeshComponentData;
class EnvCellVector;
class MatCellVector;

template <typename ContainerView_>
class ConstituentItemIndexedSelectionView;
template <typename ContainerView_>
class ConstituentItemIndexedSelectionEnumerator;
template <typename ConstituenItemType_>
class EnumeratorBuilder;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using ComponentItemInternalPtr = ComponentItemInternal*;
using IMeshComponentPtr = IMeshComponent*;
using IMeshMaterialPtr = IMeshMaterial*;
using IMeshEnvironmentPtr = IMeshEnvironment*;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! List of multi-material components of the mesh.
using MeshComponentListView = ConstArrayView<IMeshComponent*>;

//! List of mesh environments.
using MeshEnvironmentListView = ConstArrayView<IMeshEnvironment*>;

//! List of mesh materials.
using MeshMaterialListView = ConstArrayView<IMeshMaterial*>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Vector of ComponentCells originating from an IMeshComponent.
using ComponentCellVector = ComponentItemVector;

//! View type for an EnvCellVector
using EnvCellVectorView = EnvItemVectorView;

//! View type for a MatCellVector
using MatCellVectorView = MatItemVectorView;

//! View type for a ComponentCellVector
using ComponentCellVectorView = ComponentItemVectorView;

//! List of multi-material components of the mesh.
using MeshComponentList = ConstArrayView<IMeshComponent*>;

template <typename DataType> class CellMaterialVariableScalarRef;

template <typename ItemType, typename DataType>
class IScalarMeshMaterialVariable;
template <typename ItemType, typename DataType>
class IArrayMeshMaterialVariable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CellComponentCellEnumerator;
template <typename ConstituentCellType> class CellComponentCellEnumeratorT;

//! Enumerator over material meshes of a mesh.
using CellMatCellEnumerator = CellComponentCellEnumeratorT<MatCell>;

//! Enumerator over environment meshes of a mesh.
using CellEnvCellEnumerator = CellComponentCellEnumeratorT<EnvCell>;

//! Index of a MatItem in a variable.
using MatCellLocalId = MatItemLocalId;

//! Index of an EnvItem in a variable.
using EnvCellLocalId = EnvItemLocalId;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if 0
//! Selection over an 'EvnCellVectorView'
using EnvCellVectorSelectionView = ConstituentItemIndexedSelectionView<EnvCellVectorView>;
//! Selection over a 'MatCellVectorView'
using MatCellVectorSelectionView = ConstituentItemIndexedSelectionView<MatCellVectorView>;
//! Selection over a 'ComponentCellVectorView'
using ComponentCellVectorSelectionView = ConstituentItemIndexedSelectionView<ComponentCellVectorView>;
#endif
class EnvCellVectorSelectionView;
class MatCellVectorSelectionView;
class ComponentCellVectorSelectionView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static const Int32 LEVEL_MATERIAL = 1;
static const Int32 LEVEL_ENVIRONMENT = 2;
static const Int32 LEVEL_ALLENVIRONMENT = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Definition space for a material variable.
 */
enum class MatVarSpace
{
  //! Variable having values on environments and materials
  MaterialAndEnvironment = 1,
  //! Variable having values only on environments
  Environment
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Part of a component
enum class eMatPart
{
  Pure = 0,
  Impure = 1
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_ARRAY_PODTYPE(Arcane::Materials::MatVarIndex);
ARCCORE_DEFINE_ARRAY_PODTYPE(Arcane::Materials::ConstituentItemIndex);
} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::Materials::IConstituentItemVectorImpl);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
