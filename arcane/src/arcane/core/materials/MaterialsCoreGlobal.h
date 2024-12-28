// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialsCoreGlobal.h                                       (C) 2000-2024 */
/*                                                                           */
/* Déclarations générales des matériaux de Arcane.                           */
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using ComponentItemInternalPtr = ComponentItemInternal*;
using IMeshComponentPtr = IMeshComponent*;
using IMeshMaterialPtr = IMeshMaterial*;
using IMeshEnvironmentPtr = IMeshEnvironment*;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste de composants multi-matériaux du maillage.
using MeshComponentListView = ConstArrayView<IMeshComponent*>;

//! Liste de milieux du maillage.
using MeshEnvironmentListView = ConstArrayView<IMeshEnvironment*>;

//! Liste de matériaux du maillage.
using MeshMaterialListView = ConstArrayView<IMeshMaterial*>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Liste de ComponentCell issues d'un IMeshComponent.
using ComponentCellVector = ComponentItemVector;

//! Type de la vue sur un EnvCellVector
using EnvCellVectorView = EnvItemVectorView;

//! Type de la vue sur un MatCellVector
using MatCellVectorView = MatItemVectorView;

//! Type de la vue sur un ComponentCellVector
using ComponentCellVectorView = ComponentItemVectorView;

//! Liste de composants multi-matériaux du maillage.
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

//! Enumérateur sur les mailles matériaux d'une maille.
using CellMatCellEnumerator = CellComponentCellEnumeratorT<MatCell>;

//! Enumérateur sur les mailles milieux d'une maille.
using CellEnvCellEnumerator = CellComponentCellEnumeratorT<EnvCell>;

//! Index d'un MatItem dans une variable.
using MatCellLocalId = MatItemLocalId;

//! Index d'un EnvItem dans une variable.
using EnvCellLocalId = EnvItemLocalId;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static const Int32 LEVEL_MATERIAL = 1;
static const Int32 LEVEL_ENVIRONMENT = 2;
static const Int32 LEVEL_ALLENVIRONMENT = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Espace de définition d'une variable matériau.
 */
enum class MatVarSpace
{
  //! Variable ayant des valeurs sur les milieux et matériaux
  MaterialAndEnvironment = 1,
  //! Variable ayant des valeurs uniquement sur les milieux
  Environment
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Partie d'un composant
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
