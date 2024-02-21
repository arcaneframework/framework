﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define MATERIALS_BEGIN_NAMESPACE  namespace Materials {
#define MATERIALS_END_NAMESPACE    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{
class ItemBase;
}

namespace Arcane::Materials::matimpl
{
using Arcane::impl::ItemBase;
class ConstituentItemBase;
}

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllEnvCellVectorView;
class ComponentCell;
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
class EnvAndGlobalCell {};
class MatAndGlobalCell {};
class IMeshMaterialMngInternal;
class MeshEnvironmentBuildInfo;
class MeshBlockBuildInfo;
class IMeshMaterialMngInternal;
class IMeshMaterialModifierImpl;
class CellToAllEnvCellConverter;
class IMeshMaterialVariableSynchronizer;
class AllCellToAllEnvCell;

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

template<typename DataType> class CellMaterialVariableScalarRef;

template<typename ItemType,typename DataType>
class IScalarMeshMaterialVariable;
template<typename ItemType,typename DataType>
class IArrayMeshMaterialVariable;

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
  Impure =1
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
