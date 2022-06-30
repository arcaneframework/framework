﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariable.cc                                    (C) 2000-2022 */
/*                                                                           */
/* Interface des variables matériaux.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/IMeshMaterialVariable.h"
#include "arcane/core/materials/IScalarMeshMaterialVariable.h"
#include "arcane/core/materials/IArrayMeshMaterialVariable.h"
#include "arcane/core/materials/MaterialVariableTypeInfo.h"
#include "arcane/core/materials/MaterialVariableBuildInfo.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshMaterialVariableFactoryMng.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/VariableDataTypeTraits.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename TrueType> MaterialVariableTypeInfo
MeshMaterialVariableBuildTraits<TrueType>::
_buildVarTypeInfo(MatVarSpace space)
{
  using ItemType = typename TrueType::ItemTypeType;
  using DataType = typename TrueType::DataTypeType;
  int dim = TrueType::dimension();
  eItemKind ik = ItemTraitsT<ItemType>::kind();
  eDataType dt = VariableDataTypeTraitsT<DataType>::type();
  return MaterialVariableTypeInfo(ik,dt,dim,space);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename TrueType> TrueType*
MeshMaterialVariableBuildTraits<TrueType>::
getVariableReference(const MaterialVariableBuildInfo& v,MatVarSpace mvs)
{
  MaterialVariableTypeInfo x = _buildVarTypeInfo(mvs);

  MeshHandle mesh_handle = v.meshHandle();
  if (mesh_handle.isNull())
    ARCANE_FATAL("No mesh handle for material variable");

  IMeshMaterialMng* mat_mng = v.materialMng();

  // TODO: regarder si verrou necessaire
  if (!mat_mng)
    mat_mng = IMeshMaterialMng::getReference(mesh_handle,true);

  IMeshMaterialVariableFactoryMng* vm = mat_mng->variableFactoryMng();
  IMeshMaterialVariable* var = vm->createVariable(x.fullName(),v);

  auto* true_var = dynamic_cast<TrueType*>(var);
  ARCANE_CHECK_POINTER(true_var);
  return true_var;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_INSTANTIATE_MAT(type) \
  template class ARCANE_TEMPLATE_EXPORT MeshMaterialVariableBuildTraits<IMeshMaterialVariableScalar<Cell,type>>; \
  template class ARCANE_TEMPLATE_EXPORT MeshMaterialVariableBuildTraits<IMeshMaterialVariableArray<Cell,type>>

ARCANE_INSTANTIATE_MAT(Byte);
ARCANE_INSTANTIATE_MAT(Int16);
ARCANE_INSTANTIATE_MAT(Int32);
ARCANE_INSTANTIATE_MAT(Int64);
ARCANE_INSTANTIATE_MAT(Real);
ARCANE_INSTANTIATE_MAT(Real2);
ARCANE_INSTANTIATE_MAT(Real3);
ARCANE_INSTANTIATE_MAT(Real2x2);
ARCANE_INSTANTIATE_MAT(Real3x3);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
