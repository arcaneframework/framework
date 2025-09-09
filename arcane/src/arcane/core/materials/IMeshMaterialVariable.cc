// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariable.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Interface des variables matériaux.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/IMeshMaterialVariable.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/materials/IScalarMeshMaterialVariable.h"
#include "arcane/core/materials/IArrayMeshMaterialVariable.h"
#include "arcane/core/materials/MaterialVariableTypeInfo.h"
#include "arcane/core/materials/MaterialVariableBuildInfo.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshMaterialVariableFactoryMng.h"

#include "arcane/core/VariableDataTypeTraits.h"
#include "arcane/core/IVariable.h"

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
/*!
 * \brief Retourne une variable associée à un constituant.
 *
 * Retourne la variable constituant à partir des informations données par
 * \a v \et \a mvs. Si la variable n'existe pas encore, elle est créée.
 */
template <typename TrueType> TrueType*
MeshMaterialVariableBuildTraits<TrueType>::
getVariableReference(const MaterialVariableBuildInfo& v,MatVarSpace mvs)
{
  MaterialVariableTypeInfo x = _buildVarTypeInfo(mvs);

  MeshHandle mesh_handle = v.meshHandle();
  if (mesh_handle.isNull())
    ARCANE_FATAL("No mesh handle for material variable");

  // Si le gestionnaire de matériaux n'existe pas encore, on le créé.
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
/*!
 * \brief Retourne le type concret d'une variable constituant.
 *
 * Converti \a var en le type \a TrueType. Si ce n'est pas possible, lève
 * une exception.
 */
template <typename TrueType> TrueType*
MeshMaterialVariableBuildTraits<TrueType>::
getVariableReference(IMeshMaterialVariable* var)
{
  ARCANE_CHECK_POINTER(var);
  auto* true_var = dynamic_cast<TrueType*>(var);
  if (!true_var)
    ARCANE_FATAL("Can not convert variable '{0}' in the template type of this class", var->globalVariable()->fullName());
  return true_var;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_INSTANTIATE_MAT(type) \
  template class ARCANE_TEMPLATE_EXPORT MeshMaterialVariableBuildTraits<IScalarMeshMaterialVariable<Cell,type>>; \
  template class ARCANE_TEMPLATE_EXPORT MeshMaterialVariableBuildTraits<IArrayMeshMaterialVariable<Cell,type>>

ARCANE_INSTANTIATE_MAT(Byte);
ARCANE_INSTANTIATE_MAT(Int8);
ARCANE_INSTANTIATE_MAT(Int16);
ARCANE_INSTANTIATE_MAT(Int32);
ARCANE_INSTANTIATE_MAT(Int64);
ARCANE_INSTANTIATE_MAT(BFloat16);
ARCANE_INSTANTIATE_MAT(Float16);
ARCANE_INSTANTIATE_MAT(Float32);
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
