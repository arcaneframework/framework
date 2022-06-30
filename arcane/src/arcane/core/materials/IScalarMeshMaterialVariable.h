// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IScalarMeshMaterialVariable.h                               (C) 2000-2022 */
/*                                                                           */
/* Interface d'une variable matériau scalaire.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_ISCALARMESHMATERIALVARIABLE_H
#define ARCANE_MATERIALS_ISCALARMESHMATERIALVARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Array.h"

#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'accès pour CellMaterialVariableScalarRef.
 */
template<typename ItemType,typename DataType>
class IMeshMaterialVariableScalar
{
 public:

  using ThatInterface = IMeshMaterialVariableScalar<ItemType,DataType>;
  using ItemTypeType = ItemType;
  using DataTypeType = DataType;
  using VariableRefType = MeshVariableScalarRefT<ItemType,DataType>;
  using BuilderType = MeshMaterialVariableBuildTraits<ThatInterface>;
  static constexpr int dimension() { return 0; }

 public:

  virtual ~IMeshMaterialVariableScalar() = default;

 public:

  virtual ArrayView<DataType>* valuesView() = 0;
  virtual ArrayView<ArrayView<DataType>> _internalFullValuesView() = 0;
  virtual void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values) =0;
  virtual void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values,Int32ConstArrayView indexes) =0;
  virtual void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values) =0;
  virtual void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values,Int32ConstArrayView indexes) =0;
  virtual void fillPartialValues(const DataType& value) =0;
  virtual VariableRefType* globalVariableReference() const =0;
  virtual void incrementReference() =0;
  virtual IMeshMaterialVariable* toMeshMaterialVariable() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
