// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IArrayMeshMaterialVariable.h                               (C) 2000-2022 */
/*                                                                           */
/* Interface d'une variable matériau scalaire.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IARRAYMESHMATERIALVARIABLE_H
#define ARCANE_MATERIALS_IARRAYMESHMATERIALVARIABLE_H
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
 * \brief Interface d'accès pour CellMaterialVariableArrayRef.
 */
template<typename ItemType,typename DataType>
class IMeshMaterialVariableArray
{
 public:

  using ThatInterface = IMeshMaterialVariableArray<ItemType,DataType>;
  using ItemTypeType = ItemType;
  using DataTypeType = DataType;
  using BuilderType = MeshMaterialVariableBuildTraits<ThatInterface>;
  using VariableRefType = MeshVariableArrayRefT<ItemType,DataType>;
  static constexpr int dimension() { return 1; }

 public:

  virtual ~IMeshMaterialVariableArray() = default;

 public:

  virtual Array2View<DataType>* valuesView() =0;
  virtual VariableRefType* globalVariableReference() const =0;
  virtual void incrementReference() =0;
  virtual IMeshMaterialVariable* toMeshMaterialVariable() =0;

  virtual void resize(Int32 dim2_size) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
