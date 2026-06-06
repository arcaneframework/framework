// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartialVariableScalarRef.inst.h                         (C) 2000-2025 */
/*                                                                           */
/* Implementation of classes deriving from MeshPartialVariableScalarRef.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshPartialVariableScalarRef.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/VariableDataTypeTraits.h"
#include "arcane/core/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> VariableTypeInfo
ItemPartialVariableScalarRefT<DataType>::
_buildVariableTypeInfo(eItemKind ik)
{
  eDataType dt = VariableDataTypeTraitsT<DataType>::type();
  return VariableTypeInfo(ik, dt, 1, 0, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> VariableInfo
ItemPartialVariableScalarRefT<DataType>::
_buildVariableInfo(const VariableBuildInfo& vbi, eItemKind ik)
{
  VariableTypeInfo vti = _buildVariableTypeInfo(ik);
  DataStorageTypeInfo sti = vti._internalDefaultDataStorage();
  return VariableInfo(vbi.name(), vbi.itemFamilyName(), vbi.itemGroupName(), vbi.meshName(), vti, sti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
ItemPartialVariableScalarRefT<DataType>::
ItemPartialVariableScalarRefT(const VariableBuildInfo& vbi, eItemKind ik)
: PrivateVariableScalarT<DataType>(vbi, _buildVariableInfo(vbi, ik))
{
  this->_internalInit();
  internalSetUsed(this->isUsed());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
ItemPartialVariableScalarRefT<DataType>::
ItemPartialVariableScalarRefT(IVariable* var)
: PrivateVariableScalarT<DataType>(var)
{
  this->_internalInit();
  internalSetUsed(this->isUsed());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType>
ItemPartialVariableScalarRefT<DataType>::
ItemPartialVariableScalarRefT(const ItemPartialVariableScalarRefT<DataType>& rhs)
: PrivateVariableScalarT<DataType>(rhs)
{
  internalSetUsed(this->isUsed());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void
ItemPartialVariableScalarRefT<DataType>::
internalSetUsed(bool v)
{
  if (v)
    m_table = this->itemGroup().localIdToIndex();
  else
    m_table.reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType> void
ItemPartialVariableScalarRefT<DataType>::
operator=(const ItemPartialVariableScalarRefT<DataType>& rhs)
{
  PrivateVariableScalarT<DataType>::operator=(rhs);
  m_table = rhs.m_table;
  //this->updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType> void
ItemPartialVariableScalarRefT<DataType>::
fill(const DataType& v)
{
  this->m_private_part->fill(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void
ItemPartialVariableScalarRefT<DataType>::
copy(const ItemPartialVariableScalarRefT<DataType>& v)
{
  //TODO: utiliser memcpy()
  ENUMERATE_ITEM (iitem, this->itemGroup()) {
    operator[](iitem) = v[iitem];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType> VariableFactoryRegisterer
MeshPartialVariableScalarRefT<ItemType, DataType>::
m_auto_registerer(_autoCreate, _buildVariableTypeInfo());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType> VariableTypeInfo
MeshPartialVariableScalarRefT<ItemType, DataType>::
_buildVariableTypeInfo()
{
  eItemKind ik = ItemTraitsT<ItemType>::kind();
  eDataType dt = VariableDataTypeTraitsT<DataType>::type();
  return VariableTypeInfo(ik, dt, 1, 0, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
VariableRef* MeshPartialVariableScalarRefT<ItemType, DataType>::
_autoCreate(const VariableBuildInfo& vb)
{
  return new ThatClass(vb);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
MeshPartialVariableScalarRefT<ItemType, DataType>::
MeshPartialVariableScalarRefT(const VariableBuildInfo& vb)
: ItemPartialVariableScalarRefT<DataType>(vb, ItemTraitsT<ItemType>::kind())
{
  // Normally, this class should handle the initialization, but
  // since this class is just a wrapper around ItemVariableScalarRefT
  // and does nothing else, we leave the initialization to the base class,
  // which allows for generic variable creation on a mesh entity
  // based on its type.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType>
MeshPartialVariableScalarRefT<ItemType, DataType>::
MeshPartialVariableScalarRefT(const MeshPartialVariableScalarRefT<ItemType, DataType>& rhs)
: ItemPartialVariableScalarRefT<DataType>(rhs)
{
  // Normally, this class should handle the initialization, but
  // since this class is just a wrapper around ItemVariableScalarRefT
  // and does nothing else, we leave the initialization to the base class,
  // which allows for generic variable creation on a mesh entity
  // based on its type.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType> void
MeshPartialVariableScalarRefT<ItemType, DataType>::
refersTo(const MeshPartialVariableScalarRefT<ItemType, DataType>& rhs)
{
  ItemPartialVariableScalarRefT<DataType>::operator=(rhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ItemType, class DataType> auto
MeshPartialVariableScalarRefT<ItemType, DataType>::
itemGroup() const -> GroupType
{
  return GroupType(this->m_private_part->itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
