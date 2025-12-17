// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartialVariableArrayRef.inst.h                          (C) 2000-2025 */
/*                                                                           */
/* Implémentation des classes dérivant de MeshPartialVariableArrayRef.       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshPartialVariableArrayRef.h"
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

template<typename DataType> VariableTypeInfo
ItemPartialVariableArrayRefT<DataType>::
_buildVariableTypeInfo(eItemKind ik)
{
  eDataType dt = VariableDataTypeTraitsT<DataType>::type();
  return VariableTypeInfo(ik,dt,2,0,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableInfo
ItemPartialVariableArrayRefT<DataType>::
_buildVariableInfo(const VariableBuildInfo& vbi,eItemKind ik)
{
  VariableTypeInfo vti = _buildVariableTypeInfo(ik);
  DataStorageTypeInfo sti = vti._internalDefaultDataStorage();
  return VariableInfo(vbi.name(),vbi.itemFamilyName(),vbi.itemGroupName(),vbi.meshName(),vti,sti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
ItemPartialVariableArrayRefT<DataType>::
ItemPartialVariableArrayRefT(const VariableBuildInfo& vb,eItemKind ik)
: PrivateVariableArrayT<DataType>(vb,_buildVariableInfo(vb,ik))
{
  this->_internalInit();
  internalSetUsed(this->isUsed());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
ItemPartialVariableArrayRefT<DataType>::
ItemPartialVariableArrayRefT(IVariable* var)
: PrivateVariableArrayT<DataType>(var)
{
  this->_internalInit();
  internalSetUsed(this->isUsed());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
ItemPartialVariableArrayRefT<DataType>::
ItemPartialVariableArrayRefT(const ItemPartialVariableArrayRefT<DataType>& rhs)
: PrivateVariableArrayT<DataType>(rhs)
{
  internalSetUsed(this->isUsed());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemPartialVariableArrayRefT<DataType>::
internalSetUsed(bool v)
{
  if (v)
    m_table = this->itemGroup().localIdToIndex();
  else
    m_table.reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void 
ItemPartialVariableArrayRefT<DataType>::
operator=(const ItemPartialVariableArrayRefT<DataType>& rhs)
{
  PrivateVariableArrayT<DataType>::operator=(rhs);
  m_table = rhs.m_table;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void 
ItemPartialVariableArrayRefT<DataType>::
fill(const DataType& v)
{
  ENUMERATE_ITEM(iitem,this->itemGroup()){
    operator[](iitem).fill(v);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void 
ItemPartialVariableArrayRefT<DataType>::
copy(const ItemPartialVariableArrayRefT<DataType>& v)
{
  ENUMERATE_ITEM(iitem,this->itemGroup()){
    operator[](iitem).copy(v[iitem]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> VariableFactoryRegisterer
MeshPartialVariableArrayRefT<ItemType,DataType>::
m_auto_registerer(_autoCreate,_buildVariableTypeInfo());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType> VariableTypeInfo
MeshPartialVariableArrayRefT<ItemType,DataType>::
_buildVariableTypeInfo()
{
  eItemKind ik = ItemTraitsT<ItemType>::kind();
  return BaseClass::_buildVariableTypeInfo(ik);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType> VariableInfo
MeshPartialVariableArrayRefT<ItemType,DataType>::
_buildVariableInfo(const VariableBuildInfo& vbi)
{
  eItemKind ik = ItemTraitsT<ItemType>::kind();
  return BaseClass::_buildVariableInfo(vbi,ik);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> VariableRef*
MeshPartialVariableArrayRefT<ItemType,DataType>::
_autoCreate(const VariableBuildInfo& vb)
{
  return new ThatClass(vb);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> 
MeshPartialVariableArrayRefT<ItemType,DataType>::
MeshPartialVariableArrayRefT(const VariableBuildInfo& vb)
: ItemPartialVariableArrayRefT<DataType>(vb,ItemTraitsT<ItemType>::kind())
{
  // Normalement, c'est à cette classe de faire l'initilisation mais
  // comme cette classe est juste un wrapper autour de ItemVariableArrayRefT
  // et ne fait rien d'autre, on laisse l'initialisation à la classe de base,
  // ce qui permet de fabriquer de manière générique une variable sur
  // une entité du maillage à partir de son genre.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> 
MeshPartialVariableArrayRefT<ItemType,DataType>::
MeshPartialVariableArrayRefT(const MeshPartialVariableArrayRefT<ItemType,DataType>& rhs)
: ItemPartialVariableArrayRefT<DataType>(rhs)
{
  // Normalement, c'est à cette classe de faire l'initilisation mais
  // comme cette classe est juste un wrapper autour de ItemVariableArrayRefT
  // et ne fait rien d'autre, on laisse l'initialisation à la classe de base,
  // ce qui permet de fabriquer de manière générique une variable sur
  // une entité du maillage à partir de son genre.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> void
MeshPartialVariableArrayRefT<ItemType,DataType>::
refersTo(const MeshPartialVariableArrayRefT<ItemType,DataType>& rhs)
{
  ItemPartialVariableArrayRefT<DataType>::operator=(rhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType>
typename Arcane::MeshPartialVariableArrayRefT<ItemType, DataType>::GroupType
MeshPartialVariableArrayRefT<ItemType,DataType>::
itemGroup() const
{
  return GroupType(this->m_private_part->itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
