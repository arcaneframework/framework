// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartialVariableRefT.H                                   (C) 2000-2020 */
/*                                                                           */
/* Implémentation des classes dérivant de MeshParialVariableRef.             */
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

template<typename DataType> VariableTypeInfo
ItemPartialVariableScalarRefT<DataType>::
_buildVariableTypeInfo(eItemKind ik)
{
  eDataType dt = VariableDataTypeTraitsT<DataType>::type();
  return VariableTypeInfo(ik,dt,1,0,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableInfo
ItemPartialVariableScalarRefT<DataType>::
_buildVariableInfo(const VariableBuildInfo& vbi,eItemKind ik)
{
  VariableTypeInfo vti = _buildVariableTypeInfo(ik);
  DataStorageTypeInfo sti = vti._internalDefaultDataStorage();
  return VariableInfo(vbi.name(),vbi.itemFamilyName(),vbi.itemGroupName(),vbi.meshName(),vti,sti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
ItemPartialVariableScalarRefT<DataType>::
ItemPartialVariableScalarRefT(const VariableBuildInfo& vbi,eItemKind ik)
: PrivateVariableScalarT<DataType>(vbi,_buildVariableInfo(vbi,ik))
{
  this->_internalInit();
  internalSetUsed(this->isUsed());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
ItemPartialVariableScalarRefT<DataType>::
ItemPartialVariableScalarRefT(IVariable* var)
: PrivateVariableScalarT<DataType>(var)
{
  this->_internalInit();
  internalSetUsed(this->isUsed());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
ItemPartialVariableScalarRefT<DataType>::
ItemPartialVariableScalarRefT(const ItemPartialVariableScalarRefT<DataType>& rhs)
: PrivateVariableScalarT<DataType>(rhs)
{
  internalSetUsed(this->isUsed());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
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

template<class DataType> void
ItemPartialVariableScalarRefT<DataType>::
operator=(const ItemPartialVariableScalarRefT<DataType>& rhs)
{
  PrivateVariableScalarT<DataType>::operator=(rhs);
  m_table = rhs.m_table;
  //this->updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> void
ItemPartialVariableScalarRefT<DataType>::
fill(const DataType& v)
{
  this->m_private_part->fill(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemPartialVariableScalarRefT<DataType>::
copy(const ItemPartialVariableScalarRefT<DataType>& v)
{
  //TODO: utiliser memcpy()
  ENUMERATE_ITEM(iitem,this->itemGroup()){
    operator[](iitem) = v[iitem];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> VariableFactoryRegisterer
MeshPartialVariableScalarRefT<ItemType,DataType>::
m_auto_registerer(_autoCreate,_buildVariableTypeInfo());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType> VariableTypeInfo
MeshPartialVariableScalarRefT<ItemType,DataType>::
_buildVariableTypeInfo()
{
  eItemKind ik = ItemTraitsT<ItemType>::kind();
  eDataType dt = VariableDataTypeTraitsT<DataType>::type();
  return VariableTypeInfo(ik,dt,1,0,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> 
VariableRef* MeshPartialVariableScalarRefT<ItemType,DataType>::
_autoCreate(const VariableBuildInfo& vb)
{
  return new ThatClass(vb);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> 
MeshPartialVariableScalarRefT<ItemType,DataType>::
MeshPartialVariableScalarRefT(const VariableBuildInfo& vb)
: ItemPartialVariableScalarRefT<DataType>(vb,ItemTraitsT<ItemType>::kind())
{
  // Normalement, c'est à cette classe de faire l'initialisation mais
  // comme cette classe est juste un wrapper autour de ItemVariableScalarRefT
  // et ne fait rien d'autre, on laisse l'initialisation à la classe de base,
  // ce qui permet de fabriquer de manière générique une variable sur
  // une entité du maillage à partir de son genre.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> 
MeshPartialVariableScalarRefT<ItemType,DataType>::
MeshPartialVariableScalarRefT(const MeshPartialVariableScalarRefT<ItemType,DataType>& rhs)
: ItemPartialVariableScalarRefT<DataType>(rhs)
{
  // Normalement, c'est à cette classe de faire l'initialisation mais
  // comme cette classe est juste un wrapper autour de ItemVariableScalarRefT
  // et ne fait rien d'autre, on laisse l'initialisation à la classe de base,
  // ce qui permet de fabriquer de manière générique une variable sur
  // une entité du maillage à partir de son genre.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> void
MeshPartialVariableScalarRefT<ItemType,DataType>::
refersTo(const MeshPartialVariableScalarRefT<ItemType,DataType>& rhs)
{
  ItemPartialVariableScalarRefT<DataType>::operator=(rhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> auto
MeshPartialVariableScalarRefT<ItemType,DataType>::
itemGroup() const -> GroupType
{
  return GroupType(this->m_private_part->itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
