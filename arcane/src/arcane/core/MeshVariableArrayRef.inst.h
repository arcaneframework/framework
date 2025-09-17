// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableArrayRef.inst.h                                 (C) 2000-2025 */
/*                                                                           */
/* Variable vectorielle du maillage.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshVariableArrayRef.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/VariableDataTypeTraits.h"
#include "arcane/core/VariableTypeInfo.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/VariableInfo.h"
#include "arcane/core/VariableFactoryRegisterer.h"
#include "arcane/core/internal/IDataInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableTypeInfo
ItemVariableArrayRefT<DataType>::
_internalVariableTypeInfo(eItemKind ik)
{
  eDataType dt = VariableDataTypeTraitsT<DataType>::type();
  return VariableTypeInfo(ik,dt,2,0,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableInfo
ItemVariableArrayRefT<DataType>::
_internalVariableInfo(const VariableBuildInfo& vbi,eItemKind ik)
{
  VariableTypeInfo vti = _internalVariableTypeInfo(ik);
  DataStorageTypeInfo sti = vti._internalDefaultDataStorage();
  return VariableInfo(vbi.name(),vbi.itemFamilyName(),vbi.itemGroupName(),vbi.meshName(),vti,sti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
ItemVariableArrayRefT<DataType>::
ItemVariableArrayRefT(const VariableBuildInfo& vb,eItemKind ik)
: PrivateVariableArrayT<DataType>(vb,_internalVariableInfo(vb,ik))
{
  if (!vb.isNull()) {
    this->_internalInit();
    if (this->m_private_part->isPartial())
      ARCANE_FATAL("Can not assign a partial variable to a full variable");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> ItemVariableArrayRefT<DataType>::
ItemVariableArrayRefT(IVariable* var)
: PrivateVariableArrayT<DataType>(var)
{
  this->_internalInit();
  if (this->m_private_part->isPartial())
    ARCANE_FATAL("Can not assign a partial variable to a full variable");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> ItemVariableArrayRefT<DataType>::
ItemVariableArrayRefT(const ItemVariableArrayRefT<DataType>& rhs)
: PrivateVariableArrayT<DataType>(rhs)
{
  if (this->m_private_part)
    this->updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class DataType> ItemVariableArrayRefT<DataType>&
ItemVariableArrayRefT<DataType>::
operator=(const ItemVariableArrayRefT<DataType>& rhs)
{
  if (this != &rhs) {
    BaseClass::operator=(rhs);
    if (this->m_private_part) {
      this->m_private_part = rhs.m_private_part;
      this->updateFromInternal();
    }
  }
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> VariableFactoryRegisterer
MeshVariableArrayRefT<ItemType,DataType>::
m_auto_registerer(_autoCreate,_internalVariableTypeInfo());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType> VariableTypeInfo
MeshVariableArrayRefT<ItemType,DataType>::
_internalVariableTypeInfo()
{
  eItemKind ik = ItemTraitsT<ItemType>::kind();
  eDataType dt = VariableDataTypeTraitsT<DataType>::type();
  return VariableTypeInfo(ik,dt,2,0,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType> VariableInfo
MeshVariableArrayRefT<ItemType,DataType>::
_internalVariableInfo(const VariableBuildInfo& vbi)
{
  VariableTypeInfo vti = _internalVariableTypeInfo();
  DataStorageTypeInfo sti = vti._internalDefaultDataStorage();
  return VariableInfo(vbi.name(),vbi.itemFamilyName(),vbi.itemGroupName(),vbi.meshName(),vti,sti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> VariableRef*
MeshVariableArrayRefT<ItemType,DataType>::
_autoCreate(const VariableBuildInfo& vb)

{
  return new ThatClass(vb);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType>
MeshVariableArrayRefT<ItemType,DataType>::
MeshVariableArrayRefT(const VariableBuildInfo& vb)
: ItemVariableArrayRefT<DataType>(vb,ItemTraitsT<ItemType>::kind())
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
MeshVariableArrayRefT<ItemType,DataType>::
MeshVariableArrayRefT(IVariable* var)
: ItemVariableArrayRefT<DataType>(var)
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
MeshVariableArrayRefT<ItemType,DataType>::
MeshVariableArrayRefT(const MeshVariableArrayRefT<ItemType,DataType>& rhs)
: ItemVariableArrayRefT<DataType>(rhs)
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
void
MeshVariableArrayRefT<ItemType,DataType>::
refersTo(const MeshVariableArrayRefT<ItemType,DataType>& rhs)
{
  ItemVariableArrayRefT<DataType>::operator=(rhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType>
typename Arcane::MeshVariableArrayRefT<ItemType, DataType>::GroupType
MeshVariableArrayRefT<ItemType, DataType>::
itemGroup() const
{
  return GroupType(this->m_private_part->itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType>
void ItemVariableArrayRefT<DataType>::
fill(const DataType& v)
{
  this->fill(v,this->itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType> 
void ItemVariableArrayRefT<DataType>::
fill(const DataType& value,const ItemGroup& group)
{
  ENUMERATE_ITEM(iitem,group){
    operator[](iitem).fill(value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> 
void ItemVariableArrayRefT<DataType>::
copy(const ItemVariableArrayRefT<DataType>& v)
{
  this->copy(v,this->itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> 
void ItemVariableArrayRefT<DataType>::
copy(const ItemVariableArrayRefT<DataType>& v,const ItemGroup& group)
{
  ENUMERATE_ITEM(iitem,group){
    operator[](iitem).copy(v[iitem]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataTypeT> void
ItemVariableArrayRefT<DataTypeT>::
copy(const ItemVariableArrayRefT<DataTypeT>& v,RunQueue* queue)
{
  if (!queue){
    copy(v);
    return;
  }
  impl::copyContiguousData(this->m_private_part->data(), v.m_private_part->data(), *queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataTypeT> void
ItemVariableArrayRefT<DataTypeT>::
fill(const DataTypeT& v,RunQueue* queue)
{
  if (!queue){
    fill(v);
    return;
  }
  impl::fillContiguousData(this->m_private_part->data(), v, *queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Échange les valeurs de la variable \a rhs avec celles de l'instance.
 *
 * Cette méthode est optimisée pour éviter les recopie et donc l'échange
 * se fait en temps constant. Les variables échangées doivent avoir le
 * même maillage, la même famille et le même groupe. Elles doivent aussi
 * être allouées (IVariable::setUsed()==true)
 */
template<class ItemType,class DataTypeT> void
MeshVariableArrayRefT<ItemType,DataTypeT>::
swapValues(MeshVariableArrayRefT<ItemType,DataType>& rhs)
{
  this->m_private_part->swapValues(*(rhs.m_private_part));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
