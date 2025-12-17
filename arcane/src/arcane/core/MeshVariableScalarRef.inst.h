// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableScalarRef.inst.h                                (C) 2000-2025 */
/*                                                                           */
/* Variable scalaire du maillage.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshVariableScalarRef.h"

#include "arcane/core/ItemGroup.h"
#include "arcane/core/VariableScalar.h"
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
ItemVariableScalarRefT<DataType>::
_internalVariableTypeInfo(eItemKind ik)
{
  eDataType dt = VariableDataTypeTraitsT<DataType>::type();
  return VariableTypeInfo(ik,dt,1,0,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> VariableInfo
ItemVariableScalarRefT<DataType>::
_internalVariableInfo(const VariableBuildInfo& vbi,eItemKind ik)
{
  VariableTypeInfo vti = _internalVariableTypeInfo(ik);
  DataStorageTypeInfo sti = vti._internalDefaultDataStorage();
  return VariableInfo(vbi.name(),vbi.itemFamilyName(),vbi.itemGroupName(),vbi.meshName(),vti,sti);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataTypeT> 
ItemVariableScalarRefT<DataTypeT>::
ItemVariableScalarRefT(const VariableBuildInfo& vb,eItemKind ik)
: PrivateVariableScalarT<DataTypeT>(vb,_internalVariableInfo(vb,ik))
{
  if (!vb.isNull()) {
    this->_internalInit();
    if (this->m_private_part->isPartial())
      ARCANE_FATAL("Can not assign a partial variable to a full variable");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataTypeT> 
ItemVariableScalarRefT<DataTypeT>::
ItemVariableScalarRefT(IVariable* var)
: PrivateVariableScalarT<DataTypeT>(var)
{
  this->_internalInit();
  if (this->m_private_part->isPartial())
    ARCANE_FATAL("Can not assign a partial variable to a full variable");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataTypeT> 
ItemVariableScalarRefT<DataTypeT>::
ItemVariableScalarRefT(const ItemVariableScalarRefT<DataTypeT>& rhs)
: PrivateVariableScalarT<DataTypeT>(rhs)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataTypeT> void 
ItemVariableScalarRefT<DataTypeT>::
fill(const DataTypeT& v)
{
  this->m_private_part->fill(v,this->m_private_part->itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataTypeT> void 
ItemVariableScalarRefT<DataTypeT>::
fill(const DataTypeT& value,const ItemGroup& group)
{
  ENUMERATE_ITEM(iitem,group){
    operator[](iitem) = value;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataTypeT> void 
ItemVariableScalarRefT<DataTypeT>::
copy(const ItemVariableScalarRefT<DataTypeT>& v,const ItemGroup& group)
{
  ENUMERATE_ITEM(iitem,group){
    operator[](iitem) = v[iitem];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataTypeT> void
ItemVariableScalarRefT<DataTypeT>::
copy(const ItemVariableScalarRefT<DataTypeT>& v,RunQueue* queue)
{
  if (!queue){
    copy(v);
    return;
  }
  impl::copyContiguousData(this->m_private_part->data(), v.m_private_part->data(), *queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataTypeT> void
ItemVariableScalarRefT<DataTypeT>::
fill(const DataTypeT& v, RunQueue* queue)
{
  if (!queue){
    fill(v);
    return;
  }
  impl::fillContiguousData<DataTypeT>(this->m_private_part->data(), v, *queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataTypeT> void 
ItemVariableScalarRefT<DataTypeT>::
operator=(const ItemVariableScalarRefT<DataTypeT>& rhs)
{
  BaseClass::operator=(rhs);
  this->m_private_part = rhs.m_private_part;
  this->updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> VariableFactoryRegisterer
MeshVariableScalarRefT<ItemType,DataType>::
m_auto_registerer(_autoCreate,_internalVariableTypeInfo());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType> VariableInfo
MeshVariableScalarRefT<ItemType,DataType>::
_internalVariableInfo(const VariableBuildInfo& vbi)
{
  return BaseClass::_internalVariableInfo(vbi,ItemTraitsT<ItemType>::kind());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType> VariableTypeInfo
MeshVariableScalarRefT<ItemType,DataType>::
_internalVariableTypeInfo()
{
  return BaseClass::_internalVariableTypeInfo(ItemTraitsT<ItemType>::kind());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataType> 
VariableRef* MeshVariableScalarRefT<ItemType,DataType>::
_autoCreate(const VariableBuildInfo& vb)
{
  return new ThatClass(vb);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataTypeT> 
MeshVariableScalarRefT<ItemType,DataTypeT>::
MeshVariableScalarRefT(const VariableBuildInfo& vb)
: ItemVariableScalarRefT<DataTypeT>(vb,ItemTraitsT<ItemType>::kind())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataTypeT> 
MeshVariableScalarRefT<ItemType,DataTypeT>::
MeshVariableScalarRefT(IVariable* var)
: ItemVariableScalarRefT<DataTypeT>(var)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataTypeT> 
MeshVariableScalarRefT<ItemType,DataTypeT>::
MeshVariableScalarRefT(const MeshVariableScalarRefT<ItemType,DataTypeT>& rhs)
: ItemVariableScalarRefT<DataTypeT>(rhs)
{
  // Normalement, c'est à cette classe de faire l'initilisation mais
  // comme cette classe est juste un wrapper autour de ItemVariableScalarRefT
  // et ne fait rien d'autre, on laisse l'initialisation à la classe de base,
  // ce qui permet de fabriquer de manière générique une variable sur
  // une entité du maillage à partir de son genre.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataTypeT> void
MeshVariableScalarRefT<ItemType,DataTypeT>::
refersTo(const MeshVariableScalarRefT<ItemType,DataTypeT>& rhs)
{
  BaseClass::operator=(rhs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataTypeT>
typename Arcane::MeshVariableScalarRefT<ItemType, DataTypeT>::GroupType
MeshVariableScalarRefT<ItemType, DataTypeT>::
itemGroup() const
{
  return GroupType(BaseClass::itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataTypeT> void
MeshVariableScalarRefT<ItemType,DataTypeT>::
setIsSynchronized(const GroupType& group)
{
  this->m_private_part->setIsSynchronized(group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class ItemType,class DataTypeT> void
MeshVariableScalarRefT<ItemType,DataTypeT>::
setIsSynchronized()
{
  this->m_private_part->setIsSynchronized();
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
MeshVariableScalarRefT<ItemType,DataTypeT>::
swapValues(MeshVariableScalarRefT<ItemType,DataType>& rhs)
{
  this->m_private_part->swapValues(*(rhs.m_private_part));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(datatype) \
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<datatype>;\
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<datatype>;\
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Node,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Edge,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Face,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Cell,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Particle,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<DoF,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Node,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Edge,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Face,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Cell,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Particle,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<DoF,datatype>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
