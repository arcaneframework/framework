// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartialVariableArrayRef.h                               (C) 2000-2024 */
/*                                                                           */
/* Classe gérant une variable partielle array sur une entité du maillage.    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHPARTIALVARIABLEARRAYREF_H
#define ARCANE_CORE_MESHPARTIALVARIABLEARRAYREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/MeshVariableRef.h"
#include "arcane/core/PrivateVariableArray.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroupRangeIterator.h"
#include "arcane/core/ItemPairEnumerator.h"
#include "arcane/core/GroupIndexTable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Variable
 * \brief Variable partielle scalaire sur un type d'entité du maillage.
 */
template<typename DataTypeT>
class ItemPartialVariableArrayRefT
: public PrivateVariableArrayT<DataTypeT>
{
 public:
  
  typedef DataTypeT DataType;
  typedef UniqueArray2<DataType> ValueType;
  typedef ConstArrayView<DataType> ConstReturnReferenceType;
  typedef ArrayView<DataType> ReturnReferenceType;

 protected:

  typedef PrivateVariableArrayT<DataType> BaseClass;
  typedef typename BaseClass::PrivatePartType PrivatePartType;
  
  typedef ArrayView<DataType> ArrayType;
  typedef ConstArrayView<DataType> ConstArrayType;

 public:

  //! Construit une référence à la variable spécifiée dans \a vb
  ARCANE_CORE_EXPORT ItemPartialVariableArrayRefT(const VariableBuildInfo& vb,eItemKind ik);
  
  //! Construit une référence à partir de \a var
  explicit ARCANE_CORE_EXPORT ItemPartialVariableArrayRefT(IVariable* var);
  
  //! Construit une référence à partir de \a rhs
  ARCANE_CORE_EXPORT ItemPartialVariableArrayRefT(const ItemPartialVariableArrayRefT<DataType>& rhs);
  
 protected:
  
  //! Opérateur de recopie
  ARCANE_CORE_EXPORT void operator=(const ItemPartialVariableArrayRefT<DataType>& rhs);

 public:
  
  ARCANE_CORE_EXPORT void fill(const DataType& value);
  ARCANE_CORE_EXPORT void copy(const ItemPartialVariableArrayRefT<DataType>& v);
  ARCANE_CORE_EXPORT void internalSetUsed(bool v);

 public:
 
  ConstArrayType operator[](const Item& i) const
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i);
    ARCANE_ASSERT((m_table.isUsed()),("GroupIndexTable expired"));
    const GroupIndexTable& table = *m_table;
    return this->m_view[table[i.localId()]]; 
  }
  ArrayType operator[](const Item& i)
  { 
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i);
    ARCANE_ASSERT((m_table.isUsed()),("GroupIndexTable expired"));
    const GroupIndexTable& table = *m_table;
    return this->m_view[table[i.localId()]]; 
  }
  ConstArrayType operator[](const ItemGroupRangeIterator& i) const
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i);
    return this->m_view[i.index()];
  }
  ArrayType operator[](const ItemGroupRangeIterator& i)
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i);
    return this->m_view[i.index()];
  }
  ConstArrayType operator[](const ItemEnumerator& i) const
  {
    ARCANE_CHECK_ENUMERATOR(i,this->itemGroup());
    return this->m_view[i.index()];
  }
  ArrayType operator[](const ItemEnumerator& i)
  {
    ARCANE_CHECK_ENUMERATOR(i,this->itemGroup());
    return this->m_view[i.index()];
  }
  ConstArrayType operator[](ItemEnumeratorIndex i) const
  {
    return this->m_view[i.index()];
  }
  ArrayType operator[](ItemEnumeratorIndex i)
  {
    return this->m_view[i.index()];
  }

 public:

  //! Vue sur table d'indirection du groupe.
  GroupIndexTableView tableView() const { return m_table->view(); }

 protected:
  
  SharedPtrT<GroupIndexTable> m_table;

 protected:

  static VariableInfo _buildVariableInfo(const VariableBuildInfo& vbi,eItemKind ik);
  static VariableTypeInfo _buildVariableTypeInfo(eItemKind ik);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Variable
 * \brief Variable scalaire sur un type d'entité du maillage.
 */
template<typename ItemTypeT,typename DataTypeT>
class MeshPartialVariableArrayRefT
: public ItemPartialVariableArrayRefT<DataTypeT>
{
 public:
  
  typedef DataTypeT DataType;
  typedef ItemTypeT ItemType;
  typedef UniqueArray2<DataType> ValueType;
  typedef ConstArrayView<DataType> ConstReturnReferenceType;
  typedef ArrayView<DataType> ReturnReferenceType;
  typedef ItemLocalIdT<ItemType> ItemLocalIdType;

 public:

  typedef ItemPartialVariableArrayRefT<DataType> BaseClass;

  typedef typename ItemTraitsT<ItemType>::ItemGroupType GroupType;

  typedef MeshPartialVariableArrayRefT<ItemType,DataType> ThatClass;

  typedef ArrayView<DataType> ArrayType;
  typedef ConstArrayView<DataType> ConstArrayType;

 public:

  //! Construit une référence à la variable spécifiée dans \a vb
  ARCANE_CORE_EXPORT MeshPartialVariableArrayRefT(const VariableBuildInfo& vb);
  //! Construit une référence à partir de \a rhs
  ARCANE_CORE_EXPORT MeshPartialVariableArrayRefT(const MeshPartialVariableArrayRefT<ItemType,DataType>& rhs);
  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const MeshPartialVariableArrayRefT<ItemType,DataType>& rhs);
  
 public:

  ConstArrayType operator[](const ItemLocalIdType& i) const
  {
    ARCANE_ASSERT((this->m_table.isUsed()),("GroupIndexTable expired"));
    const GroupIndexTable& table = *this->m_table;
    return this->m_view[table[i.asInt32()]];
  }
  ArrayType operator[](const ItemLocalIdType& i)
  {
    ARCANE_ASSERT((this->m_table.isUsed()),("GroupIndexTable expired"));
    const GroupIndexTable& table = *this->m_table;
    return this->m_view[table[i.asInt32()]];
  }
  ConstArrayType operator[](const ItemGroupRangeIteratorT<ItemType>& i) const
  {
    return this->m_view[i.index()]; 
  }
  ArrayType operator[](const ItemGroupRangeIteratorT<ItemType>& i)
  {
    return this->m_view[i.index()]; 
  }
  ConstArrayType operator[](const ItemPairEnumeratorSubT<ItemType>& i) const
  {
    return this->m_view[i.index()]; 
  }
  ArrayType operator[](const ItemPairEnumeratorSubT<ItemType>& i)
  {
    return this->m_view[i.index()]; 
  }
  ConstArrayType operator[](const ItemEnumeratorT<ItemType>& i) const
  {
    ARCANE_CHECK_ENUMERATOR(i,this->itemGroup());
    return this->m_view[i.index()]; 
  }
  ArrayType operator[](const ItemEnumeratorT<ItemType>& i)
  {
    ARCANE_CHECK_ENUMERATOR(i,this->itemGroup());
    return this->m_view[i.index()]; 
  }
  ConstArrayType operator[](ItemEnumeratorIndexT<ItemType> i) const
  {
    return this->m_view[i.index()];
  }
  ArrayType operator[](ItemEnumeratorIndexT<ItemType> i)
  {
    return this->m_view[i.index()];
  }

  //! Groupe associé à la grandeur
  ARCANE_CORE_EXPORT GroupType itemGroup() const;

 private:

  static VariableFactoryRegisterer m_auto_registerer;
  static VariableInfo _buildVariableInfo(const VariableBuildInfo& vbi);
  static VariableTypeInfo _buildVariableTypeInfo();
  static VariableRef* _autoCreate(const VariableBuildInfo& vb);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
