// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartialVariableScalarRef.h                              (C) 2000-2025 */
/*                                                                           */
/* Class managing a partial scalar variable on a mesh entity.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHPARTIALVARIABLESCALARREF_H
#define ARCANE_CORE_MESHPARTIALVARIABLESCALARREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/MeshVariableRef.h"
#include "arcane/core/PrivateVariableScalar.h"
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
 * \brief Partial scalar variable on a mesh entity type.
 */
template<typename DataTypeT>
class ItemPartialVariableScalarRefT
: public PrivateVariableScalarT<DataTypeT>
{
 public:

  typedef DataTypeT DataType;
  typedef UniqueArray<DataType> ValueType;
  typedef const DataType& ConstReturnReferenceType;
  typedef DataType& ReturnReferenceType;

 protected:

  typedef PrivateVariableScalarT<DataType> BaseClass;
  typedef typename BaseClass::PrivatePartType PrivatePartType;
  typedef typename BaseClass::DataTypeReturnReference DataTypeReturnReference;
  
 public:

  //! Constructs a reference to the variable specified in \a vb
  ARCANE_CORE_EXPORT ItemPartialVariableScalarRefT(const VariableBuildInfo& vb,eItemKind ik);
  //! Constructs a reference from \a var
  explicit ARCANE_CORE_EXPORT ItemPartialVariableScalarRefT(IVariable* var);  
  //! Constructs a reference from \a rhs
  ARCANE_CORE_EXPORT ItemPartialVariableScalarRefT(const ItemPartialVariableScalarRefT<DataType>& rhs);
  
 protected:
  
  //! Copy operator
  ARCANE_CORE_EXPORT void operator=(const ItemPartialVariableScalarRefT<DataType>& rhs);
  
 public:
	
  ARCANE_CORE_EXPORT void fill(const DataType& value);  
  ARCANE_CORE_EXPORT void copy(const ItemPartialVariableScalarRefT<DataType>& v);
  ARCANE_CORE_EXPORT void internalSetUsed(bool v);

 public:
  
  const DataType& operator[](const Item& i) const
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i);
    ARCANE_ASSERT((m_table.isUsed()),("GroupIndexTable expired"));
    const GroupIndexTable& table = *m_table;
    return this->_value(table[i.localId()]);
  }
  DataTypeReturnReference operator[](const Item& i)
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i);
    ARCANE_ASSERT((m_table.isUsed()),("GroupIndexTable expired"));
    const GroupIndexTable& table = *m_table;
    return this->_value(table[i.localId()]);
  }

  const DataType& operator[](const ItemGroupRangeIterator& i) const
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i);
    return this->_value(i.index());
  }
  const DataType& operator[](const ItemEnumerator& i) const
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND((*i));
    ARCANE_CHECK_ENUMERATOR(i,this->itemGroup());
    return this->_value(i.index());
  }
  const DataType& operator[](ItemEnumeratorIndex i) const
  {
    return this->_value(i.index());
  }
  const DataType& operator[](const ItemPairEnumerator& i) const
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND((*i));
    return this->_value(i.index());
  }
  
  DataTypeReturnReference operator[](const ItemGroupRangeIterator& i)
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i);
    return this->_value(i.index());
  }
  DataTypeReturnReference operator[](const ItemEnumerator& i)
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND((*i));
    ARCANE_CHECK_ENUMERATOR(i,this->itemGroup());
    return this->_value(i.index());
  }
  DataTypeReturnReference operator[](ItemEnumeratorIndex i)
  {
    return this->_value(i.index());
  }
  DataTypeReturnReference operator[](const ItemPairEnumerator& i)
  {
    ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND((*i));
    return this->_value(i.index());
  }

 public:

  //! View of the group's redirection table.
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
 * \brief Scalar variable on a mesh entity type.
 */
template<typename ItemTypeT,typename DataTypeT>
class MeshPartialVariableScalarRefT
: public ItemPartialVariableScalarRefT<DataTypeT>
{
 public:
 
  typedef DataTypeT DataType;
  typedef ItemTypeT ItemType;
  typedef UniqueArray<DataType> ValueType;
  typedef const DataType& ConstReturnReferenceType;
  typedef DataType& ReturnReferenceType;
  typedef ItemLocalIdT<ItemType> ItemLocalIdType;

  typedef ItemPartialVariableScalarRefT<DataType> BaseClass;

  typedef typename ItemTraitsT<ItemType>::ItemGroupType GroupType;

  typedef MeshPartialVariableScalarRefT<ItemType,DataType> ThatClass;
  typedef typename BaseClass::DataTypeReturnReference DataTypeReturnReference;

 public:

  //! Constructs a reference to the variable specified in \a vb
  ARCANE_CORE_EXPORT MeshPartialVariableScalarRefT(const VariableBuildInfo& vb);
  //! Constructs a reference from \a rhs
  ARCANE_CORE_EXPORT MeshPartialVariableScalarRefT(const MeshPartialVariableScalarRefT<ItemType,DataType>& rhs);
  //! Positions the instance reference to the variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const MeshPartialVariableScalarRefT<ItemType,DataType>& rhs);
  
 public:

  const DataType& operator[](const ItemLocalIdType& i) const
  {
    ARCANE_ASSERT((this->m_table.isUsed()), ("GroupIndexTable expired"));
    const GroupIndexTable& table = *this->m_table;
    return this->_value(table[i.asInt32()]);
  }
  DataTypeReturnReference operator[](const ItemLocalIdType& i)
  {
    ARCANE_ASSERT((this->m_table.isUsed()), ("GroupIndexTable expired"));
    const GroupIndexTable& table = *this->m_table;
    return this->_value(table[i.asInt32()]);
  }

  const DataType& operator[](const ItemGroupRangeIteratorT<ItemType>& i) const
  {
    return this->_value(i.index());
  }
  DataTypeReturnReference operator[](const ItemGroupRangeIteratorT<ItemType>& i)
  {
    return this->_value(i.index());
  }
  const DataType& operator[](const ItemEnumeratorT<ItemType>& i) const
  {
    ARCANE_CHECK_ENUMERATOR(i,this->itemGroup());
    return this->_value(i.index());
  }
  const DataType& operator[](ItemEnumeratorIndexT<ItemType> i) const
  {
    return this->_value(i.index());
  }
  DataTypeReturnReference operator[](const ItemEnumeratorT<ItemType>& i)
  {
    ARCANE_CHECK_ENUMERATOR(i,this->itemGroup());
    return this->_value(i.index());
  }
  DataTypeReturnReference operator[](ItemEnumeratorIndexT<ItemType> i)
  {
    return this->_value(i.index());
  }
  const DataType& operator[](const ItemPairEnumeratorSubT<ItemType>& i) const
  {
    return this->_value(i.index());
  }
  DataTypeReturnReference operator[](const ItemPairEnumeratorSubT<ItemType>& i)
  {
    return this->_value(i.index());
  }

  //! Group associated with the quantity
  ARCANE_CORE_EXPORT GroupType itemGroup() const;
  
 private:

  static VariableFactoryRegisterer m_auto_registerer;
  //static VariableInfo _buildVariableInfo(const VariableBuildInfo& vbi);
  static VariableTypeInfo _buildVariableTypeInfo();
  static VariableRef* _autoCreate(const VariableBuildInfo& vb);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
