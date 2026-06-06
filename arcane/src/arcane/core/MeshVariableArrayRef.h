// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableArrayRef.h                                      (C) 2000-2024 */
/*                                                                           */
/* Class managing a vector variable on a mesh entity.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHVARIABLEARRAYREF_H
#define ARCANE_CORE_MESHVARIABLEARRAYREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/PrivateVariableArray.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroupRangeIterator.h"
#include "arcane/core/ItemPairEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Array variable on a mesh entity type.
 */
template<class DataTypeT>
class ItemVariableArrayRefT
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

  //! Constructs a reference to the variable specified in \a vb
  ARCANE_CORE_EXPORT ItemVariableArrayRefT(const VariableBuildInfo& b,eItemKind ik);
  
  //! Constructs a reference from \a var
  explicit ARCANE_CORE_EXPORT ItemVariableArrayRefT(IVariable* var);
  
  //! Constructs a reference from \a rhs
  ARCANE_CORE_EXPORT ItemVariableArrayRefT(const ItemVariableArrayRefT<DataType>& rhs);
  
 protected:
  
  //! Positions the instance reference to the variable \a rhs.
  ARCANE_CORE_EXPORT ItemVariableArrayRefT<DataType>& operator=(const ItemVariableArrayRefT<DataType>& rhs);

 public:

  //! Read-only value of entity \a item
  ConstArrayType operator[](ItemLocalId item) const { return this->m_view[item.localId()]; }

  //! Modifiable value of entity \a item
  ArrayType operator[](ItemLocalId item) { return this->m_view[item.localId()]; }

  //! Read-only value of the \a i-th value of entity \a item
  const DataType operator()(ItemLocalId item,Int32 i) const { return this->m_view.item(item.localId(),i); }

  //! Modifiable value of the \a i-th value of entity \a item
  DataType& operator()(ItemLocalId item,Int32 i) { return this->m_view[item.localId()][i]; }

 public:

  //! Copies the values of \a v into this variable
  ARCANE_CORE_EXPORT void copy(const ItemVariableArrayRefT<DataType>& v);
  //! Copies the values of \a v for the group \a group into this variable
  ARCANE_CORE_EXPORT void copy(const ItemVariableArrayRefT<DataType>& v,const ItemGroup& group);
  /*!
   * \brief Copies the values of \a v into this variable via the queue \a queue.
   *
   * \a queue may be null.
   */
  ARCANE_CORE_EXPORT void copy(const ItemVariableArrayRefT<DataType>& v,RunQueue* queue);
  //! Fills the variable with the value \a value
  ARCANE_CORE_EXPORT void fill(const DataType& value);
  //! Fills the variable with the value \a value for the entities in group \a group 
  ARCANE_CORE_EXPORT void fill(const DataType& value,const ItemGroup& group);
  //! Fills the variable with the value \a value via the queue \a queue
  ARCANE_CORE_EXPORT void fill(const DataType& value,RunQueue* queue);

 public:

  static ARCANE_CORE_EXPORT VariableInfo _internalVariableInfo(const VariableBuildInfo& vbi,eItemKind ik);
  static ARCANE_CORE_EXPORT VariableTypeInfo _internalVariableTypeInfo(eItemKind ik);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Variable
 * \brief Array variable on a mesh entity type.
 */
template<class ItemTypeT,class DataTypeT>
class MeshVariableArrayRefT
: public ItemVariableArrayRefT<DataTypeT>
{
 public:
  
  typedef DataTypeT DataType;
  typedef ItemTypeT ItemType;
  typedef UniqueArray2<DataType> ValueType;
  typedef ConstArrayView<DataType> ConstReturnReferenceType;
  typedef ArrayView<DataType> ReturnReferenceType;

 protected:

  typedef MeshVariableRef BaseClass;

  typedef Array2View<DataType> ArrayBase;

  typedef ArrayView<DataType> ArrayType;
  typedef ConstArrayView<DataType> ConstArrayType;

  typedef Array2VariableT<DataType> PrivatePartType;

  typedef typename ItemTraitsT<ItemType>::ItemGroupType GroupType;
  typedef typename ItemTypeT::LocalIdType ItemLocalIdType;

  typedef MeshVariableArrayRefT<ItemType,DataType> ThatClass;

 public:

  //! Constructs a reference to the variable specified in \a vb
  ARCANE_CORE_EXPORT MeshVariableArrayRefT(const VariableBuildInfo& b);
  //! Constructs a reference from \a var
  explicit ARCANE_CORE_EXPORT MeshVariableArrayRefT(IVariable* var);
  //! Constructs a reference from \a rhs
  ARCANE_CORE_EXPORT MeshVariableArrayRefT(const MeshVariableArrayRefT<ItemType,DataType>& rhs);
  //! Positions the instance reference to the variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const MeshVariableArrayRefT<ItemType,DataType>& rhs);

  ThatClass& operator=(const ThatClass& rhs) = delete; // Forbidden.

 public:

  //! Read-only value of entity \a item
  ConstArrayType operator[](ItemLocalIdType item) const { return this->m_view[item.localId()]; }

  //! Modifiable value of entity \a item
  ArrayType operator[](ItemLocalIdType item) { return this->m_view[item.localId()]; }

  //! Read-only value of the \a i-th value of entity \a item
  const DataType operator()(ItemLocalIdType item,Int32 i) const { return this->m_view[item.localId()][i]; }

  //! Modifiable value of the \a i-th value of entity \a item
  DataType& operator()(ItemLocalIdType item,Int32 i) { return this->m_view[item.localId()][i]; }

 public:
	
  //! Group associated with the quantity
  ARCANE_CORE_EXPORT GroupType itemGroup() const;
  ARCANE_CORE_EXPORT void swapValues(MeshVariableArrayRefT<ItemType,DataType>& rhs);
	
 public:

  static ARCANE_CORE_EXPORT VariableTypeInfo _internalVariableTypeInfo();
  static ARCANE_CORE_EXPORT VariableInfo _internalVariableInfo(const VariableBuildInfo& vbi);

 private:

  static VariableFactoryRegisterer m_auto_registerer;
  static VariableRef* _autoCreate(const VariableBuildInfo& vb);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
