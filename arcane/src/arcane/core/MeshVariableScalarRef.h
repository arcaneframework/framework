// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableScalarRef.h                                     (C) 2000-2025 */
/*                                                                           */
/* Class managing a scalar variable on a mesh entity.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHVARIABLESCALARREF_H
#define ARCANE_CORE_MESHVARIABLESCALARREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/MeshVariableRef.h"
#include "arcane/core/PrivateVariableScalar.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemGroupRangeIterator.h"
#include "arcane/core/ItemPairEnumerator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataTypeT>
class ItemNumericOperation
{
 public:
  typedef ItemVariableScalarRefT<DataTypeT> VarType;
 public:
  static void add(VarType&,const  VarType&,const ItemGroup&) { _notSupported(); }
  static void sub(VarType&,const  VarType&,const ItemGroup&) { _notSupported(); }
  static void mult(VarType&,const  VarType&,const ItemGroup&) { _notSupported(); }
  static void mult(VarType&,const DataTypeT&,const ItemGroup&) { _notSupported(); }
  static void power(VarType&,const DataTypeT&,const ItemGroup&) { _notSupported(); }
  static void _notSupported()
  { ARCANE_FATAL("ItemNumeraticOperation: operation not supported"); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ItemNumericOperation<Real>
{
 public:
  typedef ItemVariableScalarRefT<Real> VarType;
 public:
  static ARCANE_CORE_EXPORT void add  (VarType& out,const VarType& v,const ItemGroup& group);
  static ARCANE_CORE_EXPORT void sub  (VarType& out,const VarType& v,const ItemGroup& group);
  static ARCANE_CORE_EXPORT void mult (VarType& out,const VarType& v,const ItemGroup& group);
  static ARCANE_CORE_EXPORT void mult (VarType& out,Real v,const ItemGroup& group);
  static ARCANE_CORE_EXPORT void power(VarType& out,Real v,const ItemGroup& group);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Variable
 * \brief Scalar variable on a mesh entity type.
 */
template<typename DataTypeT>
class ItemVariableScalarRefT
: public PrivateVariableScalarT<DataTypeT>
{
 public:
  
  typedef DataTypeT DataType;
  typedef UniqueArray<DataTypeT> ValueType;
  typedef const DataTypeT& ConstReturnReferenceType;
  typedef DataTypeT& ReturnReferenceType;
  
  typedef ArrayView<DataTypeT> ArrayBase;
  typedef PrivateVariableScalarT<DataTypeT> BaseClass;
  typedef typename BaseClass::PrivatePartType PrivatePartType;
  typedef typename BaseClass::DataTypeReturnReference DataTypeReturnReference;
  
 public:

  //! Constructs a reference to the variable specified in \a vb
  ARCANE_CORE_EXPORT ItemVariableScalarRefT(const VariableBuildInfo& b,eItemKind ik);

  //! Constructs a reference from \a var
  explicit ARCANE_CORE_EXPORT ItemVariableScalarRefT(IVariable* var);
   
  //! Constructs a reference from \a rhs
  ARCANE_CORE_EXPORT ItemVariableScalarRefT(const ItemVariableScalarRefT<DataTypeT>& rhs);
  
 protected:

  //! Positions the instance reference to the variable \a rhs.
  ARCANE_CORE_EXPORT void operator=(const ItemVariableScalarRefT<DataTypeT>& rhs);

#ifdef ARCANE_DOTNET
 public:
#else
 protected:
#endif

  //! Default constructor
  ItemVariableScalarRefT() {}
 
 public:

  void add(const ItemVariableScalarRefT<DataTypeT>& v)
  {
    add(v,this->itemGroup());
  }
  void sub(const ItemVariableScalarRefT<DataTypeT>& v)
  {
    sub(v,this->itemGroup());
  }
  void mult(const ItemVariableScalarRefT<DataTypeT>& v)
  {
    mult(v,this->itemGroup());
  }
  void mult(const DataTypeT& v)
  {
    mult(v,this->itemGroup());
  }
  void copy(const ItemVariableScalarRefT<DataTypeT>& v)
  {
    copy(v,this->itemGroup());
  }
  void power(const DataTypeT& v)
  {
    power(v,this->itemGroup());
  }

  void add(const ItemVariableScalarRefT<DataTypeT>& v,const ItemGroup& group)
  {
    ItemNumericOperation<DataTypeT>::add(*this,v,group);
  }
  void sub(const ItemVariableScalarRefT<DataTypeT>& v,const ItemGroup& group)
  {
    ItemNumericOperation<DataTypeT>::sub(*this,v,group);
  }
  void mult(const ItemVariableScalarRefT<DataTypeT>& v,const ItemGroup& group)
  {
    ItemNumericOperation<DataTypeT>::mult(*this,v,group);
  }
  void mult(const DataTypeT& v,const ItemGroup& group)
  {
    ItemNumericOperation<DataTypeT>::mult(*this,v,group);
  }
  void power(const DataTypeT& v,const ItemGroup& group)
  {
    ItemNumericOperation<DataTypeT>::power(*this,v,group);
  }

  //! Copies the values of \a v into this variable for the entities in \a group.
  ARCANE_CORE_EXPORT void copy(const ItemVariableScalarRefT<DataTypeT>& v,const ItemGroup& group);
  ARCANE_CORE_EXPORT void fill(const DataTypeT& value);
  ARCANE_CORE_EXPORT void fill(const DataTypeT& value,const ItemGroup& group);

  /*!
   * \brief Copies the values of \a v into this variable via the \a queue.
   *
   * \a queue may be null.
   */
  ARCANE_CORE_EXPORT void copy(const ItemVariableScalarRefT<DataTypeT>& v,RunQueue* queue);
  /*!
   * \brief Fills the instance values with \a value via the \a queue.
   *
   * \a queue may be null.
   */
  ARCANE_CORE_EXPORT void fill(const DataTypeT& value,RunQueue* queue);

 public:

  //! Read-only value of entity \a item
  const DataType& operator[](ItemLocalId item) const { return this->_value(item.localId()); }

  //! Read/write value of entity \a item
  DataTypeReturnReference operator[](ItemLocalId item) { return this->_value(item.localId()); }

  //! Read-only value of entity \a item
  const DataType& operator()(ItemLocalId item) const { return this->_value(item.localId()); }

  //! Read/write value of entity \a item
  DataTypeReturnReference operator()(ItemLocalId item) { return this->_value(item.localId()); }

 public:

  static ARCANE_CORE_EXPORT VariableInfo _internalVariableInfo(const VariableBuildInfo& vbi,eItemKind ik);
  static ARCANE_CORE_EXPORT VariableTypeInfo _internalVariableTypeInfo(eItemKind ik);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Variable
 * \brief Scalar variable on a mesh entity type.
 */
template<typename ItemTypeT,typename DataTypeT>
class MeshVariableScalarRefT
: public ItemVariableScalarRefT<DataTypeT>
{
 public:
  
  typedef DataTypeT DataType;
  typedef ItemTypeT ItemType;
  typedef UniqueArray<DataTypeT> ValueType;
  typedef const DataTypeT& ConstReturnReferenceType;
  typedef DataTypeT& ReturnReferenceType;
 
 protected:

  typedef typename ItemType::LocalIdType ItemLocalIdType;
  typedef ItemVariableScalarRefT<DataTypeT> BaseClass;
  
  typedef typename ItemTraitsT<ItemType>::ItemGroupType GroupType;
  typedef MeshVariableScalarRefT<ItemType,DataTypeT> ThatClass;
  typedef typename BaseClass::DataTypeReturnReference DataTypeReturnReference;

 public:

  //! Constructs a reference to the variable specified in \a vb
  ARCANE_CORE_EXPORT MeshVariableScalarRefT(const VariableBuildInfo& vb);
  //! Constructs a reference from \a var
  explicit ARCANE_CORE_EXPORT MeshVariableScalarRefT(IVariable* var);
  //! Constructs a reference from \a rhs
  ARCANE_CORE_EXPORT MeshVariableScalarRefT(const MeshVariableScalarRefT<ItemType,DataTypeT>& rhs);

  //! Positions the instance reference to the variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const MeshVariableScalarRefT<ItemType,DataTypeT>& rhs);

  ThatClass& operator=(const ThatClass& rhs) = delete;

#ifdef ARCANE_DOTNET
 public:
#else
 protected:
#endif

  //! Default constructor
  MeshVariableScalarRefT(){}
  ThatClass& _Internal() { return *this; }

 public:

  void fill(const DataTypeT& value) { BaseClass::fill(value); }
  void fill(const DataTypeT& value,RunQueue* queue) { BaseClass::fill(value,queue); }
  void fill(const DataTypeT& value,const GroupType& group) { BaseClass::fill(value,group); }

  ARCANE_CORE_EXPORT void swapValues(MeshVariableScalarRefT<ItemType,DataType>& rhs);

  //! Group associated with the quantity
  ARCANE_CORE_EXPORT GroupType itemGroup() const;
   
  ARCANE_CORE_EXPORT void setIsSynchronized();
  ARCANE_CORE_EXPORT void setIsSynchronized(const GroupType& group);

 public:

  //! Read-only value of entity \a item
  const DataTypeT& operator[](ItemLocalIdType i) const { return this->_value(i.localId()); }

  //! Read/write value of entity \a item
  DataTypeReturnReference operator[](ItemLocalIdType i) { return this->_value(i.localId()); }

  //! Read-only value of entity \a item
  const DataTypeT& operator()(ItemLocalIdType i) const { return this->_value(i.localId()); }

  //! Read/write value of entity \a item
  DataTypeReturnReference operator()(ItemLocalIdType i) { return this->_value(i.localId()); }

  //! Read/write value of entity \a item
  DataTypeT& getReference(ItemLocalIdType item)
  {
    return this->_value(item.localId());
  }

  const DataTypeT& item(ItemLocalIdType i) const
  {
    return this->_value(i.localId());
  }
  void setItem(ItemLocalIdType i,const DataTypeT& v)
  {
    this->_value(i.localId()) = v;
  }
#ifdef ARCANE_DOTNET_WRAPPER
  const DataTypeT& item(Int32 i) const
  {
    return this->_value(i);
  }
  void setItem(Int32 i,const DataTypeT& v)
  {
    this->_value(i) = v;
  }
#endif

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
