// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableScalarRef.h                                     (C) 2000-2020 */
/*                                                                           */
/* Classe gérant une variable scalaire sur une entité du maillage.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHVARIABLESCALARREF_H
#define ARCANE_MESHVARIABLESCALARREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/MeshVariableRef.h"
#include "arcane/PrivateVariableScalar.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/ItemGroupRangeIterator.h"
#include "arcane/ItemPairEnumerator.h"

#include "arcane/utils/FatalErrorException.h"

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
  static void add  (VarType&,const  VarType&,const ItemGroup&) { _notSupported(); }
  static void sub  (VarType&,const  VarType&,const ItemGroup&) { _notSupported(); }
  static void mult (VarType&,const  VarType&,const ItemGroup&) { _notSupported(); }
  static void mult (VarType&,const DataTypeT&,const ItemGroup&) { _notSupported(); }
  static void power(VarType&,const DataTypeT&,const ItemGroup&) { _notSupported(); }
  static void _notSupported()
  { throw FatalErrorException("ItemNumeraticOperation: operation not supported"); }
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
 * \brief Variable scalaire sur un type d'entité du maillage.
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
#ifdef ARCANE_PROXY
  typedef typename BaseClass::ProxyType ProxyType;
#endif
  
 public:

  //! Construit une référence à la variable spécifiée dans \a vb
  ARCANE_CORE_EXPORT ItemVariableScalarRefT(const VariableBuildInfo& b,eItemKind ik);

  //! Construit une référence à partir de \a var
  explicit ARCANE_CORE_EXPORT ItemVariableScalarRefT(IVariable* var);
   
  //! Construit une référence à partir de \a rhs
  ARCANE_CORE_EXPORT ItemVariableScalarRefT(const ItemVariableScalarRefT<DataTypeT>& rhs);
  
 protected:

  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT void operator=(const ItemVariableScalarRefT<DataTypeT>& rhs);

#ifdef ARCANE_DOTNET
public:
#else
protected:
#endif

  //! Constructeur vide
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
  ARCANE_CORE_EXPORT void copy(const ItemVariableScalarRefT<DataTypeT>& v,const ItemGroup& group);
  ARCANE_CORE_EXPORT void fill(const DataTypeT& value);
  ARCANE_CORE_EXPORT void fill(const DataTypeT& value,const ItemGroup& group);

public:

  const DataTypeT& operator[](const Item& i) const
  {
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.localId()).setRead();
#endif
    return this->_value(i.localId());
  }
#ifdef ARCANE_PROXY
  ProxyType operator[](const Item& i)
  {
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->_getProxy(i.localId());
  }
#else
  DataTypeReturnReference operator[](const Item& i)
  {
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->_value(i.localId());
  }
#endif

  const DataTypeT& operator[](const ItemGroupRangeIteratorT<Item>& i) const
  {
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.itemLocalId()).setRead();
#endif
    return this->_value(i.itemLocalId());
  }

#ifdef ARCANE_PROXY
  ProxyType operator[](const ItemGroupRangeIterator& i)
  {
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->_getProxy(i.itemLocalId());
  }
#else
  DataTypeReturnReference operator[](const ItemGroupRangeIterator& i)
  {
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->_value(i.itemLocalId());
  }
#endif

  const DataTypeT& operator[](const ItemEnumerator& i) const
  {
    ARCANE_ASSERT((i->kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.itemLocalId()).setRead();
#endif
    return this->_value(i.itemLocalId());
  }

#ifdef ARCANE_PROXY
  ProxyType operator[](const ItemEnumerator& i)
  {
    ARCANE_ASSERT((i->kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->_getProxy(i.itemLocalId());
  }
#else
  DataTypeReturnReference operator[](const ItemEnumerator& i)
  {
    ARCANE_ASSERT((i->kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->_value(i.itemLocalId());
  }
#endif

  const DataTypeT& operator[](const ItemPairEnumerator& i) const
  {
    ARCANE_ASSERT(((*i).kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.itemLocalId()).setRead();
#endif
    return this->_value(i.itemLocalId());
  }
#ifdef ARCANE_PROXY
  ProxyType operator[](const ItemPairEnumerator& i)
  {
    ARCANE_ASSERT(((*i).kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->_getProxy(i.itemLocalId());
  }
#else
  DataTypeReturnReference operator[](const ItemPairEnumerator& i)
  {
    ARCANE_ASSERT(((*i).kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->_value(i.itemLocalId());
  }
#endif

 public:

  static ARCANE_CORE_EXPORT VariableInfo _internalVariableInfo(const VariableBuildInfo& vbi,eItemKind ik);
  static ARCANE_CORE_EXPORT VariableTypeInfo _internalVariableTypeInfo(eItemKind ik);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Variable
 * \brief Variable scalaire sur un type d'entité du maillage.
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

  typedef typename ItemType::Index ItemIndexType;
  typedef typename ItemType::LocalIdType ItemLocalIdType;
  typedef ItemVariableScalarRefT<DataTypeT> BaseClass;
  
  typedef typename ItemTraitsT<ItemType>::ItemGroupType GroupType;
  typedef MeshVariableScalarRefT<ItemType,DataTypeT> ThatClass;
  typedef typename BaseClass::DataTypeReturnReference DataTypeReturnReference;

public:

  //! Construit une référence à la variable spécifiée dans \a vb
  ARCANE_CORE_EXPORT MeshVariableScalarRefT(const VariableBuildInfo& vb);
  //! Construit une référence à partir de \a var
  explicit ARCANE_CORE_EXPORT MeshVariableScalarRefT(IVariable* var);
  //! Construit une référence à partir de \a rhs
  ARCANE_CORE_EXPORT MeshVariableScalarRefT(const MeshVariableScalarRefT<ItemType,DataTypeT>& rhs);

  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const MeshVariableScalarRefT<ItemType,DataTypeT>& rhs);

  ThatClass& operator=(const ThatClass& rhs) = delete;

#ifdef ARCANE_DOTNET
public:
#else
protected:
#endif

  //! Constructeur vide
  MeshVariableScalarRefT(){}
  ThatClass& _Internal() { return *this; }

public:

#ifdef ARCANE_PROXY
  typedef typename DataTypeTraitsT<DataTypeT>::ProxyType ProxyType;
#endif

public:

  void fill(const DataTypeT& value) { BaseClass::fill(value); }
  
  void fill(const DataTypeT& value,const GroupType& group) { BaseClass::fill(value,group); }

  ARCANE_CORE_EXPORT void swapValues(MeshVariableScalarRefT<ItemType,DataType>& rhs);

  //! Groupe associé à la grandeur
  ARCANE_CORE_EXPORT GroupType itemGroup() const;
   
  ARCANE_CORE_EXPORT void setIsSynchronized();
  ARCANE_CORE_EXPORT void setIsSynchronized(const GroupType& group);

public:

  const DataTypeT& operator[](const ItemType& i) const
  {
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.localId()).setRead();
#endif
    return this->_value(i.localId());
  }
#ifdef ARCANE_PROXY
  ProxyType operator[](const ItemType& i)
  {
    return this->_getProxy(i.localId());
  }
#else
  DataTypeReturnReference operator[](const ItemType& i)
  {
    return this->_value(i.localId());
  }
#endif
  DataTypeT& getReference(const ItemType& i)
  {
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.localId()).setReadOrWrite();
#endif
    return this->_value(i.localId());
  }

  const DataTypeT& operator[](const ItemGroupRangeIteratorT<ItemType>& i) const
  {
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.itemLocalId()).setRead();
#endif
    return this->_value(i.itemLocalId());
  }
#ifdef ARCANE_PROXY
  ProxyType operator[](const ItemGroupRangeIteratorT<ItemType>& i)
  {
    return this->_getProxy(i.itemLocalId());
  }
#else
  DataTypeReturnReference operator[](const ItemGroupRangeIteratorT<ItemType>& i)
  {
    return this->_value(i.itemLocalId());
  }
#endif

  const DataTypeT& operator[](const ItemEnumeratorT<ItemType>& i) const
  {
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.itemLocalId()).setRead();
#endif
    return this->_value(i.itemLocalId());
  }
#ifdef ARCANE_PROXY
  ProxyType operator[](const ItemEnumeratorT<ItemType>& i)
  {
    return this->_getProxy(i.itemLocalId());
  }
#else
  DataTypeReturnReference operator[](const ItemEnumeratorT<ItemType>& i)
  {
    return this->_value(i.itemLocalId());
  }
#endif

  const DataTypeT& operator[](const ItemPairEnumeratorSubT<ItemType>& i) const
  {
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.itemLocalId()).setRead();
#endif
    return this->_value(i.itemLocalId());
  }
#ifdef ARCANE_PROXY
  ProxyType operator[](const ItemPairEnumeratorSubT<ItemType>& i)
  {
    return this->_getProxy(i.itemLocalId());
  }
#else
  DataTypeReturnReference operator[](const ItemPairEnumeratorSubT<ItemType>& i)
  {
    return this->_value(i.itemLocalId());
  }
#endif

  const DataTypeT& item(const ItemGroupRangeIteratorT<ItemType>& i) const
  {
    return this->_value(i.index());
  }
  void setItem(const ItemGroupRangeIteratorT<ItemType>& i,const DataTypeT& v)
  {
    this->_value(i.index()) = v;
  }
  const DataTypeT& item(const ItemType& i) const
  {
    return this->_value(i.localId());
  }
  void setItem(const ItemType& i,const DataTypeT& v)
  {
    this->_value(i.localId()) = v;
  }
  const DataTypeT& item(const ItemEnumeratorT<ItemType>& i) const
  {
    return this->_value(i.index());
  }
  void setItem(const ItemEnumeratorT<ItemType>& i,const DataTypeT& v)
  {
    this->_value(i.index()) = v;
  }
  const DataTypeT& item(const ItemPairEnumeratorSubT<ItemType>& i) const
  {
    return this->_value(i.index());
  }
  void setItem(const ItemPairEnumeratorSubT<ItemType>& i,const DataTypeT& v)
  {
    this->_value(i.index()) = v;
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

  const DataType& operator[](ItemIndexType i) const
  {
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.localId()).setRead();
#endif
    return this->_value(i.localId());
  }
#ifdef ARCANE_PROXY
  ProxyType operator[](ItemIndexType i)
  {
    return this->_getProxy(i.localId());
  }
#else
  DataTypeReturnReference operator[](ItemIndexType i)
  {
    return this->_value(i.localId());
  }
#endif

  const DataType& operator[](ItemLocalIdType i) const
  {
#ifdef ARCANE_PROXY
    this->_getMemoryInfo(i.localId()).setRead();
#endif
    return this->_value(i.localId());
  }
#ifdef ARCANE_PROXY
  ProxyType operator[](ItemLocalIdType i)
  {
    return this->_getProxy(i.localId());
  }
#else
  DataTypeReturnReference operator[](ItemLocalIdType i)
  {
    return this->_value(i.localId());
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
