// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableScalarRef.h                                     (C) 2000-2025 */
/*                                                                           */
/* Classe gérant une variable scalaire sur une entité du maillage.           */
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

  //! Copie pour les entités de \a group, les valeurs de \a v dans cette variable.
  ARCANE_CORE_EXPORT void copy(const ItemVariableScalarRefT<DataTypeT>& v,const ItemGroup& group);
  ARCANE_CORE_EXPORT void fill(const DataTypeT& value);
  ARCANE_CORE_EXPORT void fill(const DataTypeT& value,const ItemGroup& group);

  /*!
   * \brief Copie les valeurs de \a v dans cette variable via la file \a queue.
   *
   * \a queue peut être nul.
   */
  ARCANE_CORE_EXPORT void copy(const ItemVariableScalarRefT<DataTypeT>& v,RunQueue* queue);
  /*!
   * \brief Remplit les valeurs de l'instance par \a value via la file \a queue.
   *
   * \a queue peut être nul.
   */
  ARCANE_CORE_EXPORT void fill(const DataTypeT& value,RunQueue* queue);

 public:

  //! Valeur non modifiable de l'entité \a item
  const DataType& operator[](ItemLocalId item) const { return this->_value(item.localId()); }

  //! Valeur modifiable de l'entité \a item
  DataTypeReturnReference operator[](ItemLocalId item) { return this->_value(item.localId()); }

  //! Valeur non modifiable de l'entité \a item
  const DataType& operator()(ItemLocalId item) const { return this->_value(item.localId()); }

  //! Valeur modifiable de l'entité \a item
  DataTypeReturnReference operator()(ItemLocalId item) { return this->_value(item.localId()); }

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

  void fill(const DataTypeT& value) { BaseClass::fill(value); }
  void fill(const DataTypeT& value,RunQueue* queue) { BaseClass::fill(value,queue); }
  void fill(const DataTypeT& value,const GroupType& group) { BaseClass::fill(value,group); }

  ARCANE_CORE_EXPORT void swapValues(MeshVariableScalarRefT<ItemType,DataType>& rhs);

  //! Groupe associé à la grandeur
  ARCANE_CORE_EXPORT GroupType itemGroup() const;
   
  ARCANE_CORE_EXPORT void setIsSynchronized();
  ARCANE_CORE_EXPORT void setIsSynchronized(const GroupType& group);

 public:

  //! Valeur non modifiable de l'entité \a item
  const DataTypeT& operator[](ItemLocalIdType i) const { return this->_value(i.localId()); }

  //! Valeur modifiable de l'entité \a item
  DataTypeReturnReference operator[](ItemLocalIdType i) { return this->_value(i.localId()); }

  //! Valeur non modifiable de l'entité \a item
  const DataTypeT& operator()(ItemLocalIdType i) const { return this->_value(i.localId()); }

  //! Valeur modifiable de l'entité \a item
  DataTypeReturnReference operator()(ItemLocalIdType i) { return this->_value(i.localId()); }

  //! Valeur modifiable de l'entité \a item
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
