// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableArrayRef.h                                      (C) 2000-2020 */
/*                                                                           */
/* Classe gérant une variable vectorielle sur une entité du maillage.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHVARIABLEARRAYREF_H
#define ARCANE_MESHVARIABLEARRAYREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/PrivateVariableArray.h"
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
/*!
 * \brief Variable tableau sur un type d'entité du maillage.
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

  //! Construit une référence à la variable spécifiée dans \a vb
  ARCANE_CORE_EXPORT ItemVariableArrayRefT(const VariableBuildInfo& b,eItemKind ik);
  
  //! Construit une référence à partir de \a var
  explicit ARCANE_CORE_EXPORT ItemVariableArrayRefT(IVariable* var);
  
  //! Construit une référence à partir de \a rhs
  ARCANE_CORE_EXPORT ItemVariableArrayRefT(const ItemVariableArrayRefT<DataType>& rhs);
  
 protected:
  
  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT void operator=(const ItemVariableArrayRefT<DataType>& rhs);
  
 public:

  ConstArrayType operator[](const Item& i) const
  { 
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->m_view[i.localId()]; 
  }
  ArrayType operator[](const Item& i)
  { 
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->m_view[i.localId()]; 
  }
  ConstArrayType operator[](const ItemGroupRangeIterator& i) const
  {
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->m_view[i.itemLocalId()];
  }
  ArrayType operator[](const ItemGroupRangeIterator& i)
  {
    ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->m_view[i.itemLocalId()];
  }
  ConstArrayType operator[](const ItemEnumerator& i) const
  {
   ARCANE_ASSERT((i->kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
     return this->m_view[i.localId()];
  }
  ArrayType operator[](const ItemEnumerator& i)
  {
    ARCANE_ASSERT((i->kind() == this->itemGroup().itemKind()),("Item and group kind not same"));
    return this->m_view[i.localId()];
  }

 public:

  //! Copie les valeurs de \a v dans cette variable
  ARCANE_CORE_EXPORT void copy(const ItemVariableArrayRefT<DataType>& v);
  //! Copie les valeurs de \a v pour le groupe \a group dans cette variable
  ARCANE_CORE_EXPORT void copy(const ItemVariableArrayRefT<DataType>& v,const ItemGroup& group);
  //! Remplit la variable avec la valeur \a value
  ARCANE_CORE_EXPORT void fill(const DataType& value);
  //! Remplit la variable avec la valeur \a value pour les entités du groupe \a group 
  ARCANE_CORE_EXPORT void fill(const DataType& value,const ItemGroup& group);

 public:

  static ARCANE_CORE_EXPORT VariableInfo _internalVariableInfo(const VariableBuildInfo& vbi,eItemKind ik);
  static ARCANE_CORE_EXPORT VariableTypeInfo _internalVariableTypeInfo(eItemKind ik);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Variable
 * \brief Variable tableau sur un type d'entité du maillage.
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

  //! Construit une référence à la variable spécifiée dans \a vb
  ARCANE_CORE_EXPORT MeshVariableArrayRefT(const VariableBuildInfo& b);
  //! Construit une référence à partir de \a var
  explicit ARCANE_CORE_EXPORT MeshVariableArrayRefT(IVariable* var);
  //! Construit une référence à partir de \a rhs
  ARCANE_CORE_EXPORT MeshVariableArrayRefT(const MeshVariableArrayRefT<ItemType,DataType>& rhs);
  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const MeshVariableArrayRefT<ItemType,DataType>& rhs);

  ThatClass& operator=(const ThatClass& rhs) = delete; // Interdit.

 public:

  ConstArrayType operator[](const ItemType& i) const
  { return this->m_view[i.localId()]; }
  ArrayType operator[](const ItemType& i)
  { return this->m_view[i.localId()]; }
  ConstArrayType operator[](const ItemGroupRangeIteratorT<ItemType>& i) const
  { return this->m_view[i.itemLocalId()]; }
  ArrayType operator[](const ItemGroupRangeIteratorT<ItemType>& i)
  { return this->m_view[i.itemLocalId()]; }
  ConstArrayType operator[](const ItemPairEnumeratorSubT<ItemType>& i) const
  { return this->m_view[i.itemLocalId()]; }
  ArrayType operator[](const ItemPairEnumeratorSubT<ItemType>& i)
  { return this->m_view[i.itemLocalId()]; }
  ConstArrayType operator[](const ItemEnumeratorT<ItemType>& i) const
  { return this->m_view[i.localId()]; }
  ArrayType operator[](const ItemEnumeratorT<ItemType>& i)
  { return this->m_view[i.localId()]; }
  ConstArrayType operator[](ItemLocalIdType i) const
  { return this->m_view[i.localId()]; }
  ArrayType operator[](ItemLocalIdType i)
  { return this->m_view[i.localId()]; }

 public:
	
  //! Groupe associé à la grandeur
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
