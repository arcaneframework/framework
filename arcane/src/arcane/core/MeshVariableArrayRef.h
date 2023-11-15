// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableArrayRef.h                                      (C) 2000-2023 */
/*                                                                           */
/* Classe gérant une variable vectorielle sur une entité du maillage.        */
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

  //! Valeur non modifiable de l'entité \a item
  ConstArrayType operator[](ItemLocalId item) const { return this->m_view[item.localId()]; }

  //! Valeur modifiable de l'entité \a item
  ArrayType operator[](ItemLocalId item) { return this->m_view[item.localId()]; }

  //! Valeur non modifiable de la \a i-éme valeur de l'entité \a item
  const DataType operator()(ItemLocalId item,Int32 i) const { return this->m_view.item(item.localId(),i); }

  //! Valeur modifiable de la \a i-éme valeur de l'entité \a item
  DataType& operator()(ItemLocalId item,Int32 i) { return this->m_view[item.localId()][i]; }

 public:

  //! Copie les valeurs de \a v dans cette variable
  ARCANE_CORE_EXPORT void copy(const ItemVariableArrayRefT<DataType>& v);
  //! Copie les valeurs de \a v pour le groupe \a group dans cette variable
  ARCANE_CORE_EXPORT void copy(const ItemVariableArrayRefT<DataType>& v,const ItemGroup& group);
  /*!
   * \brief Copie les valeurs de \a v dans cette variable via la file \a queue.
   *
   * \a queue peut être nul.
   */
  ARCANE_CORE_EXPORT void copy(const ItemVariableArrayRefT<DataType>& v,RunQueue* queue);
  //! Remplit la variable avec la valeur \a value
  ARCANE_CORE_EXPORT void fill(const DataType& value);
  //! Remplit la variable avec la valeur \a value pour les entités du groupe \a group 
  ARCANE_CORE_EXPORT void fill(const DataType& value,const ItemGroup& group);
  //! Remplit la variable avec la valeur \a value via la file \a queue
  ARCANE_CORE_EXPORT void fill(const DataType& value,RunQueue* queue);

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

  //! Valeur non modifiable de l'entité \a item
  ConstArrayType operator[](ItemLocalIdType item) const { return this->m_view[item.localId()]; }

  //! Valeur modifiable de l'entité \a item
  ArrayType operator[](ItemLocalIdType item) { return this->m_view[item.localId()]; }

  //! Valeur non modifiable de la \a i-éme valeur de l'entité \a item
  const DataType operator()(ItemLocalIdType item,Int32 i) const { return this->m_view[item.localId()][i]; }

  //! Valeur modifiable de la \a i-éme valeur de l'entité \a item
  DataType& operator()(ItemLocalIdType item,Int32 i) { return this->m_view[item.localId()][i]; }

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
