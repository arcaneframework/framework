// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefArray.h                                          (C) 2000-2021 */
/*                                                                           */
/* Classe gérant une référence sur une variable tableau 1D.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEREFARRAY_H
#define ARCANE_VARIABLEREFARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> class VariableRefArrayLockT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Variable tableau.
 */
template<typename T>
class VariableRefArrayT
: public VariableRef
, public ArrayView<T>
{
 public:

  typedef T DataType;
  typedef Array<T> ValueType;
  typedef typename ArrayView<T>::const_reference ConstReturnReferenceType;
  typedef typename ArrayView<T>::reference ReturnReferenceType;

  //! Type des éléments de la variable
  typedef DataType ElementType;
  //! Type de la classe de base
  typedef VariableRef BaseClass;
  //! Type de la classe gérant la valeur de la variable
  typedef Array<DataType> ContainerType;
  //! Type du tableau permettant d'accéder à la variable
  typedef ArrayView<DataType> ArrayBase;

  typedef VariableArrayT<DataType> PrivatePartType;

  typedef VariableRefArrayT<DataType> ThatClass;

  typedef VariableRefArrayLockT<DataType> LockType;

 public:

  //! Construit une référence à une variable tableau spécifiée dans \a vb
  ARCANE_CORE_EXPORT explicit VariableRefArrayT(const VariableBuildInfo& vb);
  //! Construit une référence à partir de \a rhs
  ARCANE_CORE_EXPORT VariableRefArrayT(const VariableRefArrayT<DataType>& rhs);
  //! Construit une référence à partir de \a var
  explicit ARCANE_CORE_EXPORT VariableRefArrayT(IVariable* var);

  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const VariableRefArrayT<DataType>& rhs);

  ARCANE_CORE_EXPORT ~VariableRefArrayT() override; //!< Libère les ressources

 public:

  //! Redimensionne le tableau pour contenir \a new_size éléments
  virtual ARCANE_CORE_EXPORT void resize(Integer new_size);

  //! Redimensionne le tableau pour contenir \a new_size éléments
  virtual ARCANE_CORE_EXPORT void resizeWithReserve(Integer new_size,Integer nb_additional);

 public:

  virtual bool isArrayVariable() const { return true; }
  Integer arraySize() const override { return this->size(); }
  ARCANE_CORE_EXPORT void updateFromInternal() override;

 public:

  ArrayView<DataType> asArray() { return (*this); }
  ConstArrayView<DataType> asArray() const { return (*this); }

 public:

  ARCCORE_DEPRECATED_2021("This method is internal to Arcane")
  LockType ARCANE_CORE_EXPORT lock();

 public:

  /*!
    \brief Retourne le conteneur des valeurs de cette variable.
    *
    L'appel à cette méthode n'est possible que pour les variables
    privées (propriété PPrivate). Pour les autres, une exception est
    levée (il faut utiliser lock()).
    */
  ARCCORE_DEPRECATED_2021("Use _internalTrueData() instead.")
  ARCANE_CORE_EXPORT ContainerType& internalContainer();

 public:

  //! \internal
  ARCANE_CORE_EXPORT IArrayDataInternalT<T>* _internalTrueData();

 public:

  static ARCANE_CORE_EXPORT VariableTypeInfo _internalVariableTypeInfo();
  static ARCANE_CORE_EXPORT VariableInfo _internalVariableInfo(const VariableBuildInfo& vbi);

 private:

  PrivatePartType* m_private_part;

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
