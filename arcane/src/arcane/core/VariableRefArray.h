// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefArray.h                                          (C) 2000-2025 */
/*                                                                           */
/* Class managing a reference to a 1D array variable.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEREFARRAY_H
#define ARCANE_CORE_VARIABLEREFARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> class VariableRefArrayLockT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Array variable.
 */
template <typename T>
class VariableRefArrayT
: public VariableRef
, public ArrayView<T>
{
 public:

  typedef T DataType;
  typedef Array<T> ValueType;
  typedef typename ArrayView<T>::const_reference ConstReturnReferenceType;
  typedef typename ArrayView<T>::reference ReturnReferenceType;

  //! Type of the variable elements
  typedef DataType ElementType;
  //! Type of the base class
  typedef VariableRef BaseClass;
  //! Type of the class managing the variable value
  typedef Array<DataType> ContainerType;
  //! Type of the array allowing access to the variable
  typedef ArrayView<DataType> ArrayBase;

  typedef VariableArrayT<DataType> PrivatePartType;

  typedef VariableRefArrayT<DataType> ThatClass;

  typedef VariableRefArrayLockT<DataType> LockType;

 public:

  //! Constructs a reference to a 1D array variable specified in \a vb
  ARCANE_CORE_EXPORT explicit VariableRefArrayT(const VariableBuildInfo& vb);
  //! Constructs a reference from \a rhs
  ARCANE_CORE_EXPORT VariableRefArrayT(const VariableRefArrayT<DataType>& rhs);
  //! Constructs a reference from \a var
  explicit ARCANE_CORE_EXPORT VariableRefArrayT(IVariable* var);

  //! Positions the instance's reference to the variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const VariableRefArrayT<DataType>& rhs);

  ARCANE_CORE_EXPORT ~VariableRefArrayT() override; //!< Frees resources

 public:

  //! Resizes the array to contain \a new_size elements
  virtual ARCANE_CORE_EXPORT void resize(Integer new_size);

  //! Resizes the array to contain \a new_size elements
  virtual ARCANE_CORE_EXPORT void resizeWithReserve(Integer new_size, Integer nb_additional);

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
    \brief Returns the container of this variable's values.
    *
    Calling this method is only possible for private variables
    (PPrivate property). For others, an exception is
    raised (you must use lock()).
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
