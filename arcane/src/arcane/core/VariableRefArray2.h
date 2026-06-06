// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefArray2.h                                         (C) 2000-2025 */
/*                                                                           */
/* Class managing a reference to a 2D array variable.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEREFARRAY2_H
#define ARCANE_CORE_VARIABLEREFARRAY2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array2View.h"
#include "arcane/core/VariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Two-dimensional array variable.
 *
 * This variable manages classic 2D arrays.
 */
template<typename T>
class VariableRefArray2T
: public VariableRef
, public Array2View<T>
{
 public:
  
  typedef T DataType;
  typedef Array2<T> ValueType;
  typedef ConstArrayView<T> ConstReturnReferenceType;
  typedef ArrayView<T> ReturnReferenceType;
  
  //! Type of the variable elements
  typedef DataType ElementType;
  //! Type of the base class
  typedef VariableRef BaseClass;
  //! Type of the class managing the variable value
  typedef Array2<DataType> ContainerType;
  //! Type of the array used to access the variable
  typedef Array2View<DataType> ArrayBase;

  typedef Array2VariableT<DataType> PrivatePartType;

  typedef VariableRefArray2T<DataType> ThatClass;

 public:

  //! Constructs a reference to a 2D array variable specified in \a vb
  ARCANE_CORE_EXPORT VariableRefArray2T(const VariableBuildInfo& vb);
  //! Constructs a reference from \a rhs
  ARCANE_CORE_EXPORT VariableRefArray2T(const VariableRefArray2T<DataType>& rhs);
  //! Constructs a reference from \a var
  explicit ARCANE_CORE_EXPORT VariableRefArray2T(IVariable* var);
  /*!
   * \brief Copy assignment operator.
   * \deprecated Use refersTo() instead.
   */
  ARCCORE_DEPRECATED_2021("Use refersTo() instead.")
  ARCANE_CORE_EXPORT void operator=(const VariableRefArray2T<DataType>& rhs);
  virtual ARCANE_CORE_EXPORT ~VariableRefArray2T(); //!< Frees resources

 public:

  /*!
   * \brief Reallocates the number of elements in the first dimension of the array.
   *
   * The number of elements in the second dimension is set to zero.
   * \warning reallocation does not preserve previous values.
   */
  virtual ARCANE_CORE_EXPORT void resize(Integer new_size);

  /*!
   * \brief Reallocates the number of elements in the array.
   *
   * Reallocates the array with \a dim1_size as the size of the first
   * dimension and \a dim2_size as the size of the second.
   * \warning reallocation does not preserve previous values.
   */
  ARCANE_CORE_EXPORT void resize(Integer dim1_size,Integer dim2_size);

  //! Fills the variable with the value \a value
  ARCANE_CORE_EXPORT void fill(const DataType& value);

  //! Positions the instance reference to the variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const VariableRefArray2T<DataType>& rhs);

 public:

  virtual bool isArrayVariable() const { return true; }
  virtual Integer arraySize() const { return this->dim1Size(); }
  Integer size() const { return this->dim1Size(); }
  virtual ARCANE_CORE_EXPORT void updateFromInternal();

  /*!
    \brief Returns the container of the variable's values.
    *
    Calling this method is only possible for private variables (PPrivate property).
    */
  ARCCORE_DEPRECATED_2021("Use _internalTrueData() instead.")
  ARCANE_CORE_EXPORT ContainerType& internalContainer();

 public:

  //! \internal
  ARCANE_CORE_EXPORT IArray2DataInternalT<T>* _internalTrueData();

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
