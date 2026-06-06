// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRefScalar.h                                         (C) 2000-2025 */
/*                                                                           */
/* Class managing a reference to a scalar variable.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEREFSCALAR_H
#define ARCANE_CORE_VARIABLEREFSCALAR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableRef.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/MathUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class VariableFactoryRegisterer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reference to a scalar variable.
 *
 * The operator operator()() allows access to the variable's value for
 * read-only purposes. To modify the variable's value, you must
 * use the assign() method or the operator operator=(). Note that
 * assignment triggers a reference update and can be costly.
 */
template<typename DataType>
class VariableRefScalarT
: public VariableRef
{
 public:

  //! Type of the variable elements
  typedef DataType ElementType;
  //! Type of the base class
  typedef VariableRef BaseClass;

  typedef VariableScalarT<DataType> PrivatePartType;

  typedef VariableRefScalarT<DataType> ThatClass;

 public:

  //! Constructs a reference to a scalar variable specified in \a vb
  explicit ARCANE_CORE_EXPORT VariableRefScalarT(const VariableBuildInfo& b);
  //! Constructs a reference from \a rhs
  ARCANE_CORE_EXPORT VariableRefScalarT(const VariableRefScalarT<DataType>& rhs);
  //! Constructs a reference from \a var
  explicit ARCANE_CORE_EXPORT VariableRefScalarT(IVariable* var);
  //! Positions the instance's reference to the variable \a rhs.
  ARCANE_CORE_EXPORT void refersTo(const VariableRefScalarT<DataType>& rhs);

#ifdef ARCANE_DOTNET
 public:
#else
 protected:
#endif

  //! Default constructor
  VariableRefScalarT() : m_private_part(nullptr) {}

 public:

  virtual bool isArrayVariable() const { return false; }
  virtual Integer arraySize() const { return 0; }
  virtual ARCANE_CORE_EXPORT void updateFromInternal();

 public:

  ArrayView<DataType> asArray() { return ArrayView<DataType>(1,&(m_private_part->value())); }
  ConstArrayView<DataType> asArray() const { return ConstArrayView<DataType>(1,&(m_private_part->value())); }

 public:
	
  void operator=(const DataType& v) { assign(v); }
  VariableRefScalarT<DataType>& operator=(const VariableRefScalarT<DataType>& v)
  {
    assign(v());
    return (*this);
  }

  //! Resets the variable to its default value
  void reset() { assign(DataType()); }

  //! Scalar value
  const DataType& operator()() const { return m_private_part->value(); }

  //! Scalar value
  const DataType& value() const { return m_private_part->value(); }

  /*!
   * \brief Compares the variable with the value \a v.
   */
  bool isEqual(const DataType& v) const
    { return math::isEqual(m_private_part->value(),v); }

  /*!
   * \brief Compares the variable with the value 0.
   * \sa isEqual().
   */
  bool isZero() const
    { return math::isZero(m_private_part->value()); }

  /*!
   * \brief Compares the variable with the value \a v.
   *
   * For a floating-point type, the comparison is done within an epsilon,
   * defined in float_info<T>::nearlyEpsilon().
   */
  bool isNearlyEqual(const DataType& v) const
    { return math::isNearlyEqual(m_private_part->value(),v); }
  /*!
   * \brief Compares the variable with the value 0.
   * \sa isEqual().
   */
  bool isNearlyZero() const
    { return math::isNearlyZero(m_private_part->value()); }

  //! Assigns the value \a v to the variable
  ARCANE_CORE_EXPORT void assign(const DataType& v);

  //! Performs a type \a type reduction on the variable
  ARCANE_CORE_EXPORT void reduce(Parallel::eReduceType type);

  ARCANE_CORE_EXPORT void swapValues(VariableRefScalarT<DataType>& rhs);

 protected:
  
 private:

  PrivatePartType* m_private_part;

 private:

  static VariableFactoryRegisterer m_auto_registerer;
  static VariableTypeInfo _buildVariableTypeInfo();
  static VariableInfo _buildVariableInfo(const VariableBuildInfo& vbi);
  static VariableRef* _autoCreate(const VariableBuildInfo& vb);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
