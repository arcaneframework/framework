// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MathNumeric.h                                               (C) 2000-2026 */
/*                                                                           */
/* Mathematical operations on numeric types (Real2, Real3, NumVector, ...)   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_MATHNUMERIC_H
#define ARCCORE_BASE_MATHNUMERIC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/NumVector.h"
#include "arccore/base/NumMatrix.h"

#include "arccore/base/MathReal2.h"
#include "arccore/base/MathReal3.h"
#include "arccore/base/MathReal2x2.h"
#include "arccore/base/MathReal3x3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compares the vector with the vector zero.
 *
 * The matrix is nearly zero if and only if each of its components
 * is less than a given epsilon. The epsilon value used is that
 * of FloatInfo<DataType>::nearlyEpsilon():
 * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon,|A.z|<\epsilon \f]
 */
template <typename DataType, int Size> constexpr ARCCORE_HOST_DEVICE bool
isNearlyZero(const NumVector<DataType, Size>& v)
{
  bool is_nearly_zero = true;
  for (int i = 0; i < Size; ++i)
    is_nearly_zero = is_nearly_zero && math::isNearlyZero(v[i]);
  return is_nearly_zero;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compares the matrix with the zero matrix.
 *
 * The matrix is zero if and only if each of its components
 * is less than a given epsilon. The epsilon value used is that
 * of float_info<value_type>::nearlyEpsilon():
 * \f[A=0 \Leftrightarrow |A.x|<\epsilon,|A.y|<\epsilon,|A.z|<\epsilon \f]
 *
 * \retval true if the matrix is equal to the zero matrix,
 * \retval false otherwise.
 */
template <typename DataType, int RowSize, int ColumnSize> constexpr ARCCORE_HOST_DEVICE bool
isNearlyZero(const NumMatrix<DataType, RowSize, ColumnSize>& v)
{
  bool is_nearly_zero = true;
  for (int i = 0; i < RowSize; ++i)
    is_nearly_zero = is_nearly_zero && math::isNearlyZero(v.row(i));
  return is_nearly_zero;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Returns the square of the L2 norm of the vector
template <typename DataType, int Size> constexpr ARCCORE_HOST_DEVICE DataType
squareNormL2(const NumVector<DataType, Size>& v)
{
  DataType norm = {};
  for (int i = 0; i < Size; ++i)
    norm += v[i] * v[i];
  return norm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Returns the L2 norm of the vector
template <typename DataType, int Size> ARCCORE_HOST_DEVICE Real
normL2(const NumVector<DataType, Size>& v)
{
  return Arcane::math::sqrt(squareNormL2(v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
