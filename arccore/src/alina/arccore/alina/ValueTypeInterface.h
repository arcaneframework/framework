// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueTypeInterface.h                                        (C) 2000-2026 */
/*                                                                           */
/* Support for various value types.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_VALUETYPEINTERFACE_H
#define ARCCORE_ALINA_VALUETYPEINTERFACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <type_traits>
#include <complex>

namespace Arcane::Alina::math
{

/// Scalar type of a non-scalar type.
template <class T, class Enable = void>
struct scalar_of
{
  typedef T type;
};

/// RHS type corresponding to a non-scalar type.
template <class T, class Enable = void>
struct rhs_of
{
  typedef T type;
};

/// Element type of a non-scalar type
template <class T, class Enable = void>
struct element_of
{
  typedef T type;
};

/// Replace scalar type in the static matrix
template <class T, class S, class Enable = void>
struct replace_scalar
{
  typedef S type;
};

/// Whether the value type is a statically sized matrix.
template <class T, class Enable = void>
struct is_static_matrix : std::false_type
{};

/// Number of rows for statically sized matrix types.
template <class T, class Enable = void>
struct static_rows : std::integral_constant<int, 1>
{};

/// Number of columns for statically sized matrix types.
template <class T, class Enable = void>
struct static_cols : std::integral_constant<int, 1>
{};

/// Default implementation for conjugate transpose.
/** \note Used in adjoint() */
template <typename ValueType, class Enable = void>
struct adjoint_impl
{
  typedef ValueType return_type;

  static ValueType get(ValueType x)
  {
    return x;
  }
};

/// Default implementation for inner product
/** \note Used in inner_product() */
template <typename ValueType, class Enable = void>
struct inner_product_impl
{
  typedef ValueType return_type;

  static return_type get(ValueType x, ValueType y)
  {
    return x * y;
  }
};

/// Default implementation for element norm.
/** \note Used in zero() */
template <typename ValueType, class Enable = void>
struct norm_impl
{
  static typename scalar_of<ValueType>::type get(ValueType x)
  {
    return std::abs(x);
  }
};

/// Default implementation for the zero element.
/** \note Used in zero() */
template <typename ValueType, class Enable = void>
struct zero_impl
{
  static ValueType get()
  {
    return static_cast<ValueType>(0);
  }
};

/// Default implementation for zero check.
/** \note Used in is_zero() */
template <typename ValueType, class Enable = void>
struct is_zero_impl
{
  static bool get(const ValueType& x)
  {
    return x == zero_impl<ValueType>::get();
  }
};

/// Default implementation for the identity element.
/** \note Used in identity() */
template <typename ValueType, class Enable = void>
struct identity_impl
{
  static ValueType get()
  {
    return static_cast<ValueType>(1);
  }
};

/// Default implementation for the constant element.
/** \note Used in constant() */
template <typename ValueType, class Enable = void>
struct constant_impl
{
  static ValueType get(typename scalar_of<ValueType>::type c)
  {
    return static_cast<ValueType>(c);
  }
};

/// Default implementation of inversion operation.
/** \note Used in inverse() */
template <typename ValueType, class Enable = void>
struct inverse_impl
{
  static ValueType get(const ValueType& x)
  {
    return identity_impl<ValueType>::get() / x;
  }
};

/// Return conjugate transpose of argument.
template <typename ValueType>
typename adjoint_impl<ValueType>::return_type
adjoint(ValueType x)
{
  return adjoint_impl<ValueType>::get(x);
}

/// Return inner product of two arguments.
template <typename ValueType>
typename inner_product_impl<ValueType>::return_type
inner_product(ValueType x, ValueType y)
{
  return inner_product_impl<ValueType>::get(x, y);
}

/// Compute norm of an element.
template <typename ValueType>
typename scalar_of<ValueType>::type norm(const ValueType& a)
{
  return norm_impl<ValueType>::get(a);
}

/// Create zero element of type ValueType.
template <typename ValueType>
ValueType zero()
{
  return zero_impl<ValueType>::get();
}

/// Return true if argument is considered zero.
template <typename ValueType>
bool is_zero(const ValueType& x)
{
  return is_zero_impl<ValueType>::get(x);
}

/// Create identity of type ValueType.
template <typename ValueType>
ValueType identity()
{
  return identity_impl<ValueType>::get();
}

/// Create one element of type ValueType.
template <typename ValueType>
ValueType constant(typename scalar_of<ValueType>::type c)
{
  return constant_impl<ValueType>::get(c);
}

/// Return inverse of the argument.
template <typename ValueType>
ValueType inverse(const ValueType& x)
{
  return inverse_impl<ValueType>::get(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
