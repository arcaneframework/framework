// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueTypeComplex.h                                          (C) 2000-2026 */
/*                                                                           */
/* Enable std::complex<T> as value type.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_VALUETYPECOMPLEX_H
#define ARCCORE_ALINA_VALUETYPECOMPLEX_H
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

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/ValueTypeInterface.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

/// Enable std::complex as a value-type.
template <typename T>
struct is_builtin_vector<std::vector<std::complex<T>>> : std::true_type
{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::backend

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Specialization that extracts the scalar type of a complex type.
template <class T>
struct scalar_of<std::complex<T>>
{
  typedef T type;
};

/// Replace scalar type in the complex type.
template <class T, class S>
struct replace_scalar<std::complex<T>, S>
{
  typedef std::complex<S> type;
};

/// Specialization of conjugate transpose for scalar complex arguments.
template <typename T>
struct adjoint_impl<std::complex<T>>
{
  typedef std::complex<T> return_type;

  static std::complex<T> get(std::complex<T> x)
  {
    return std::conj(x);
  }
};

/*!
 * \brief Default implementation for inner product.
 *
 * \note Used in adjoint().
 */
template <typename T>
struct inner_product_impl<std::complex<T>>
{
  typedef std::complex<T> return_type;

  static return_type get(std::complex<T> x, std::complex<T> y)
  {
    return x * std::conj(y);
  }
};

//! Specialization of constant element for complex type.
template <typename T>
struct constant_impl<std::complex<T>>
{
  static std::complex<T> get(T c)
  {
    return std::complex<T>(c, c);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace std
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//GG: not sure it is legal to specialize in namespace std
template <typename V>
bool operator<(const std::complex<V>& a, const std::complex<V>& b)
{
  return std::abs(a) < std::abs(b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace std

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
