// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MatrixOperationsImpl.h                                      (C) 2000-2026 */
/*                                                                           */
/* Sparse matrix operations for matrices that provide row_iterator.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_MATRIXOPERATIONSIMPL_H
#define ARCCORE_ALINA_MATRIXOPERATIONSIMPL_H
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

#include "arccore/alina/BackendInterface.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace detail
{

  template <class Matrix, class Enable = void>
  struct use_builtin_matrix_ops : std::false_type
  {};

} // namespace detail

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Alpha, class Matrix, class Vector1, class Beta, class Vector2>
struct spmv_impl<Alpha, Matrix, Vector1, Beta, Vector2,
                 std::enable_if_t<
                 detail::use_builtin_matrix_ops<Matrix>::value &&
                 math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector1>::type>::value &&
                 math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector2>::type>::value>>
{
  static void apply(Alpha alpha, const Matrix& A, const Vector1& x, Beta beta, Vector2& y)
  {
    typedef typename value_type<Vector2>::type V;

    const ptrdiff_t n = static_cast<ptrdiff_t>(nbRow(A));

    if (!math::is_zero(beta)) {
#pragma omp parallel for
      for (ptrdiff_t i = 0; i < n; ++i) {
        V sum = math::zero<V>();
        for (typename row_iterator<Matrix>::type a = row_begin(A, i); a; ++a)
          sum += a.value() * x[a.col()];
        y[i] = alpha * sum + beta * y[i];
      }
    }
    else {
#pragma omp parallel for
      for (ptrdiff_t i = 0; i < n; ++i) {
        V sum = math::zero<V>();
        for (typename row_iterator<Matrix>::type a = row_begin(A, i); a; ++a)
          sum += a.value() * x[a.col()];
        y[i] = alpha * sum;
      }
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Matrix, class Vector1, class Vector2, class Vector3>
struct residual_impl<Matrix, Vector1, Vector2, Vector3,
                     std::enable_if_t<detail::use_builtin_matrix_ops<Matrix>::value &&
                                      math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector1>::type>::value &&
                                      math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector2>::type>::value &&
                                      math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector3>::type>::value>>
{
  static void apply(Vector1 const& rhs,
                    Matrix const& A,
                    Vector2 const& x,
                    Vector3& res)
  {
    typedef typename value_type<Vector3>::type V;
    const ptrdiff_t n = static_cast<ptrdiff_t>(nbRow(A));

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin+size); ++i) {
        V sum = math::zero<V>();
        for (typename row_iterator<Matrix>::type a = row_begin(A, i); a; ++a)
          sum += a.value() * x[a.col()];
        res[i] = rhs[i] - sum;
      }
    });
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Allows to do matrix-vector products with mixed scalar/nonscalar types.
 * Reinterprets pointers to the vectors data into appropriate types.
 */
template <class Alpha, class Matrix, class Vector1, class Beta, class Vector2>
struct spmv_impl<Alpha, Matrix, Vector1, Beta, Vector2,
                 std::enable_if_t<detail::use_builtin_matrix_ops<Matrix>::value &&
                                  (math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector1>::type>::value ||
                                   math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector2>::type>::value)>>
{
  static void apply(Alpha alpha, const Matrix& A, const Vector1& x, Beta beta, Vector2& y)
  {
    typedef typename value_type<Matrix>::type V;

    auto X = backend::reinterpret_as_rhs<V>(x);
    auto Y = backend::reinterpret_as_rhs<V>(y);

    spmv(alpha, A, X, beta, Y);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Matrix, class Vector1, class Vector2, class Vector3>
struct residual_impl<Matrix, Vector1, Vector2, Vector3,
                     std::enable_if_t<detail::use_builtin_matrix_ops<Matrix>::value &&
                                      (math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector1>::type>::value ||
                                       math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector2>::type>::value ||
                                       math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector3>::type>::value)>>
{
  static void apply(Vector1 const& f,
                    Matrix const& A,
                    Vector2 const& x,
                    Vector3& r)
  {
    typedef typename value_type<Matrix>::type V;

    auto X = backend::reinterpret_as_rhs<V>(x);
    auto F = backend::reinterpret_as_rhs<V>(f);
    auto R = backend::reinterpret_as_rhs<V>(r);

    residual(F, A, X, R);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::backend

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
