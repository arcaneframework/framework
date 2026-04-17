// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ValueTypeInterface.h                                        (C) 2000-2026 */
/*                                                                           */
/* Enable statically sized matrices as value types.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_STATICMATRIX_H
#define ARCCORE_ALINA_STATICMATRIX_H
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

#include <array>
#include <type_traits>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/DenseMatrixInverseImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, int N, int M>
struct StaticMatrix
{
  std::array<T, N * M> buf;

  T operator()(int i, int j) const
  {
    return buf[i * M + j];
  }

  T& operator()(int i, int j)
  {
    return buf[i * M + j];
  }

  T operator()(int i) const
  {
    return buf[i];
  }

  T& operator()(int i)
  {
    return buf[i];
  }

  const T* data() const
  {
    return buf.data();
  }

  T* data()
  {
    return buf.data();
  }

  template <typename U>
  const StaticMatrix& operator=(const StaticMatrix<U, N, M>& y)
  {
    for (int i = 0; i < N * M; ++i)
      buf[i] = y.buf[i];
    return *this;
  }

  template <typename U>
  const StaticMatrix& operator+=(const StaticMatrix<U, N, M>& y)
  {
    for (int i = 0; i < N * M; ++i)
      buf[i] += y.buf[i];
    return *this;
  }

  template <typename U>
  const StaticMatrix& operator-=(const StaticMatrix<U, N, M>& y)
  {
    for (int i = 0; i < N * M; ++i)
      buf[i] -= y.buf[i];
    return *this;
  }

  const StaticMatrix& operator*=(T c)
  {
    for (int i = 0; i < N * M; ++i)
      buf[i] *= c;
    return *this;
  }

  friend StaticMatrix operator*(T a, StaticMatrix x)
  {
    return x *= a;
  }

  friend StaticMatrix operator-(StaticMatrix x)
  {
    for (int i = 0; i < N * M; ++i)
      x.buf[i] = -x.buf[i];
    return x;
  }

  friend bool operator<(const StaticMatrix& x, const StaticMatrix& y)
  {
    T xtrace = math::zero<T>();
    T ytrace = math::zero<T>();

    const int K = N < M ? N : M;

    for (int i = 0; i < K; ++i) {
      xtrace += x(i, i);
      ytrace += y(i, i);
    }

    return xtrace < ytrace;
  }

  friend std::ostream& operator<<(std::ostream& os, const StaticMatrix& a)
  {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        os << " " << a(i, j);
      }
      os << std::endl;
    }
    return os;
  }
};

template <typename T, typename U, int N, int M>
StaticMatrix<T, N, M> operator+(StaticMatrix<T, N, M> a, const StaticMatrix<U, N, M>& b)
{
  return a += b;
}

template <typename T, typename U, int N, int M>
StaticMatrix<T, N, M> operator-(StaticMatrix<T, N, M> a, const StaticMatrix<U, N, M>& b)
{
  return a -= b;
}

template <typename T, typename U, int N, int K, int M>
StaticMatrix<T, N, M> operator*(const StaticMatrix<T, N, K>& a, const StaticMatrix<U, K, M>& b)
{
  StaticMatrix<T, N, M> c;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j)
      c(i, j) = math::zero<T>();
    for (int k = 0; k < K; ++k) {
      T aik = a(i, k);
      for (int j = 0; j < M; ++j)
        c(i, j) += aik * b(k, j);
    }
  }
  return c;
}

template <class T> struct is_static_matrix : std::false_type
{};

template <class T, int N, int M>
struct is_static_matrix<StaticMatrix<T, N, M>> : std::true_type
{};

} // namespace Arcane::Alina

namespace Arcane::Alina::backend
{

/// Enable static matrix as a value-type.
template <typename T, int N, int M>
struct is_builtin_vector<std::vector<StaticMatrix<T, N, M>>> : std::true_type
{};

} // namespace Arcane::Alina::backend

namespace Arcane::Alina::math
{

/// Scalar type of a non-scalar type.
template <class T, int N, int M>
struct scalar_of<StaticMatrix<T, N, M>>
{
  typedef typename scalar_of<T>::type type;
};

/// Replace scalar type in the static matrix.
template <class T, int N, int M, class S>
struct replace_scalar<StaticMatrix<T, N, M>, S>
{
  typedef StaticMatrix<S, N, M> type;
};

/// RHS type corresponding to a non-scalar type.
template <class T, int N>
struct rhs_of<StaticMatrix<T, N, N>>
{
  typedef StaticMatrix<T, N, 1> type;
};

/// Element type of a non-scalar type
template <class T, int N, int M>
struct element_of<StaticMatrix<T, N, M>>
{
  typedef T type;
};

/// Whether the value type is a statically sized matrix.
template <class T, int N, int M>
struct is_static_matrix<StaticMatrix<T, N, M>> : std::true_type
{};

/// Number of rows for statically sized matrix types.
template <class T, int N, int M>
struct static_rows<StaticMatrix<T, N, M>> : std::integral_constant<int, N>
{};

/// Number of columns for statically sized matrix types.
template <class T, int N, int M>
struct static_cols<StaticMatrix<T, N, M>> : std::integral_constant<int, M>
{};

/// Specialization of conjugate transpose for static matrices.
template <typename T, int N, int M>
struct adjoint_impl<StaticMatrix<T, N, M>>
{
  typedef StaticMatrix<T, M, N> return_type;

  static StaticMatrix<T, M, N> get(const StaticMatrix<T, N, M>& x)
  {
    StaticMatrix<T, M, N> y;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        y(j, i) = math::adjoint(x(i, j));
    return y;
  }
};

/// Inner-product result of two static vectors.
template <class T, int N>
struct inner_product_impl<StaticMatrix<T, N, 1>>
{
  typedef T return_type;
  static T get(const StaticMatrix<T, N, 1>& x, const StaticMatrix<T, N, 1>& y)
  {
    T sum = math::zero<T>();
    for (int i = 0; i < N; ++i)
      sum += x(i) * math::adjoint(y(i));
    return sum;
  }
};

/// Inner-product result of two static matrices.
template <class T, int N, int M>
struct inner_product_impl<StaticMatrix<T, N, M>>
{
  typedef StaticMatrix<T, M, M> return_type;

  static return_type get(const StaticMatrix<T, N, M>& x, const StaticMatrix<T, N, M>& y)
  {
    StaticMatrix<T, M, M> p;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < M; ++j) {
        T sum = math::zero<T>();
        for (int k = 0; k < N; ++k)
          sum += x(k, i) * math::adjoint(y(k, j));
        p(i, j) = sum;
      }
    }
    return p;
  }
};

/// Implementation of Frobenius norm for static matrices.
template <typename T, int N, int M>
struct norm_impl<StaticMatrix<T, N, M>>
{
  static typename math::scalar_of<T>::type get(const StaticMatrix<T, N, M>& x)
  {
    T s = math::zero<T>();
    for (int i = 0; i < N * M; ++i)
      s += x(i) * math::adjoint(x(i));
    return sqrt(math::norm(s));
  }
};

/// Specialization of zero element for static matrices.
template <typename T, int N, int M>
struct zero_impl<StaticMatrix<T, N, M>>
{
  static StaticMatrix<T, N, M> get()
  {
    StaticMatrix<T, N, M> z;
    for (int i = 0; i < N * M; ++i)
      z(i) = math::zero<T>();
    return z;
  }
};

/// Specialization of zero element for static matrices.
template <typename T, int N, int M>
struct is_zero_impl<StaticMatrix<T, N, M>>
{
  static bool get(const StaticMatrix<T, N, M>& x)
  {
    for (int i = 0; i < N * M; ++i)
      if (!math::is_zero(x(i)))
        return false;
    return true;
  }
};

/// Specialization of identity for static matrices.
template <typename T, int N>
struct identity_impl<StaticMatrix<T, N, N>>
{
  static StaticMatrix<T, N, N> get()
  {
    StaticMatrix<T, N, N> I;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        I(i, j) = static_cast<T>(i == j);
    return I;
  }
};

/// Specialization of constant for static matrices.
template <typename T, int N, int M>
struct constant_impl<StaticMatrix<T, N, M>>
{
  static StaticMatrix<T, N, M> get(T c)
  {
    StaticMatrix<T, N, M> C;
    for (int i = 0; i < N * M; ++i)
      C(i) = c;
    return C;
  }
};

/// Specialization of inversion for static matrices.
template <typename T, int N>
struct inverse_impl<StaticMatrix<T, N, N>>
{
  static StaticMatrix<T, N, N> get(StaticMatrix<T, N, N> A)
  {
    std::array<T, N * N> buf;
    std::array<int, N> p;
    detail::inverse(N, A.data(), buf.data(), p.data());
    return A;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
