// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BuiltinBackend.h                                            (C) 2000-2026 */
/*                                                                           */
/* Builtin backend using CSR matrix.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_BUILTINBACKEND_H
#define ARCCORE_ALINA_BUILTINBACKEND_H
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

//#pragma GCC diagnostic ignored "-Wconversion"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/CSRMatrixOperations.h"
#include "arccore/alina/SkylineLUSolver.h"
#include "arccore/alina/MatrixOperationsImpl.h"

#include "arccore/base/ConcurrencyBase.h"
#include "arccore/common/SmallArray.h"

#include <numeric>
#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * The builtin backend does not have any dependencies, and uses OpenMP for
 * parallelization. Matrices are stored in the CRS format, and vectors are
 * instances of ``std::vector<value_type>``. There is no usual overhead of
 * moving the constructed hierarchy to the builtin backend, since the backend
 * is used internally during setup.
 */
template <typename ValueType, typename ColumnType = ptrdiff_t, typename PointerType = ColumnType>
struct BuiltinBackend
{
  typedef ValueType value_type;
  typedef ColumnType index_type;
  typedef ColumnType col_type;
  typedef PointerType ptr_type;

  typedef typename math::rhs_of<value_type>::type rhs_type;

  struct provides_row_iterator : std::true_type
  {};

  typedef CSRMatrix<value_type, col_type, ptr_type> matrix;
  typedef numa_vector<rhs_type> vector;
  typedef numa_vector<value_type> matrix_diagonal;
  typedef solver::SkylineLUSolver<value_type> direct_solver;

  /// The backend has no parameters.
  typedef Alina::detail::empty_params params;

  static std::string name() { return "builtin"; }

  // Copy matrix. This is a noop for builtin backend.
  static std::shared_ptr<matrix>
  copy_matrix(std::shared_ptr<matrix> A, const params&)
  {
    return A;
  }

  // Copy vector to builtin backend.
  template <class T>
  static std::shared_ptr<numa_vector<T>>
  copy_vector(const std::vector<T>& x, const params&)
  {
    return std::make_shared<numa_vector<T>>(x);
  }

  // Copy vector to builtin backend. This is a noop for builtin backend.
  template <class T>
  static std::shared_ptr<numa_vector<T>>
  copy_vector(std::shared_ptr<numa_vector<T>> x, const params&)
  {
    return x;
  }

  // Create vector of the specified size.
  static std::shared_ptr<vector>
  create_vector(size_t size, const params&)
  {
    return std::make_shared<vector>(size);
  }

  struct gather
  {
    std::vector<col_type> I;

    gather(size_t /*size*/, const std::vector<col_type>& I, const params&)
    : I(I)
    {}

    template <class InVec, class OutVec>
    void operator()(const InVec& vec, OutVec& vals) const
    {
      for (size_t i = 0; i < I.size(); ++i)
        vals[i] = vec[I[i]];
    }
  };

  struct scatter
  {
    std::vector<col_type> I;

    scatter(size_t /*size*/, const std::vector<col_type>& I, const params&)
    : I(I)
    {}

    template <class InVec, class OutVec>
    void operator()(const InVec& vals, OutVec& vec) const
    {
      for (size_t i = 0; i < I.size(); ++i)
        vec[I[i]] = vals[i];
    }
  };

  // Create direct solver for coarse level
  static std::shared_ptr<direct_solver>
  create_solver(std::shared_ptr<matrix> A, const params&)
  {
    return std::make_shared<direct_solver>(*A);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

namespace Arcane::Alina::backend
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
struct is_builtin_vector : std::false_type
{};

template <class V>
struct is_builtin_vector<std::vector<V>> : std::is_arithmetic<V>
{};

template <class V>
struct is_builtin_vector<SmallSpan<V>> : std::true_type
{};

template <class V>
struct is_builtin_vector<Span<V>> : std::true_type
{};

template <class V>
struct is_builtin_vector<numa_vector<V>> : std::true_type
{};

//---------------------------------------------------------------------------
// Specialization of backend interface
//---------------------------------------------------------------------------
template <typename T1, typename T2>
struct backends_compatible<BuiltinBackend<T1>, BuiltinBackend<T2>> : std::true_type
{};

template <typename V, typename C, typename P>
struct rows_impl<CSRMatrix<V, C, P>>
{
  static size_t get(const CSRMatrix<V, C, P>& A)
  {
    return A.nbRow();
  }
};

template <typename V, typename C, typename P>
struct cols_impl<CSRMatrix<V, C, P>>
{
  static size_t get(const CSRMatrix<V, C, P>& A)
  {
    return A.ncols;
  }
};

template <class Vec>
struct bytes_impl<Vec, typename std::enable_if<is_builtin_vector<Vec>::value>::type>
{
  static size_t get(const Vec& x)
  {
    typedef typename backend::value_type<Vec>::type V;
    return x.size() * sizeof(V);
  }
};

template <typename V, typename C, typename P>
struct ptr_data_impl<CSRMatrix<V, C, P>>
{
  typedef const P* type;
  static type get(const CSRMatrix<V, C, P>& A)
  {
    return &A.ptr[0];
  }
};

template <typename V, typename C, typename P>
struct col_data_impl<CSRMatrix<V, C, P>>
{
  typedef const C* type;
  static type get(const CSRMatrix<V, C, P>& A)
  {
    return &A.col[0];
  }
};

template <typename V, typename C, typename P>
struct val_data_impl<CSRMatrix<V, C, P>>
{
  typedef const V* type;
  static type get(const CSRMatrix<V, C, P>& A)
  {
    return &A.val[0];
  }
};

template <typename V, typename C, typename P>
struct nonzeros_impl<CSRMatrix<V, C, P>>
{
  static size_t get(const CSRMatrix<V, C, P>& A)
  {
    return A.nbRow() == 0 ? 0 : A.ptr[A.nbRow()];
  }
};

template <typename V, typename C, typename P>
struct row_nonzeros_impl<CSRMatrix<V, C, P>>
{
  static size_t get(const CSRMatrix<V, C, P>& A, size_t row)
  {
    return A.ptr[row + 1] - A.ptr[row];
  }
};

template <class Vec>
struct clear_impl<Vec, typename std::enable_if<is_builtin_vector<Vec>::value>::type>
{
  static void apply(Vec& x)
  {
    typedef typename backend::value_type<Vec>::type V;

    const size_t n = x.size();
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        x[i] = math::zero<V>();
      }
    });
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Vec1, class Vec2>
struct inner_product_impl<Vec1, Vec2,
                          typename std::enable_if<
                          is_builtin_vector<Vec1>::value &&
                          is_builtin_vector<Vec2>::value>::type>
{
  typedef typename value_type<Vec1>::type V;

  typedef typename math::inner_product_impl<V>::return_type return_type;

  static return_type get(const Vec1& x, const Vec2& y)
  {
    if (ConcurrencyBase::maxAllowedThread() > 1) {
      return parallel(x, y);
    }
    else {
      return serial(x, y);
    }
  }

  static return_type serial(const Vec1& x, const Vec2& y)
  {
    const size_t n = x.size();

    return_type s = math::zero<return_type>();
    return_type c = math::zero<return_type>();

    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      return_type d = math::inner_product(x[i], y[i]) - c;
      return_type t = s + d;
      c = (t - s) - d;
      s = t;
    }

    return s;
  }

  static return_type parallel(const Vec1& x, const Vec2& y)
  {
    const size_t n = x.size();
    // TODO: Use padding to avoir sharing cache line between threads
    SmallArray<return_type, 256> sum_array;
    Int32 nb_thread = ConcurrencyBase::maxAllowedThread();
    const return_type zero = math::zero<return_type>();
    sum_array.resize(nb_thread, zero);
    SmallSpan<return_type> sum = sum_array.view();
    for (Int32 i = 0; i < nb_thread; ++i)
      sum[i] = zero;
    // NOTE GG: NOT reproducible
    
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      const int tid = TaskFactory::currentTaskThreadIndex();
      return_type s = zero;
      return_type c = zero;
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        return_type d = math::inner_product(x[i], y[i]) - c;
        return_type t = s + d;
        c = (t - s) - d;
        s = t;
      }

      sum[tid] += s;
    });
    return_type total = zero;
    for (Int32 i = 0; i < nb_thread; ++i) {
      total += sum[i];
    }
    return total;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class A, class Vec1, class B, class Vec2>
struct axpby_impl<A, Vec1, B, Vec2, typename std::enable_if<is_builtin_vector<Vec1>::value && is_builtin_vector<Vec2>::value>::type>
{
  static void apply(A a, const Vec1& x, B b, Vec2& y)
  {
    const size_t n = x.size();
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      if (!math::is_zero(b)) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          y[i] = a * x[i] + b * y[i];
        }
      }
      else {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          y[i] = a * x[i];
        }
      }
    });
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class A, class Vec1, class B, class Vec2, class C, class Vec3>
struct axpbypcz_impl<A, Vec1, B, Vec2, C, Vec3,
                     typename std::enable_if<
                     is_builtin_vector<Vec1>::value &&
                     is_builtin_vector<Vec2>::value &&
                     is_builtin_vector<Vec3>::value>::type>
{
  static void apply(A a, const Vec1& x, B b, const Vec2& y, C c, Vec3& z)
  {
    const size_t n = x.size();
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      if (!math::is_zero(c)) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          z[i] = a * x[i] + b * y[i] + c * z[i];
        }
      }
      else {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          z[i] = a * x[i] + b * y[i];
        }
      }
    });
  }   
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Alpha, class Vec1, class Vec2, class Beta, class Vec3>
struct vmul_impl<Alpha, Vec1, Vec2, Beta, Vec3,
                 typename std::enable_if<
                 is_builtin_vector<Vec1>::value &&
                 is_builtin_vector<Vec2>::value &&
                 is_builtin_vector<Vec3>::value &&
                 math::static_rows<typename value_type<Vec1>::type>::value == math::static_rows<typename value_type<Vec2>::type>::value &&
                 math::static_rows<typename value_type<Vec1>::type>::value == math::static_rows<typename value_type<Vec3>::type>::value>::type>
{
  static void apply(Alpha a, const Vec1& x, const Vec2& y, Beta b, Vec3& z)
  {
    const size_t n = x.size();
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      if (!math::is_zero(b)) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          z[i] = a * x[i] * y[i] + b * z[i];
        }
      }
      else {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          z[i] = a * x[i] * y[i];
        }
      }
    });
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Support for mixed scalar/nonscalar types
template <class Alpha, class Vec1, class Vec2, class Beta, class Vec3>
struct vmul_impl<Alpha, Vec1, Vec2, Beta, Vec3,
                 typename std::enable_if<is_builtin_vector<Vec1>::value &&
                                         is_builtin_vector<Vec2>::value &&
                                         is_builtin_vector<Vec3>::value &&
                                         (math::static_rows<typename value_type<Vec1>::type>::value != math::static_rows<typename value_type<Vec2>::type>::value ||
                                          math::static_rows<typename value_type<Vec1>::type>::value != math::static_rows<typename value_type<Vec3>::type>::value)>::type>
{
  static void apply(Alpha a, const Vec1& x, const Vec2& y, Beta b, Vec3& z)
  {
    typedef typename value_type<Vec1>::type M_type;
    auto Y = backend::reinterpret_as_rhs<M_type>(y);
    auto Z = backend::reinterpret_as_rhs<M_type>(z);

    const size_t n = x.size();

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      if (!math::is_zero(b)) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          Z[i] = a * x[i] * Y[i] + b * Z[i];
        }
      }
      else {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          Z[i] = a * x[i] * Y[i];
        }
      }
    });
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Vec1, class Vec2>
struct copy_impl<Vec1, Vec2,
                 typename std::enable_if<
                 is_builtin_vector<Vec1>::value &&
                 is_builtin_vector<Vec2>::value>::type>
{
  static void apply(const Vec1& x, Vec2& y)
  {
    const size_t n = x.size();
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        y[i] = x[i];
      }
    });
  }
};

template <class MatrixValue, class Vector, bool IsConst>
struct reinterpret_as_rhs_impl<MatrixValue, Vector, IsConst,
                               typename std::enable_if<is_builtin_vector<Vector>::value>::type>
{
  typedef typename backend::value_type<Vector>::type src_type;
  typedef typename math::scalar_of<src_type>::type scalar_type;
  typedef typename math::rhs_of<MatrixValue>::type rhs_type;
  typedef typename math::replace_scalar<rhs_type, scalar_type>::type dst_type;
  typedef typename std::conditional<IsConst, const dst_type*, dst_type*>::type ptr_type;
  typedef typename std::conditional<IsConst, const dst_type, dst_type>::type return_value_type;
  typedef Span<return_value_type> return_type;

  template <class V>
  static return_type get(V&& x)
  {
    auto ptr = reinterpret_cast<ptr_type>(&x[0]);
    const size_t n = x.size() * sizeof(src_type) / sizeof(dst_type);
    return Span<return_value_type>(ptr, n);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace detail
{

  template <typename V, typename C, typename P>
  struct use_builtin_matrix_ops<CSRMatrix<V, C, P>>
  : std::true_type
  {};

} // namespace detail

} // namespace Arcane::Alina::backend

namespace Arcane::Alina::detail
{

// Backend with scalar value_type of highest precision.
template <class V1, class V2>
struct common_scalar_backend<BuiltinBackend<V1>, BuiltinBackend<V2>,
                             typename std::enable_if<math::static_rows<V1>::value != 1 || math::static_rows<V2>::value != 1>::type>
{
  typedef typename math::scalar_of<V1>::type S1;
  typedef typename math::scalar_of<V2>::type S2;

  typedef typename std::conditional<(sizeof(S1) > sizeof(S2)), BuiltinBackend<S1>, BuiltinBackend<S2>>::type type;
};

} // namespace Arcane::Alina::detail

#endif
