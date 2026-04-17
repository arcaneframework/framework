// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BlockCSRBackend.h                                           (C) 2000-2026 */
/*                                                                           */
/* Sparse matrix in block-CSR format.                         .              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_BLOCKCSRBACKEND_H
#define ARCCORE_ALINA_BLOCKCSRBACKEND_H
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

#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/SkylineLUSolver.h"
#include "arccore/alina/BlockCSRMatrix.h"

#include <algorithm>
#include <numeric>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief block_crs backend definition.
 *
 * \param real Value type.
 * \ingroup backends
 */
template <typename real>
struct BlockCSRBackend
{
  typedef real value_type;
  typedef ptrdiff_t index_type;
  typedef ptrdiff_t col_type;
  typedef ptrdiff_t ptr_type;

  typedef BlockCSRMatrix<real, index_type, index_type> matrix;
  typedef typename BuiltinBackend<real>::vector vector;
  typedef typename BuiltinBackend<real>::vector matrix_diagonal;
  typedef solver::SkylineLUSolver<value_type> direct_solver;

  struct provides_row_iterator : std::false_type
  {};

  /// Backend parameters.
  struct params
  {
    /// Block size to use with the created matrices.
    Int32 block_size;

    explicit params(Int32 block_size = 4)
    : block_size(block_size)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, block_size)
    {
      p.check_params( { "block_size" });
    }
    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, block_size);
    }
  };

  static std::string name() { return "block_crs"; }

  /// Copy matrix from builtin backend.
  static std::shared_ptr<matrix>
  copy_matrix(std::shared_ptr<typename BuiltinBackend<real>::matrix> A,
              const params& prm)
  {
    return std::make_shared<matrix>(*A, prm.block_size);
  }

  /// Copy vector from builtin backend.
  static std::shared_ptr<vector>
  copy_vector(const vector& x, const params&)
  {
    return std::make_shared<vector>(x);
  }

  static std::shared_ptr<vector>
  copy_vector(const std::vector<value_type>& x, const params&)
  {
    return std::make_shared<vector>(x);
  }

  /// Copy vector from builtin backend.
  static std::shared_ptr<vector>
  copy_vector(std::shared_ptr<vector> x, const params&)
  {
    return x;
  }

  /// Create vector of the specified size.
  static std::shared_ptr<vector>
  create_vector(size_t size, const params&)
  {
    return std::make_shared<vector>(size);
  }

  static std::shared_ptr<direct_solver>
  create_solver(std::shared_ptr<typename BuiltinBackend<real>::matrix> A,
                const params&)
  {
    return std::make_shared<direct_solver>(*A);
  }
};

//---------------------------------------------------------------------------
// Specialization of backend interface
//---------------------------------------------------------------------------
template <typename V, typename C, typename P>
struct rows_impl<BlockCSRMatrix<V, C, P>>
{
  static size_t get(const BlockCSRMatrix<V, C, P>& A)
  {
    return A.m_nbRow;
  }
};

template <typename V, typename C, typename P>
struct cols_impl<BlockCSRMatrix<V, C, P>>
{
  static size_t get(const BlockCSRMatrix<V, C, P>& A)
  {
    return A.ncols;
  }
};

template <typename V, typename C, typename P>
struct nonzeros_impl<BlockCSRMatrix<V, C, P>>
{
  static size_t get(const BlockCSRMatrix<V, C, P>& A)
  {
    return A.ptr.back() * A.block_size * A.block_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Alpha, typename Beta, typename V, typename C, typename P, class Vec1, class Vec2>
struct spmv_impl<Alpha, BlockCSRMatrix<V, C, P>, Vec1, Beta, Vec2>
{
  typedef BlockCSRMatrix<V, C, P> matrix;

  static void apply(Alpha alpha, const matrix& A, const Vec1& x, Beta beta, Vec2& y)
  {
    const size_t nb = A.brows;
    const size_t na = A.nrows;
    const size_t ma = A.ncols;
    const size_t b1 = A.block_size;
    const size_t b2 = b1 * b1;

    if (!math::is_zero(beta)) {
      if (beta != 1) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(na); ++i) {
          y[i] *= beta;
        }
      }
    }
    else {
      backend::clear(y);
    }

#pragma omp parallel for
    for (ptrdiff_t ib = 0; ib < static_cast<ptrdiff_t>(nb); ++ib) {
      for (P jb = A.ptr[ib], eb = A.ptr[ib + 1]; jb < eb; ++jb) {
        size_t x0 = A.col[jb] * b1;
        size_t y0 = ib * b1;
        block_prod(b1, std::min(b1, ma - x0), std::min(b1, na - y0),
                   alpha, &A.val[jb * b2], &x[x0], &y[y0]);
      }
    }
  }

  static void block_prod(size_t dim, size_t nx, size_t ny,
                         Alpha alpha, const V* A, const V* x, V* y)
  {
    for (size_t i = 0; i < ny; ++i, ++y) {
      const V* xx = x;
      V sum = 0;
      for (size_t j = 0; j < dim; ++j, ++A, ++xx)
        if (j < nx)
          sum += (*A) * (*xx);
      *y += alpha * sum;
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename V, typename C, typename P, class Vec1, class Vec2, class Vec3>
struct residual_impl<BlockCSRMatrix<V, C, P>, Vec1, Vec2, Vec3>
{
  typedef BlockCSRMatrix<V, C, P> matrix;

  static void apply(const Vec1& rhs, const matrix& A, const Vec2& x, Vec3& r)
  {
    typedef typename math::scalar_of<V>::type S;
    const auto one = math::identity<S>();
    backend::copy(rhs, r);
    backend::spmv(-one, A, x, one, r);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::backend

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
