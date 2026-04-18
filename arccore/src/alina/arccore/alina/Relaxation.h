// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Relaxation.h                                                (C) 2000-2026 */
/*                                                                           */
/* Various relaxation algorithms.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_RELAXATION_H
#define ARCCORE_ALINA_RELAXATION_H
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

#include <vector>
#include <memory>
#include <numeric>
#include <cmath>
#include <deque>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "arccore/alina/BackendInterface.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/QRFactorizationImpl.h"
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/DenseMatrixInverseImpl.h"
#include "arccore/alina/BackendInterface.h"
#include "arccore/alina/ILUSolverImpl.h"
#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/RelaxationBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converts input matrix to block format before constructing an AMG smoother.
 */
template <class BlockBackend, template <class> class Relax>
struct RelaxationAsBlock
{
  typedef typename BlockBackend::value_type BlockType;

  template <class Backend>
  class type
  {
   public:

    typedef Backend backend_type;

    typedef Relax<BlockBackend> Base;

    typedef typename Backend::matrix matrix;
    typedef typename Backend::vector vector;
    typedef typename Base::params params;
    typedef typename Backend::params backend_params;

    typedef typename Backend::value_type value_type;
    typedef typename Backend::col_type col_type;
    typedef typename Backend::ptr_type ptr_type;
    typedef typename BuiltinBackend<value_type, col_type, ptr_type>::matrix build_matrix;

    template <class Matrix>
    type(const Matrix& A,
         const params& prm = params(),
         const backend_params& bprm = backend_params())
    : base(*std::make_shared<CSRMatrix<BlockType, col_type, ptr_type>>(adapter::block_matrix<BlockType>(A)), prm, bprm)
    , nrows(backend::nbRow(A) / math::static_rows<BlockType>::value)
    {}

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_pre(const Matrix& A,
                   const VectorRHS& rhs,
                   VectorX& x,
                   VectorTMP& tmp) const
    {
      auto F = backend::reinterpret_as_rhs<BlockType>(rhs);
      auto X = backend::reinterpret_as_rhs<BlockType>(x);
      auto T = backend::reinterpret_as_rhs<BlockType>(tmp);
      base.apply_pre(A, F, X, T);
    }

    template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
    void apply_post(const Matrix& A,
                    const VectorRHS& rhs,
                    VectorX& x,
                    VectorTMP& tmp) const
    {
      auto F = backend::reinterpret_as_rhs<BlockType>(rhs);
      auto X = backend::reinterpret_as_rhs<BlockType>(x);
      auto T = backend::reinterpret_as_rhs<BlockType>(tmp);
      base.apply_post(A, F, X, T);
    }

    template <class Matrix, class Vec1, class Vec2>
    void apply(const Matrix& A, const Vec1& rhs, Vec2&& x) const
    {
      auto F = backend::reinterpret_as_rhs<BlockType>(rhs);
      auto X = backend::reinterpret_as_rhs<BlockType>(x);
      base.apply(A, F, X);
    }

    const matrix& system_matrix() const
    {
      return base.system_matrix();
    }

    std::shared_ptr<matrix> system_matrix_ptr() const
    {
      return base.system_matrix_ptr();
    }

    size_t bytes() const
    {
      return base.bytes();
    }

   private:

    Base base;
    size_t nrows;
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allows to use an AMG smoother as standalone preconditioner.
 */
template <class Backend, template <class> class Relax>
class RelaxationAsPreconditioner
{
 public:

  typedef Backend backend_type;

  typedef Relax<Backend> smoother;

  typedef typename Backend::matrix matrix;
  typedef typename Backend::vector vector;
  typedef typename smoother::params params;
  typedef typename Backend::params backend_params;

  typedef typename Backend::value_type value_type;
  typedef typename Backend::col_type col_type;
  typedef typename Backend::ptr_type ptr_type;
  typedef typename BuiltinBackend<value_type, col_type, ptr_type>::matrix build_matrix;

  template <class Matrix>
  RelaxationAsPreconditioner(const Matrix& M,
                    const params& prm = params(),
                    const backend_params& bprm = backend_params())
  : prm(prm)
  {
    init(std::make_shared<build_matrix>(M), bprm);
  }

  RelaxationAsPreconditioner(std::shared_ptr<build_matrix> M,
                    const params& prm = params(),
                    const backend_params& bprm = backend_params())
  : prm(prm)
  {
    init(M, bprm);
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    S->apply(*A, rhs, x);
  }

  const matrix& system_matrix() const
  {
    return *A;
  }

  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    return A;
  }

  size_t bytes() const
  {
    size_t b = 0;

    if (A)
      b += backend::bytes(*A);
    if (S)
      b += backend::bytes(*S);

    return b;
  }

 private:

  params prm;

  std::shared_ptr<matrix> A;
  std::shared_ptr<smoother> S;

  void init(std::shared_ptr<build_matrix> M, const backend_params& bprm)
  {
    A = Backend::copy_matrix(M, bprm);
    S = std::make_shared<smoother>(*M, prm, bprm);
  }

  friend std::ostream& operator<<(std::ostream& os, const RelaxationAsPreconditioner& p)
  {
    os << "Relaxation as preconditioner" << std::endl;
    os << "  Unknowns: " << backend::nbRow(p.system_matrix()) << std::endl;
    os << "  Nonzeros: " << backend::nonzeros(p.system_matrix()) << std::endl;
    os << "  Memory:   " << human_readable_memory(p.bytes()) << std::endl;

    return os;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Chebyshev polynomial smoother.
 *
 * \param Backend Backend for temporary structures allocation.
 * \ingroup relaxation
 *
 * Implements Algorithm 1 from
 * P. Ghysels, P. Kłosiewicz, and W. Vanroose.
 * "Improving the arithmetic intensity of multigrid with the help of polynomial smoothers".
 * Numer. Linear Algebra Appl. 2012;19:253-267. DOI: 10.1002/nla.1808
 */
template <class Backend>
class ChebyshevRelaxation
: public RelaxationBase
{
 public:

  typedef typename Backend::value_type value_type;
  typedef typename Backend::vector vector;

  typedef typename math::scalar_of<value_type>::type scalar_type;

  /// Relaxation parameters.
  struct params
  {
    /// Chebyshev polynomial degree.
    Int32 degree = 5;

    /// highest eigen value safety upscaling.
    // use boosting factor for a more conservative upper bound estimate
    // See: Adams, Brezina, Hu, Tuminaro,
    //      PARALLEL MULTIGRID SMOOTHING: POLYNOMIAL VERSUS
    //      GAUSS-SEIDEL, J. Comp. Phys. 188 (2003) 593-610.
    //
    double higher = 1.0;

    /// Lowest-to-highest eigen value ratio.
    double lower = 1.0 / 30.0;

    // Number of power iterations to apply for the spectral radius
    // estimation. When 0, use Gershgorin disk theorem to estimate
    // spectral radius.
    Int32 power_iters = 0;

    // Scale the system matrix
    bool scale = false;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, degree)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, higher)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, lower)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, power_iters)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, scale)
    {
      p.check_params( { "degree", "higher", "lower", "power_iters", "scale" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, degree);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, higher);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, lower);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, power_iters);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, scale);
    }
  } prm;

  /// \copydoc DampedJacobiRelaxation::DampedJacobiRelaxation
  template <class Matrix>
  ChebyshevRelaxation(const Matrix& A, const params& prm,
                      const typename Backend::params& backend_prm)
  : prm(prm)
  , p(Backend::create_vector(backend::nbRow(A), backend_prm))
  , r(Backend::create_vector(backend::nbRow(A), backend_prm))
  {
    scalar_type hi, lo;

    //using spectral_radius;

    if (prm.scale) {
      M = Backend::copy_vector(diagonal(A, /*invert*/ true), backend_prm);
      hi = spectral_radius<true>(A, prm.power_iters);
    }
    else {
      hi = spectral_radius<false>(A, prm.power_iters);
    }

    lo = hi * prm.lower;
    hi *= prm.higher;

    // Centre of ellipse containing the eigenvalues of A:
    d = 0.5 * (hi + lo);

    // Semi-major axis of ellipse containing the eigenvalues of A:
    c = 0.5 * (hi - lo);
  }

  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP&) const
  {
    solve(A, rhs, x);
  }

  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP&) const
  {
    solve(A, rhs, x);
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix& A, const VectorRHS& rhs, VectorX& x) const
  {
    backend::clear(x);
    solve(A, rhs, x);
  }

  size_t bytes() const
  {
    size_t b = backend::bytes(*p) + backend::bytes(*r);
    if (prm.scale)
      b += backend::bytes(*M);
    return b;
  }

 private:

  std::shared_ptr<typename Backend::matrix_diagonal> M;
  mutable std::shared_ptr<vector> p, r;

  scalar_type c, d;

  template <class Matrix, class VectorB, class VectorX>
  void solve(const Matrix& A, const VectorB& b, VectorX& x) const
  {
    static const scalar_type one = math::identity<scalar_type>();
    static const scalar_type zero = math::zero<scalar_type>();

    scalar_type alpha = zero, beta = zero;

    for (unsigned k = 0; k < prm.degree; ++k) {
      backend::residual(b, A, x, *r);

      if (prm.scale)
        backend::vmul(one, *M, *r, zero, *r);

      if (k == 0) {
        alpha = math::inverse(d);
        beta = zero;
      }
      else if (k == 1) {
        alpha = 2 * d * math::inverse(2 * d * d - c * c);
        beta = alpha * d - one;
      }
      else {
        alpha = math::inverse(d - 0.25 * alpha * c * c);
        beta = alpha * d - one;
      }

      backend::axpby(alpha, *r, beta, *p);
      backend::axpby(one, *p, one, x);
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Damped Jacobi relaxation.
 *
 * \param Backend Backend for temporary structures allocation.
 * \ingroup relaxation
 */
template <class Backend>
struct DampedJacobiRelaxation
: public RelaxationBase
{
  typedef typename Backend::value_type value_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;

  /// Relaxation parameters.
  struct params
  {
    /// Damping factor.
    scalar_type damping;

    params(scalar_type damping = 0.72)
    : damping(damping)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, damping)
    {
      p.check_params( { "damping" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, damping);
    }
  } prm;

  std::shared_ptr<typename Backend::matrix_diagonal> dia;

  /// Constructs smoother for the system matrix.
  /*!
   * \param A           The system matrix.
   * \param prm         Relaxation parameters.
   * \param backend_prm Backend parameters.
   */
  template <class Matrix>
  DampedJacobiRelaxation(const Matrix& A,
                         const params& prm,
                         const typename Backend::params& backend_prm)
  : prm(prm)
  , dia(Backend::copy_vector(diagonal(A, true), backend_prm))
  {}

  /// Apply pre-relaxation
  /*!
   * \param A   System matrix.
   * \param rhs Right-hand side.
   * \param x   Solution vector.
   * \param tmp Scratch vector.
   * \param prm Relaxation parameters.
   */
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    backend::vmul(prm.damping, *dia, tmp, math::identity<scalar_type>(), x);
  }

  /// Apply post-relaxation
  /*!
   * \param A   System matrix.
   * \param rhs Right-hand side.
   * \param x   Solution vector.
   * \param tmp Scratch vector.
   * \param prm Relaxation parameters.
   */
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    backend::vmul(prm.damping, *dia, tmp, math::identity<scalar_type>(), x);
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix&, const VectorRHS& rhs, VectorX& x) const
  {
    backend::vmul(math::identity<scalar_type>(), *dia, rhs, math::zero<scalar_type>(), x);
  }

  size_t bytes() const
  {
    return backend::bytes(*dia);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gauss-Seidel relaxation.
 *
 * \note This is a serial relaxation and is only applicable to backends that
 * support matrix row iteration (e.g. BuiltinBackend or
 * EigenBackend).
 *
 * \param Backend Backend for temporary structures allocation.
 * \ingroup relaxation
 */
template <class Backend>
struct GaussSeidelRelaxation
: public RelaxationBase
{
  /// Relaxation parameters.
  struct params
  {
    /// Use serial version of the algorithm
    bool serial;

    params()
    : serial(false)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, serial)
    {
      p.check_params( { "serial" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, serial);
    }
  };

  bool is_serial;

  /// \copydoc DampedJacobiRelaxation::DampedJacobiRelaxation
  template <class Matrix>
  GaussSeidelRelaxation(const Matrix& A, const params& prm, const typename Backend::params&)
  : is_serial(prm.serial || num_threads() < 4)
  {
    if (!is_serial) {
      forward = std::make_shared<parallel_sweep<true>>(A);
      backward = std::make_shared<parallel_sweep<false>>(A);
    }
  }

  /// \copydoc DampedJacobiRelaxation::apply_pre
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP&) const
  {
    if (is_serial)
      serial_sweep(A, rhs, x, true);
    else
      forward->sweep(rhs, x);
  }

  /// \copydoc DampedJacobiRelaxation::apply_post
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP&) const
  {
    if (is_serial)
      serial_sweep(A, rhs, x, false);
    else
      backward->sweep(rhs, x);
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix& A, const VectorRHS& rhs, VectorX& x) const
  {
    backend::clear(x);
    if (is_serial) {
      serial_sweep(A, rhs, x, true);
      serial_sweep(A, rhs, x, false);
    }
    else {
      forward->sweep(rhs, x);
      backward->sweep(rhs, x);
    }
  }

  size_t bytes() const
  {
    size_t b = 0;
    if (forward)
      b += forward->bytes();
    if (backward)
      b += backward->bytes();
    return b;
  }

 private:

  static int num_threads()
  {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
  }

  static int thread_id()
  {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
  }

  template <class Matrix, class VectorRHS, class VectorX>
  static void serial_sweep(const Matrix& A, const VectorRHS& rhs, VectorX& x, bool forward)
  {
    typedef typename backend::value_type<Matrix>::type val_type;
    typedef typename math::rhs_of<val_type>::type rhs_type;

    const ptrdiff_t n = backend::nbRow(A);

    const ptrdiff_t beg = forward ? 0 : n - 1;
    const ptrdiff_t end = forward ? n : -1;
    const ptrdiff_t inc = forward ? 1 : -1;

    for (ptrdiff_t i = beg; i != end; i += inc) {
      val_type D = math::identity<val_type>();
      rhs_type X;
      X = rhs[i];

      for (auto a = backend::row_begin(A, i); a; ++a) {
        ptrdiff_t c = a.col();
        val_type v = a.value();

        if (c == i)
          D = v;
        else
          X -= v * x[c];
      }

      x[i] = math::inverse(D) * X;
    }
  }

  template <bool forward>
  struct parallel_sweep
  {
    typedef typename Backend::value_type value_type;
    typedef typename math::rhs_of<value_type>::type rhs_type;

    struct task
    {
      ptrdiff_t beg, end;
      task(ptrdiff_t beg, ptrdiff_t end)
      : beg(beg)
      , end(end)
      {}
    };

    int nthreads;

    // thread-specific storage:
    std::vector<std::vector<task>> tasks;
    std::vector<std::vector<ptrdiff_t>> ptr;
    std::vector<std::vector<ptrdiff_t>> col;
    std::vector<std::vector<value_type>> val;
    std::vector<std::vector<ptrdiff_t>> ord;

    template <class Matrix>
    parallel_sweep(const Matrix& A)
    : nthreads(num_threads())
    , tasks(nthreads)
    , ptr(nthreads)
    , col(nthreads)
    , val(nthreads)
    , ord(nthreads)
    {
      ptrdiff_t n = backend::nbRow(A);
      ptrdiff_t nlev = 0;

      std::vector<ptrdiff_t> level(n, 0);
      std::vector<ptrdiff_t> order(n, 0);

      // 1. split rows into levels.
      ptrdiff_t beg = forward ? 0 : n - 1;
      ptrdiff_t end = forward ? n : -1;
      ptrdiff_t inc = forward ? 1 : -1;

      for (ptrdiff_t i = beg; i != end; i += inc) {
        ptrdiff_t l = level[i];

        for (auto a = backend::row_begin(A, i); a; ++a) {
          ptrdiff_t c = a.col();

          if (forward) {
            if (c >= i)
              continue;
          }
          else {
            if (c <= i)
              continue;
          }

          l = std::max(l, level[c] + 1);
        }

        level[i] = l;
        nlev = std::max(nlev, l + 1);
      }

      // 2. reorder matrix rows.
      std::vector<ptrdiff_t> start(nlev + 1, 0);

      for (ptrdiff_t i = 0; i < n; ++i)
        ++start[level[i] + 1];

      std::partial_sum(start.begin(), start.end(), start.begin());

      for (ptrdiff_t i = 0; i < n; ++i)
        order[start[level[i]]++] = i;

      std::rotate(start.begin(), start.end() - 1, start.end());
      start[0] = 0;

      // 3. Organize matrix rows into tasks.
      //    Each level is split into nthreads tasks.
      std::vector<ptrdiff_t> thread_rows(nthreads, 0);
      std::vector<ptrdiff_t> thread_cols(nthreads, 0);

#pragma omp parallel
      {
        int tid = thread_id();
        tasks[tid].reserve(nlev);

        for (ptrdiff_t lev = 0; lev < nlev; ++lev) {
          // split each level into tasks.
          ptrdiff_t lev_size = start[lev + 1] - start[lev];
          ptrdiff_t chunk_size = (lev_size + nthreads - 1) / nthreads;

          ptrdiff_t beg = std::min(tid * chunk_size, lev_size);
          ptrdiff_t end = std::min(beg + chunk_size, lev_size);

          beg += start[lev];
          end += start[lev];

          tasks[tid].push_back(task(beg, end));

          // count rows and nonzeros in the current task
          thread_rows[tid] += end - beg;
          for (ptrdiff_t i = beg; i < end; ++i) {
            ptrdiff_t j = order[i];
            thread_cols[tid] += backend::row_nonzeros(A, j);
          }
        }
      }

      // 4. reorganize matrix data for better cache and NUMA locality.
#pragma omp parallel
      {
        int tid = thread_id();

        col[tid].reserve(thread_cols[tid]);
        val[tid].reserve(thread_cols[tid]);
        ord[tid].reserve(thread_rows[tid]);
        ptr[tid].reserve(thread_rows[tid] + 1);
        ptr[tid].push_back(0);

        for (task& t : tasks[tid]) {
          ptrdiff_t loc_beg = ptr[tid].size() - 1;
          ptrdiff_t loc_end = loc_beg;

          for (ptrdiff_t r = t.beg; r < t.end; ++r, ++loc_end) {
            ptrdiff_t i = order[r];

            ord[tid].push_back(i);

            for (auto a = backend::row_begin(A, i); a; ++a) {
              col[tid].push_back(a.col());
              val[tid].push_back(a.value());
            }

            ptr[tid].push_back(col[tid].size());
          }

          t.beg = loc_beg;
          t.end = loc_end;
        }
      }
    }

    template <class Vector1, class Vector2>
    void sweep(const Vector1& rhs, Vector2& x) const
    {
#pragma omp parallel
      {
        int tid = thread_id();

        for (const task& t : tasks[tid]) {
          for (ptrdiff_t r = t.beg; r < t.end; ++r) {
            ptrdiff_t i = ord[tid][r];
            ptrdiff_t beg = ptr[tid][r];
            ptrdiff_t end = ptr[tid][r + 1];

            value_type D = math::identity<value_type>();
            rhs_type X;
            X = rhs[i];

            for (ptrdiff_t j = beg; j < end; ++j) {
              ptrdiff_t c = col[tid][j];
              value_type v = val[tid][j];

              if (c == i)
                D = v;
              else
                X -= v * x[c];
            }

            x[i] = math::inverse(D) * X;
          }

          // each task corresponds to a level, so we need
          // to synchronize across threads at this point:
#pragma omp barrier
          ;
        }
      }
    }

    size_t bytes() const
    {
      size_t b = 0;

      for (int i = 0; i < nthreads; ++i) {
        b += sizeof(task) * tasks[i].size();
        b += backend::bytes(ptr[i]);
        b += backend::bytes(col[i]);
        b += backend::bytes(val[i]);
        b += backend::bytes(ord[i]);
      }

      return b;
    }
  };

  std::shared_ptr<parallel_sweep<true>> forward;
  std::shared_ptr<parallel_sweep<false>> backward;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \note ILU(0) is a serial algorithm and is only applicable to backends that
 * support matrix row iteration (e.g. backend::builtin or backend::eigen).
 *
 * \param Backend Backend for temporary structures allocation.
 * \ingroup relaxation
 */
template <class Backend>
struct ILU0Relaxation
: public RelaxationBase
{
  typedef typename Backend::value_type value_type;
  typedef typename Backend::col_type col_type;
  typedef typename Backend::ptr_type ptr_type;
  typedef typename Backend::vector vector;
  typedef typename Backend::matrix matrix;
  typedef typename Backend::matrix_diagonal matrix_diagonal;

  typedef typename math::scalar_of<value_type>::type scalar_type;
  typedef Impl::ILUSolver<Backend> ilu_solve;

  /// Relaxation parameters.
  struct params
  {
    /// Damping factor.
    scalar_type damping;

    /// Parameters for sparse triangular system solver
    typename ilu_solve::params solve;

    params()
    : damping(1)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, damping)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, solve)
    {
      p.check_params( { "damping", "solve" }, { "k" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, damping);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, solve);
    }
  } prm;

  /// \copydoc DampedJacobiRelaxation::DampedJacobiRelaxation
  template <class Matrix>
  ILU0Relaxation(const Matrix& A, const params& prm, const typename Backend::params& bprm)
  : prm(prm)
  {
    typedef typename BuiltinBackend<value_type, col_type, ptr_type>::matrix build_matrix;
    const size_t n = backend::nbRow(A);

    size_t Lnz = 0, Unz = 0;

    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      ptrdiff_t row_beg = A.ptr[i];
      ptrdiff_t row_end = A.ptr[i + 1];

      for (ptrdiff_t j = row_beg; j < row_end; ++j) {
        ptrdiff_t c = A.col[j];
        if (c < i)
          ++Lnz;
        else if (c > i)
          ++Unz;
      }
    }

    auto L = std::make_shared<build_matrix>();
    auto U = std::make_shared<build_matrix>();

    L->set_size(n, n);
    L->set_nonzeros(Lnz);
    L->ptr[0] = 0;
    U->set_size(n, n);
    U->set_nonzeros(Unz);
    U->ptr[0] = 0;

    size_t Lhead = 0;
    size_t Uhead = 0;

    auto D = std::make_shared<numa_vector<value_type>>(n, false);

    std::vector<value_type*> work(n, NULL);

    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      ptrdiff_t row_beg = A.ptr[i];
      ptrdiff_t row_end = A.ptr[i + 1];

      for (ptrdiff_t j = row_beg; j < row_end; ++j) {
        ptrdiff_t c = A.col[j];
        value_type v = A.val[j];

        if (c < i) {
          L->col[Lhead] = c;
          L->val[Lhead] = v;
          work[c] = L->val + Lhead;
          ++Lhead;
        }
        else if (c == i) {
          (*D)[i] = v;
          work[c] = &(*D)[i];
        }
        else {
          U->col[Uhead] = c;
          U->val[Uhead] = v;
          work[c] = U->val + Uhead;
          ++Uhead;
        }
      }

      L->ptr[i + 1] = Lhead;
      U->ptr[i + 1] = Uhead;

      for (ptrdiff_t j = row_beg; j < row_end; ++j) {
        ptrdiff_t c = A.col[j];

        // Exit if diagonal is reached
        if (c >= i) {
          precondition(c == i, "No diagonal value in system matrix");
          precondition(!math::is_zero((*D)[i]), "Zero pivot in ILU");

          (*D)[i] = math::inverse((*D)[i]);
          break;
        }

        // Compute the multiplier for jrow
        value_type tl = (*work[c]) * (*D)[c];
        *work[c] = tl;

        // Perform linear combination
        for (ptrdiff_t k = U->ptr[c]; k < static_cast<ptrdiff_t>(U->ptr[c + 1]); ++k) {
          value_type* w = work[U->col[k]];
          if (w)
            *w -= tl * U->val[k];
        }
      }

      // Get rid of zeros in the factors
      Lhead = L->ptr[i];
      Uhead = U->ptr[i];

      for (ptrdiff_t j = Lhead, e = L->ptr[i + 1]; j < e; ++j) {
        auto v = L->val[j];
        if (!math::is_zero(v)) {
          L->col[Lhead] = L->col[j];
          L->val[Lhead] = v;
          ++Lhead;
        }
      }

      for (ptrdiff_t j = Uhead, e = U->ptr[i + 1]; j < e; ++j) {
        auto v = U->val[j];
        if (!math::is_zero(v)) {
          U->col[Uhead] = U->col[j];
          U->val[Uhead] = v;
          ++Uhead;
        }
      }
      L->ptr[i + 1] = Lhead;
      U->ptr[i + 1] = Uhead;

      // Refresh work
      for (ptrdiff_t j = row_beg; j < row_end; ++j)
        work[A.col[j]] = NULL;
    }

    L->setNbNonZero(Lhead);
    U->setNbNonZero(Uhead);

    ilu = std::make_shared<ilu_solve>(L, U, D, prm.solve, bprm);
  }

  /// \copydoc DampedJacobiRelaxation::apply_pre
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    ilu->solve(tmp);
    backend::axpby(prm.damping, tmp, math::identity<scalar_type>(), x);
  }

  /// \copydoc DampedJacobiRelaxation::apply_post
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    ilu->solve(tmp);
    backend::axpby(prm.damping, tmp, math::identity<scalar_type>(), x);
  }

  /// \copydoc DampedJacobiRelaxation::apply_post
  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix&, const VectorRHS& rhs, VectorX& x) const
  {
    backend::copy(rhs, x);
    ilu->solve(x);
  }

  size_t bytes() const
  {
    return ilu->bytes();
  }

 private:

  std::shared_ptr<ilu_solve> ilu;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief ILU(k) smoother.
 */
template <class Backend>
struct ILUKRelaxation
: public RelaxationBase
{
  typedef typename Backend::value_type value_type;
  typedef typename Backend::col_type col_type;
  typedef typename Backend::ptr_type ptr_type;
  typedef typename Backend::matrix matrix;
  typedef typename Backend::matrix_diagonal matrix_diagonal;
  typedef typename Backend::vector vector;

  typedef typename math::scalar_of<value_type>::type scalar_type;

  typedef Impl::ILUSolver<Backend> ilu_solve;

  /// Relaxation parameters.
  struct params
  {
    /// Level of fill-in.
    int k;

    /// Damping factor.
    scalar_type damping;

    /// Parameters for sparse triangular system solver
    typename ilu_solve::params solve;

    params()
    : k(1)
    , damping(1)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, k)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, damping)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, solve)
    {
      p.check_params( { "k", "damping", "solve" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, k);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, damping);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, solve);
    }
  } prm;

  /// \copydoc DampedJacobiRelaxation::DampedJacobiRelaxation
  template <class Matrix>
  ILUKRelaxation(const Matrix& A, const params& prm, const typename Backend::params& bprm)
  : prm(prm)
  {
    typedef typename BuiltinBackend<value_type, col_type, ptr_type>::matrix build_matrix;

    const size_t n = backend::nbRow(A);

    size_t Anz = backend::nonzeros(A);

    std::vector<ptrdiff_t> Lptr;
    Lptr.reserve(n + 1);
    Lptr.push_back(0);
    std::vector<ptrdiff_t> Lcol;
    Lcol.reserve(Anz / 3);
    std::vector<value_type> Lval;
    Lval.reserve(Anz / 3);

    std::vector<ptrdiff_t> Uptr;
    Uptr.reserve(n + 1);
    Uptr.push_back(0);
    std::vector<ptrdiff_t> Ucol;
    Ucol.reserve(Anz / 3);
    std::vector<value_type> Uval;
    Uval.reserve(Anz / 3);

    std::vector<int> Ulev;
    Ulev.reserve(Anz / 3);

    auto D = std::make_shared<numa_vector<value_type>>(n, false);

    sparse_vector w(n, prm.k);

    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      w.reset(i);

      for (auto a = backend::row_begin(A, i); a; ++a) {
        w.add(a.col(), a.value(), 0);
      }

      while (!w.q.empty()) {
        nonzero& a = w.next_nonzero();
        a.val = a.val * (*D)[a.col];

        for (ptrdiff_t j = Uptr[a.col], e = Uptr[a.col + 1]; j < e; ++j) {
          int lev = std::max(a.lev, Ulev[j]) + 1;
          w.add(Ucol[j], -a.val * Uval[j], lev);
        }
      }

      w.sort();

      for (const nonzero& e : w.nz) {
        if (e.col < i) {
          Lcol.push_back(e.col);
          Lval.push_back(e.val);
        }
        else if (e.col == i) {
          (*D)[i] = math::inverse(e.val);
        }
        else {
          Ucol.push_back(e.col);
          Uval.push_back(e.val);
          Ulev.push_back(e.lev);
        }
      }

      Lptr.push_back(Lcol.size());
      Uptr.push_back(Ucol.size());
    }

    ilu = std::make_shared<ilu_solve>(
    std::make_shared<build_matrix>(n, n, Lptr, Lcol, Lval),
    std::make_shared<build_matrix>(n, n, Uptr, Ucol, Uval),
    D, prm.solve, bprm);
  }

  /// \copydoc DampedJacobiRelaxation::apply_pre
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    ilu->solve(tmp);
    backend::axpby(prm.damping, tmp, math::identity<scalar_type>(), x);
  }

  /// \copydoc DampedJacobiRelaxation::apply_post
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    ilu->solve(tmp);
    backend::axpby(prm.damping, tmp, math::identity<scalar_type>(), x);
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix&, const VectorRHS& rhs, VectorX& x) const
  {
    backend::copy(rhs, x);
    ilu->solve(x);
  }

  size_t bytes() const
  {
    return ilu->bytes();
  }

 private:

  std::shared_ptr<ilu_solve> ilu;

  struct nonzero
  {
    ptrdiff_t col;
    value_type val;
    int lev;

    nonzero()
    : col(-1)
    {}

    nonzero(ptrdiff_t col, const value_type& val, int lev)
    : col(col)
    , val(val)
    , lev(lev)
    {}

    friend bool operator<(const nonzero& a, const nonzero& b)
    {
      return a.col < b.col;
    }
  };

  struct sparse_vector
  {
    struct comp_indices
    {
      const std::deque<nonzero>& nz;

      comp_indices(const std::deque<nonzero>& nz)
      : nz(nz)
      {}

      bool operator()(int a, int b) const
      {
        return nz[a].col > nz[b].col;
      }
    };

    typedef std::priority_queue<int, std::vector<int>, comp_indices>
    priority_queue;

    int lfil;

    std::deque<nonzero> nz;
    std::vector<ptrdiff_t> idx;
    priority_queue q;

    ptrdiff_t dia;

    sparse_vector(size_t n, int lfil)
    : lfil(lfil)
    , idx(n, -1)
    , q(comp_indices(nz))
    , dia(0)
    {}

    void add(ptrdiff_t col, const value_type& val, int lev)
    {
      if (idx[col] < 0) {
        if (lev <= lfil) {
          int p = nz.size();
          idx[col] = p;
          nz.push_back(nonzero(col, val, lev));
          if (col < dia)
            q.push(p);
        }
      }
      else {
        nonzero& a = nz[idx[col]];
        a.val += val;
        a.lev = std::min(a.lev, lev);
      }
    }

    typename std::deque<nonzero>::iterator begin()
    {
      return nz.begin();
    }

    typename std::deque<nonzero>::iterator end()
    {
      return nz.end();
    }

    nonzero& next_nonzero()
    {
      int p = q.top();
      q.pop();
      return nz[p];
    }

    void sort()
    {
      std::sort(nz.begin(), nz.end());
    }

    void reset(ptrdiff_t d)
    {
      for (const nonzero& e : nz)
        idx[e.col] = -1;
      nz.clear();
      dia = d;
    }
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace detail
{

  template <class Matrix>
  std::shared_ptr<Matrix> symb_product(const Matrix& A, const Matrix& B)
  {
    auto C = std::make_shared<Matrix>();

    C->set_size(A.nbRow(), B.ncols);

    auto A_ptr = A.ptr.data();
    auto A_col = A.col.data();
    auto B_ptr = B.ptr.data();
    auto B_col = B.col.data();
    auto C_ptr = C->ptr.data();
    C_ptr[0] = 0;

#pragma omp parallel
    {
      std::vector<ptrdiff_t> marker(B.ncols, -1);

#pragma omp for
      for (ptrdiff_t ia = 0; ia < static_cast<ptrdiff_t>(A.nbRow()); ++ia) {
        ptrdiff_t C_cols = 0;
        for (ptrdiff_t ja = A_ptr[ia], ea = A_ptr[ia + 1]; ja < ea; ++ja) {
          ptrdiff_t ca = A_col[ja];

          for (ptrdiff_t jb = B_ptr[ca], eb = B_ptr[ca + 1]; jb < eb; ++jb) {
            ptrdiff_t cb = B_col[jb];
            if (marker[cb] != ia) {
              marker[cb] = ia;
              ++C_cols;
            }
          }
        }
        C_ptr[ia + 1] = C_cols;
      }
    }

    C->set_nonzeros(C->scan_row_sizes(), /*need_values = */ false);
    auto C_col = C->col.data();

#pragma omp parallel
    {
      std::vector<ptrdiff_t> marker(B.ncols, -1);

#pragma omp for
      for (ptrdiff_t ia = 0; ia < static_cast<ptrdiff_t>(A.nbRow()); ++ia) {
        ptrdiff_t row_beg = C_ptr[ia];
        ptrdiff_t row_end = row_beg;

        for (ptrdiff_t ja = A_ptr[ia], ea = A_ptr[ia + 1]; ja < ea; ++ja) {
          ptrdiff_t ca = A_col[ja];

          for (ptrdiff_t jb = B_ptr[ca], eb = B_ptr[ca + 1]; jb < eb; ++jb) {
            ptrdiff_t cb = B_col[jb];

            if (marker[cb] < row_beg) {
              marker[cb] = row_end;
              C_col[row_end] = cb;
              ++row_end;
            }
          }
        }

        std::sort(C_col + row_beg, C_col + row_end);
      }
    }

    return C;
  }

} // namespace detail

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief ILU(k) smoother.
 */
template <class Backend>
struct ILUPRelaxation
: public RelaxationBase
{
  typedef typename Backend::value_type value_type;

  typedef ILU0Relaxation<Backend> Base;

  /// Relaxation parameters.
  struct params : Base::params
  {
    typedef typename Base::params BasePrm;

    /// Level of fill-in.
    int k;

    params()
    : k(1)
    {}

    params(const PropertyTree& p)
    : BasePrm(p)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, k)
    {
      p.check_params( { "k", "damping", "solve" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      BasePrm::get(p, path);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, k);
    }
  } prm;

  /// \copydoc DampedJacobiRelaxation::DampedJacobiRelaxation
  template <class Matrix>
  ILUPRelaxation(const Matrix& A, const params& prm, const typename Backend::params& bprm)
  : prm(prm)
  {
    if (prm.k == 0) {
      base = std::make_shared<Base>(A, prm, bprm);
    }
    else {
      auto P = detail::symb_product(A, A);
      for (int k = 1; k < prm.k; ++k) {
        P = detail::symb_product(*P, A);
      }

      ptrdiff_t n = backend::nbRow(A);
      P->val.resize(P->nbNonZero());

      arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          ptrdiff_t p_beg = P->ptr[i];
          ptrdiff_t p_end = P->ptr[i + 1];
          ptrdiff_t a_beg = A.ptr[i];
          ptrdiff_t a_end = A.ptr[i + 1];

          std::fill(P->val + p_beg, P->val + p_end, math::zero<value_type>());

          for (ptrdiff_t ja = a_beg, ea = a_end, jp = p_beg, ep = p_end; ja < ea; ++ja) {
            ptrdiff_t ca = A.col[ja];
            while (jp < ep && P->col[jp] < ca)
              ++jp;
            if (P->col[jp] == ca)
              P->val[jp] = A.val[ja];
          }
        }
      });

      base = std::make_shared<Base>(*P, prm, bprm);
    }
  }

  /// \copydoc DampedJacobiRelaxation::apply_pre
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    base->apply_pre(A, rhs, x, tmp);
  }

  /// \copydoc DampedJacobiRelaxation::apply_post
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    base->apply_post(A, rhs, x, tmp);
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix& A, const VectorRHS& rhs, VectorX& x) const
  {
    base->apply(A, rhs, x);
  }

  size_t bytes() const
  {
    return base->bytes();
  }

 private:

  std::shared_ptr<Base> base;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \note ILUT is a serial algorithm and is only applicable to backends that
 * support matrix row iteration (e.g. BuiltinBaclend or
 * EigenBackend).
 *
 * \param Backend Backend for temporary structures allocation.
 * \ingroup relaxation
 */
template <class Backend>
struct ILUTRelaxation
: public RelaxationBase
{
  typedef typename Backend::value_type value_type;
  typedef typename Backend::col_type col_type;
  typedef typename Backend::ptr_type ptr_type;
  typedef typename Backend::matrix matrix;
  typedef typename Backend::matrix_diagonal matrix_diagonal;
  typedef typename Backend::vector vector;

  typedef typename math::scalar_of<value_type>::type scalar_type;

  typedef Impl::ILUSolver<Backend> ilu_solve;

  /// Relaxation parameters.
  struct params
  {
    /// Fill factor.
    double p = 2.0;

    /// Minimum magnitude of non-zero elements relative to the current row norm.
    double tau = 1.0e-2;

    /// Damping factor.
    double damping = 1.0;

    /// Parameters for sparse triangular system solver
    typename ilu_solve::params solve;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, p)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, tau)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, damping)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, solve)
    {
      p.check_params( { "p", "tau", "damping", "solve" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      double p2 = this->p;
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, p2);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, tau);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, damping);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, solve);
    }
  } prm;

  /// \copydoc DampedJacobiRelaxation::DampedJacobiRelaxation
  template <class Matrix>
  ILUTRelaxation(const Matrix& A, const params& prm, const typename Backend::params& bprm)
  : prm(prm)
  {
    const size_t n = backend::nbRow(A);

    size_t Lnz = 0, Unz = 0;

    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      ptrdiff_t row_beg = A.ptr[i];
      ptrdiff_t row_end = A.ptr[i + 1];

      int lenL = 0, lenU = 0;
      for (ptrdiff_t j = row_beg; j < row_end; ++j) {
        ptrdiff_t c = A.col[j];
        if (c < i)
          ++lenL;
        else if (c > i)
          ++lenU;
      }

      Lnz += static_cast<size_t>(lenL * prm.p);
      Unz += static_cast<size_t>(lenU * prm.p);
    }

    auto L = std::make_shared<build_matrix>();
    auto U = std::make_shared<build_matrix>();

    L->set_size(n, n);
    L->set_nonzeros(Lnz);
    L->ptr[0] = 0;
    U->set_size(n, n);
    U->set_nonzeros(Unz);
    U->ptr[0] = 0;

    auto D = std::make_shared<numa_vector<value_type>>(n, false);

    sparse_vector w(n);

    for (ptrdiff_t i = 0, Lhead = 0, Uhead = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      w.dia = i;

      int lenL = 0;
      int lenU = 0;

      scalar_type tol = math::zero<scalar_type>();

      for (auto a = backend::row_begin(A, i); a; ++a) {
        w[a.col()] = a.value();
        tol += math::norm(a.value());

        if (a.col() < i)
          ++lenL;
        if (a.col() > i)
          ++lenU;
      }
      tol *= prm.tau / (lenL + lenU);

      while (!w.q.empty()) {
        ptrdiff_t k = w.next_nonzero();
        w[k] = w[k] * (*D)[k];
        value_type wk = w[k];

        if (math::norm(wk) > tol) {
          for (ptrdiff_t j = U->ptr[k]; j < static_cast<ptrdiff_t>(U->ptr[k + 1]); ++j)
            w[U->col[j]] -= wk * U->val[j];
        }
      }

      w.move_to(
      static_cast<int>(lenL * prm.p),
      static_cast<int>(lenU * prm.p),
      tol, Lhead, *L, Uhead, *U, *D);

      L->ptr[i + 1] = Lhead;
      U->ptr[i + 1] = Uhead;
    }

    L->setNbNonZero(L->ptr[n]);
    U->setNbNonZero(U->ptr[n]);

    ilu = std::make_shared<ilu_solve>(L, U, D, prm.solve, bprm);
  }

  /// \copydoc DampedJacobiRelaxation::apply_pre
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    ilu->solve(tmp);
    backend::axpby(prm.damping, tmp, math::identity<scalar_type>(), x);
  }

  /// \copydoc DampedJacobiRelaxation::apply_post
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    ilu->solve(tmp);
    backend::axpby(prm.damping, tmp, math::identity<scalar_type>(), x);
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix&, const VectorRHS& rhs, VectorX& x) const
  {
    backend::copy(rhs, x);
    ilu->solve(x);
  }

  size_t bytes() const
  {
    return ilu->bytes();
  }

 private:

  typedef typename BuiltinBackend<value_type, col_type, ptr_type>::matrix build_matrix;
  std::shared_ptr<ilu_solve> ilu;

  struct sparse_vector
  {
    struct nonzero
    {
      ptrdiff_t col;
      value_type val;

      nonzero()
      : col(-1)
      {}

      nonzero(ptrdiff_t col, const value_type& val = math::zero<value_type>())
      : col(col)
      , val(val)
      {}
    };

    struct comp_indices
    {
      const std::vector<nonzero>& nz;

      comp_indices(const std::vector<nonzero>& nz)
      : nz(nz)
      {}

      bool operator()(int a, int b) const
      {
        return nz[a].col > nz[b].col;
      }
    };

    typedef std::priority_queue<int, std::vector<int>, comp_indices> priority_queue;

    std::vector<nonzero> nz;
    std::vector<ptrdiff_t> idx;
    priority_queue q;

    ptrdiff_t dia;

    sparse_vector(size_t n)
    : idx(n, -1)
    , q(comp_indices(nz))
    , dia(0)
    {
      nz.reserve(16);
    }

    value_type operator[](ptrdiff_t i) const
    {
      if (idx[i] >= 0)
        return nz[idx[i]].val;
      return math::zero<value_type>();
    }

    value_type& operator[](ptrdiff_t i)
    {
      if (idx[i] == -1) {
        int p = nz.size();
        idx[i] = p;
        nz.push_back(nonzero(i));
        if (i < dia)
          q.push(p);
      }
      return nz[idx[i]].val;
    }

    typename std::vector<nonzero>::iterator begin()
    {
      return nz.begin();
    }

    typename std::vector<nonzero>::iterator end()
    {
      return nz.end();
    }

    ptrdiff_t next_nonzero()
    {
      int p = q.top();
      q.pop();
      return nz[p].col;
    }

    struct higher_than
    {
      scalar_type tol;
      ptrdiff_t dia;

      higher_than(scalar_type tol, ptrdiff_t dia)
      : tol(tol)
      , dia(dia)
      {}

      bool operator()(const nonzero& v) const
      {
        return v.col == dia || math::norm(v.val) > tol;
      }
    };

    struct L_first
    {
      ptrdiff_t dia;

      L_first(ptrdiff_t dia)
      : dia(dia)
      {}

      bool operator()(const nonzero& v) const
      {
        return v.col < dia;
      }
    };

    struct by_abs_val
    {
      ptrdiff_t dia;

      by_abs_val(ptrdiff_t dia)
      : dia(dia)
      {}

      bool operator()(const nonzero& a, const nonzero& b) const
      {
        if (a.col == dia)
          return true;
        if (b.col == dia)
          return false;

        return math::norm(a.val) > math::norm(b.val);
      }
    };

    struct by_col
    {
      bool operator()(const nonzero& a, const nonzero& b) const
      {
        return a.col < b.col;
      }
    };

    void move_to(int lp, int up, scalar_type tol,
                 ptrdiff_t& Lhead, build_matrix& L,
                 ptrdiff_t& Uhead, build_matrix& U,
                 numa_vector<value_type>& D)
    {
      typedef typename std::vector<nonzero>::iterator ptr;

      ptr b = nz.begin();
      ptr e = nz.end();

      // Move zeros to back:
      e = std::partition(b, e, higher_than(tol, dia));

      // Split L and U:
      ptr m = std::partition(b, e, L_first(dia));

      // Get largest p elements in L and U.
      ptr lend = std::min(b + lp, m);
      ptr uend = std::min(m + up, e);

      if (lend != m)
        std::nth_element(b, lend, m, by_abs_val(dia));
      if (uend != e)
        std::nth_element(m, uend, e, by_abs_val(dia));

      // Sort entries by column number
      std::sort(b, lend, by_col());
      std::sort(m, uend, by_col());

      // copy L to the output matrix.
      for (ptr a = b; a != lend; ++a) {
        L.col[Lhead] = a->col;
        L.val[Lhead] = a->val;

        ++Lhead;
      }

      // Store inverted diagonal.
      D[dia] = math::inverse(m->val);

      if (m != uend) {
        ++m;

        // copy U to the output matrix.
        for (ptr a = m; a != uend; ++a) {
          U.col[Uhead] = a->col;
          U.val[Uhead] = a->val;

          ++Uhead;
        }
      }

      for (const nonzero& e : nz)
        idx[e.col] = -1;
      nz.clear();
    }
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sparse approximate interface smoother.
 *
 * The inverse matrix is approximated with diagonal matrix.
 *
 * \tparam Backend Backend for temporary structures allocation.
 * \ingroup relaxation
 * \sa \cite Broker2002
 */
template <class Backend>
struct SPAI0Relaxation
: public RelaxationBase
{
  typedef typename Backend::value_type value_type;
  typedef typename Backend::matrix_diagonal matrix_diagonal;

  typedef typename math::scalar_of<value_type>::type scalar_type;
  /// Relaxation parameters.
  typedef Alina::detail::empty_params params;

  /// \copydoc DampedJacobiRelaxation::DampedJacobiRelaxation
  template <class Matrix>
  SPAI0Relaxation(const Matrix& A, const params&, const typename Backend::params& backend_prm)
  {
    const size_t n = backend::nbRow(A);

    auto m = std::make_shared<numa_vector<value_type>>(n, false);

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        value_type num = math::zero<value_type>();
        scalar_type den = math::zero<scalar_type>();

        for (auto a = backend::row_begin(A, i); a; ++a) {
          value_type v = a.value();
          scalar_type norm_v = math::norm(v);
          den += norm_v * norm_v;
          if (a.col() == i)
            num += v;
        }

        (*m)[i] = math::inverse(den) * num;
      }
    });

    M = Backend::copy_vector(m, backend_prm);
  }

  /// \copydoc DampedJacobiRelaxation::apply_pre
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    static const scalar_type one = math::identity<scalar_type>();
    backend::residual(rhs, A, x, tmp);
    backend::vmul(one, *M, tmp, one, x);
  }

  /// \copydoc DampedJacobiRelaxation::apply_post
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    static const scalar_type one = math::identity<scalar_type>();
    backend::residual(rhs, A, x, tmp);
    backend::vmul(one, *M, tmp, one, x);
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix&, const VectorRHS& rhs, VectorX& x) const
  {
    backend::vmul(math::identity<scalar_type>(), *M, rhs, math::zero<scalar_type>(), x);
  }

  size_t bytes() const
  {
    return backend::bytes(*M);
  }

  std::shared_ptr<matrix_diagonal> M;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sparse approximate interface smoother.
 *
 * Sparsity pattern of the approximate inverse matrix coincides with that of A.
 *
 * \tparam Backend Backend for temporary structures allocation.
 * \ingroup relaxation
 * \sa \cite Broker2002
 */
template <class Backend>
struct SPAI1Relaxation
: public RelaxationBase
{
  typedef typename Backend::value_type value_type;
  typedef typename Backend::vector vector;

  typedef typename math::scalar_of<value_type>::type scalar_type;

  /// Relaxation parameters.
  typedef Alina::detail::empty_params params;

  /// \copydoc DampedJacobiRelaxation::DampedJacobiRelaxation
  template <class Matrix>
  SPAI1Relaxation(const Matrix& A, const params&, const typename Backend::params& backend_prm)
  {
    typedef typename backend::value_type<Matrix>::type value_type;

    const size_t n = backend::nbRow(A);
    const size_t m = backend::nbColumn(A);

    auto Ainv = std::make_shared<Matrix>(A);

    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      std::vector<ptrdiff_t> marker(m, -1);
      std::vector<ptrdiff_t> I, J;
      std::vector<value_type> B, ek;
      Alina::detail::QRFactorization<value_type> qr;

      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        ptrdiff_t row_beg = A.ptr[i];
        ptrdiff_t row_end = A.ptr[i + 1];

        I.assign(A.col + row_beg, A.col + row_end);

        J.clear();
        for (ptrdiff_t j = row_beg; j < row_end; ++j) {
          ptrdiff_t c = A.col[j];

          for (ptrdiff_t jj = A.ptr[c], ee = A.ptr[c + 1]; jj < ee; ++jj) {
            ptrdiff_t cc = A.col[jj];
            if (marker[cc] < 0) {
              marker[cc] = 1;
              J.push_back(cc);
            }
          }
        }
        std::sort(J.begin(), J.end());
        B.assign(I.size() * J.size(), math::zero<value_type>());
        ek.assign(J.size(), math::zero<value_type>());
        for (size_t j = 0; j < J.size(); ++j) {
          marker[J[j]] = j;
          if (J[j] == static_cast<ptrdiff_t>(i))
            ek[j] = math::identity<value_type>();
        }

        for (ptrdiff_t j = row_beg; j < row_end; ++j) {
          ptrdiff_t c = A.col[j];

          for (auto a = backend::row_begin(A, c); a; ++a)
            B[marker[a.col()] + J.size() * (j - row_beg)] = a.value();
        }

        qr.solve(J.size(), I.size(), &B[0], &ek[0], &Ainv->val[row_beg],
                 Alina::detail::col_major);

        for (size_t j = 0; j < J.size(); ++j)
          marker[J[j]] = -1;
      }
    });

    M = Backend::copy_matrix(Ainv, backend_prm);
  }

  /// \copydoc DampedJacobiRelaxation::apply_pre
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_pre(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    backend::spmv(math::identity<scalar_type>(), *M, tmp, math::identity<scalar_type>(), x);
  }

  /// \copydoc DampedJacobiRelaxation::apply_post
  template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
  void apply_post(const Matrix& A, const VectorRHS& rhs, VectorX& x, VectorTMP& tmp) const
  {
    backend::residual(rhs, A, x, tmp);
    backend::spmv(math::identity<scalar_type>(), *M, tmp, math::identity<scalar_type>(), x);
  }

  template <class Matrix, class VectorRHS, class VectorX>
  void apply(const Matrix&, const VectorRHS& rhs, VectorX& x) const
  {
    backend::spmv(math::identity<scalar_type>(), *M, rhs, math::zero<scalar_type>(), x);
  }

  size_t bytes() const
  {
    return backend::bytes(*M);
  }

  std::shared_ptr<typename Backend::matrix> M;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

template <class Backend>
struct relaxation_is_supported<Backend, SPAI1Relaxation,
                               typename std::enable_if<(Alina::math::static_rows<typename Backend::value_type>::value > 1)>::type> : std::false_type
{};

template <class Backend>
struct relaxation_is_supported<Backend, GaussSeidelRelaxation,
                               typename std::enable_if<
                               !Backend::provides_row_iterator::value>::type> : std::false_type
{};

} // namespace Arcane::Alina::backend

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
