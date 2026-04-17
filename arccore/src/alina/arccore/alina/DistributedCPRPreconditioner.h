// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DistributedCPRPreconditioner.h                              (C) 2000-2026 */
/*                                                                           */
/* Distributed CPR preconditioner.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_DISTRIBUTEDCPRPRECONDITIONER_H
#define ARCCORE_ALINA_DISTRIBUTEDCPRPRECONDITIONER_H
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

#include <cassert>
#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/DistributedInnerProduct.h"
#include "arccore/alina/DistributedMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Distributed CPR preconditioner.
 */
template <class PPrecond, class SPrecond>
class DistributedCPRPreconditioner
{
  static_assert(std::is_same<
                typename PPrecond::BackendType,
                typename SPrecond::BackendType>::value,
                "Backends for pressure and flow preconditioners should coinside!");

 public:

  using BackendType = PPrecond::BackendType;

  typedef BackendType::value_type value_type;
  typedef math::scalar_of<value_type>::type scalar_type;
  typedef BackendType::matrix bmatrix;
  typedef BackendType::vector vector;
  typedef BackendType::params backend_params;

  typedef DistributedMatrix<BackendType> matrix;
  typedef typename BuiltinBackend<value_type>::matrix build_matrix;

  struct params
  {
    typedef typename PPrecond::params pprecond_params;
    typedef typename SPrecond::params sprecond_params;

    pprecond_params pprecond;
    sprecond_params sprecond;

    int block_size = 2;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, pprecond)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, sprecond)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, block_size)
    {
      p.check_params({ "pprecond", "sprecond", "block_size", "active_rows" });
    }

    void get(PropertyTree& p, const std::string& path = "") const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, pprecond);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, sprecond);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, block_size);
    }
  };

  template <class Matrix>
  DistributedCPRPreconditioner(mpi_communicator comm,
                               const Matrix& K,
                               const params& prm = params(),
                               const backend_params& bprm = backend_params())
  : prm(prm)
  , comm(comm)
  , n(backend::nbRow(K))
  {
    init(std::make_shared<matrix>(comm, K, backend::nbRow(K)), bprm);
  }

  DistributedCPRPreconditioner(mpi_communicator comm,
                               std::shared_ptr<matrix> K,
                               const params& prm = params(),
                               const backend_params& bprm = backend_params())
  : prm(prm)
  , comm(comm)
  , n(K->loc_rows())
  {
    init(K, bprm);
  }

  template <class Vec1, class Vec2>
  void apply(const Vec1& rhs, Vec2&& x) const
  {
    const auto one = math::identity<scalar_type>();
    const auto zero = math::zero<scalar_type>();

    S->apply(rhs, x);
    backend::residual(rhs, S->system_matrix(), x, *rs);

    backend::spmv(one, *Fpp, *rs, zero, *rp);
    P->apply(*rp, *xp);

    backend::spmv(one, *Scatter, *xp, one, x);
  }

  std::shared_ptr<matrix> system_matrix_ptr() const
  {
    return S->system_matrix_ptr();
  }

  const matrix& system_matrix() const
  {
    return S->system_matrix();
  }

 public:

  params prm;

 private:

  mpi_communicator comm;
  size_t n, np;

  std::shared_ptr<PPrecond> P;
  std::shared_ptr<SPrecond> S;

  std::shared_ptr<bmatrix> Fpp, Scatter;
  std::shared_ptr<vector> rp, xp, rs;

  void init(std::shared_ptr<matrix> K, const backend_params& bprm)
  {
    typedef typename backend::row_iterator<build_matrix>::type row_iterator;
    const int B = prm.block_size;

    auto _K_loc = K->local();
    auto _K_rem = K->remote();

    auto& K_loc = *_K_loc;
    auto& K_rem = *_K_rem;

    np = n / B;

    auto fpp = std::make_shared<build_matrix>();
    fpp->set_size(np, n);
    fpp->set_nonzeros(n);
    fpp->ptr[0] = 0;

    auto App_loc = std::make_shared<build_matrix>();
    auto App_rem = std::make_shared<build_matrix>();

    App_loc->set_size(np, np, true);
    App_rem->set_size(np, 0, true);

    // Get the pressure matrix nonzero pattern,
    // extract and invert block diagonals.
#pragma omp parallel
    {
      std::vector<row_iterator> k;
      k.reserve(B);
      multi_array<value_type, 2> v(B, B);

#pragma omp for
      for (ptrdiff_t ip = 0; ip < static_cast<ptrdiff_t>(np); ++ip) {
        ptrdiff_t ik = ip * B;
        bool done = true;
        ptrdiff_t cur_col = 0;

        // Local part
        k.clear();
        for (int i = 0; i < B; ++i) {
          k.push_back(backend::row_begin(K_loc, ik + i));

          if (k.back()) {
            ptrdiff_t col = k.back().col() / B;
            if (done) {
              cur_col = col;
              done = false;
            }
            else {
              cur_col = std::min(cur_col, col);
            }
          }

          fpp->col[ik + i] = ik + i;
        }
        fpp->ptr[ip + 1] = ik + B;

        while (!done) {
          ++App_loc->ptr[ip + 1];

          ptrdiff_t end = (cur_col + 1) * B;

          if (cur_col == ip) {
            // This is diagonal block.
            // Capture its (transposed) value,
            // invert it and put the relevant row into fpp.
            for (int i = 0; i < B; ++i)
              for (int j = 0; j < B; ++j)
                v(i, j) = 0;

            for (int i = 0; i < B; ++i)
              for (; k[i] && k[i].col() < end; ++k[i])
                v(k[i].col() % B, i) = k[i].value();

            invert(v, &fpp->val[ik]);
          }
          else {
            // This is off-diagonal block.
            // Just skip it.
            for (int i = 0; i < B; ++i)
              while (k[i] && k[i].col() < end)
                ++k[i];
          }

          // Get next column number.
          done = true;
          for (int i = 0; i < B; ++i) {
            if (k[i]) {
              ptrdiff_t col = k[i].col() / B;
              if (done) {
                cur_col = col;
                done = false;
              }
              else {
                cur_col = std::min(cur_col, col);
              }
            }
          }
        }

        // Remote part
        k.clear();
        for (int i = 0; i < B; ++i) {
          k.push_back(backend::row_begin(K_rem, ik + i));

          if (k.back()) {
            ptrdiff_t col = k.back().col() / B;
            if (done) {
              cur_col = col;
              done = false;
            }
            else {
              cur_col = std::min(cur_col, col);
            }
          }
        }

        while (!done) {
          ++App_rem->ptr[ip + 1];

          ptrdiff_t end = (cur_col + 1) * B;

          for (int i = 0; i < B; ++i)
            while (k[i] && k[i].col() < end)
              ++k[i];

          // Get next column number.
          done = true;
          for (int i = 0; i < B; ++i) {
            if (k[i]) {
              ptrdiff_t col = k[i].col() / B;
              if (done) {
                cur_col = col;
                done = false;
              }
              else {
                cur_col = std::min(cur_col, col);
              }
            }
          }
        }
      }
    }

    App_loc->set_nonzeros(App_loc->scan_row_sizes());
    App_rem->set_nonzeros(App_rem->scan_row_sizes());

    auto scatter = std::make_shared<build_matrix>();
    scatter->set_size(n, np);
    scatter->set_nonzeros(np);
    scatter->ptr[0] = 0;

#pragma omp parallel
    {
      std::vector<row_iterator> k;
      k.reserve(B);

#pragma omp for
      for (ptrdiff_t ip = 0; ip < static_cast<ptrdiff_t>(np); ++ip) {
        ptrdiff_t ik = ip * B;
        bool done = true;
        ptrdiff_t cur_col;

        value_type* d = &fpp->val[ik];

        // Local part
        ptrdiff_t head = App_loc->ptr[ip];
        k.clear();
        for (int i = 0; i < B; ++i) {
          k.push_back(backend::row_begin(K_loc, ik + i));

          if (k.back()) {
            ptrdiff_t col = k.back().col() / B;
            if (done) {
              cur_col = col;
              done = false;
            }
            else {
              cur_col = std::min(cur_col, col);
            }
          }
        }

        while (!done) {
          ptrdiff_t end = (cur_col + 1) * B;
          value_type app = 0;

          for (int i = 0; i < B; ++i) {
            for (; k[i] && k[i].col() < end; ++k[i]) {
              if (k[i].col() % B == 0) {
                app += d[i] * k[i].value();
              }
            }
          }

          App_loc->col[head] = cur_col;
          App_loc->val[head] = app;
          ++head;

          // Get next column number.
          done = true;
          for (int i = 0; i < B; ++i) {
            if (k[i]) {
              ptrdiff_t col = k[i].col() / B;
              if (done) {
                cur_col = col;
                done = false;
              }
              else {
                cur_col = std::min(cur_col, col);
              }
            }
          }
        }

        // Remote part
        head = App_rem->ptr[ip];
        k.clear();
        for (int i = 0; i < B; ++i) {
          k.push_back(backend::row_begin(K_rem, ik + i));

          if (k.back()) {
            ptrdiff_t col = k.back().col() / B;
            if (done) {
              cur_col = col;
              done = false;
            }
            else {
              cur_col = std::min(cur_col, col);
            }
          }
        }

        while (!done) {
          ptrdiff_t end = (cur_col + 1) * B;
          value_type app = 0;

          for (int i = 0; i < B; ++i) {
            for (; k[i] && k[i].col() < end; ++k[i]) {
              if (k[i].col() % B == 0) {
                app += d[i] * k[i].value();
              }
            }
          }

          App_rem->col[head] = cur_col;
          App_rem->val[head] = app;
          ++head;

          // Get next column number.
          done = true;
          for (int i = 0; i < B; ++i) {
            if (k[i]) {
              ptrdiff_t col = k[i].col() / B;
              if (done) {
                cur_col = col;
                done = false;
              }
              else {
                cur_col = std::min(cur_col, col);
              }
            }
          }
        }

        scatter->col[ip] = ip;
        scatter->val[ip] = math::identity<value_type>();

        ptrdiff_t nnz = ip;
        for (int i = 0; i < B; ++i) {
          if (i == 0)
            ++nnz;
          scatter->ptr[ik + i + 1] = nnz;
        }
      }
    }

    auto App = std::make_shared<matrix>(comm, App_loc, App_rem);

    P = std::make_shared<PPrecond>(comm, App, prm.pprecond, bprm);
    S = std::make_shared<SPrecond>(comm, K, prm.sprecond, bprm);

    Fpp = BackendType::copy_matrix(fpp, bprm);
    Scatter = BackendType::copy_matrix(scatter, bprm);

    rp = BackendType::create_vector(np, bprm);
    xp = BackendType::create_vector(np, bprm);
    rs = BackendType::create_vector(n, bprm);
  }

  // Inverts dense matrix A;
  // Returns the first column of the inverted matrix.
  void invert(multi_array<value_type, 2>& A, value_type* y)
  {
    const int B = prm.block_size;

    // Perform LU-factorization of A in-place
    for (int k = 0; k < B; ++k) {
      value_type d = A(k, k);
      assert(!math::is_zero(d));
      for (int i = k + 1; i < B; ++i) {
        A(i, k) /= d;
        for (int j = k + 1; j < B; ++j)
          A(i, j) -= A(i, k) * A(k, j);
      }
    }

    // Invert unit vector in-place.
    // Lower triangular solve:
    for (int i = 0; i < B; ++i) {
      value_type b = static_cast<value_type>(i == 0);
      for (int j = 0; j < i; ++j)
        b -= A(i, j) * y[j];
      y[i] = b;
    }

    // Upper triangular solve:
    for (int i = B; i-- > 0;) {
      for (int j = i + 1; j < B; ++j)
        y[i] -= A(i, j) * y[j];
      y[i] /= A(i, i);
    }
  }

  template <class P, class S>
  friend std::ostream& operator<<(std::ostream& os, const DistributedCPRPreconditioner<P, S>& cpr)
  {
    os << "CPR (two-stage preconditioner)\n"
          "### Pressure preconditioner:\n"
       << *cpr.P << "\n"
                    "### Global preconditioner:\n"
       << *cpr.S << std::endl;
    return os;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
