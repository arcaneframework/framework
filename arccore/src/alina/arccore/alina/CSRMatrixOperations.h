// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CSRMatrixOperations.h                                       (C) 2000-2026 */
/*                                                                           */
/* Operations on CSRMatrix.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_CSRMATRIXOPERATIONS_H
#define ARCCORE_ALINA_CSRMATRIXOPERATIONS_H
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

#pragma GCC diagnostic ignored "-Wconversion"

#ifdef _OPENMP
#include <omp.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/NumaVector.h"
#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/CSRMatrix.h"
#include "arccore/alina/BackendInterface.h"
#include "arccore/alina/SparseMatrixMatrixProduct.h"

#include "arccore/accelerator/Atomic.h"

#include <random>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Sort rows of the matrix column-wise.
template <typename V, typename C, typename P>
void sort_rows(CSRMatrix<V, C, P>& A)
{
  const size_t n = A.nbRow();

  arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    for (ptrdiff_t i = begin; i < (begin + size); ++i) {
      P beg = A.ptr[i];
      P end = A.ptr[i + 1];
      detail::sort_row(A.col + beg, A.val + beg, end - beg);
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Transpose of a sparse matrix.
template <typename V, typename C, typename P>
std::shared_ptr<CSRMatrix<V, C, P>>
transpose(const CSRMatrix<V, C, P>& A)
{
  const size_t n = A.nbRow();
  const size_t m = A.ncols;
  const size_t nnz = A.nbNonZero();

  auto T = std::make_shared<CSRMatrix<V, C, P>>();
  T->set_size(m, n, true);

  for (size_t j = 0; j < nnz; ++j)
    ++(T->ptr[A.col[j] + 1]);

  T->scan_row_sizes();
  T->set_nonzeros();

  for (size_t i = 0; i < n; i++) {
    for (P j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
      P head = T->ptr[A.col[j]]++;

      T->col[head] = static_cast<C>(i);
      T->val[head] = math::adjoint(A.val[j]);
    }
  }

  std::rotate(T->ptr.data(), T->ptr.data() + m, T->ptr.data() + m + 1);
  T->ptr[0] = 0;

  return T;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Matrix-matrix product.
template <class Val, class Col, class Ptr>
std::shared_ptr<CSRMatrix<Val, Col, Ptr>>
product(const CSRMatrix<Val, Col, Ptr>& A, const CSRMatrix<Val, Col, Ptr>& B, bool sort = false)
{
  auto C = std::make_shared<CSRMatrix<Val, Col, Ptr>>();

  const int max_nb_threads = ConcurrencyBase::maxAllowedThread();

  // TODO: find a way to configure this value for testing purpose.
  if (max_nb_threads >= 16) {
    spgemm_rmerge(A, B, *C);
  }
  else {
    spgemm_saad(A, B, *C, sort);
  }

  return C;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Sum of two matrices
template <class Val, class Col, class Ptr>
std::shared_ptr<CSRMatrix<Val, Col, Ptr>>
sum(Val alpha, const CSRMatrix<Val, Col, Ptr>& A, Val beta,
    const CSRMatrix<Val, Col, Ptr>& B, bool sort = false)
{
  typedef ptrdiff_t Idx;

  auto C = std::make_shared<CSRMatrix<Val, Col, Ptr>>();
  precondition(A.nbRow() == B.nbRow() && A.ncols == B.ncols, "matrices should have same shape!");
  C->set_size(A.nbRow(), A.ncols);

  C->ptr[0] = 0;
  Int32 nb_row = static_cast<Idx>(C->nbRow());
  arccoreParallelFor(0, nb_row, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    std::vector<ptrdiff_t> marker(C->ncols, -1);
    for (Idx i = begin; i < (begin + size); ++i) {
      Idx C_cols = 0;

      for (Idx j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
        Idx c = A.col[j];

        if (marker[c] != i) {
          marker[c] = i;
          ++C_cols;
        }
      }

      for (Idx j = B.ptr[i], e = B.ptr[i + 1]; j < e; ++j) {
        Idx c = B.col[j];

        if (marker[c] != i) {
          marker[c] = i;
          ++C_cols;
        }
      }

      C->ptr[i + 1] = C_cols;
    }
  });

  C->set_nonzeros(C->scan_row_sizes());

  arccoreParallelFor(0, nb_row, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    std::vector<ptrdiff_t> marker(C->ncols, -1);
    for (Idx i = begin; i < (begin + size); ++i) {
      Idx row_beg = C->ptr[i];
      Idx row_end = row_beg;

      for (Idx j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
        Idx c = A.col[j];
        Val v = alpha * A.val[j];

        if (marker[c] < row_beg) {
          marker[c] = row_end;
          C->col[row_end] = c;
          C->val[row_end] = v;
          ++row_end;
        }
        else {
          C->val[marker[c]] += v;
        }
      }

      for (Idx j = B.ptr[i], e = B.ptr[i + 1]; j < e; ++j) {
        Idx c = B.col[j];
        Val v = beta * B.val[j];

        if (marker[c] < row_beg) {
          marker[c] = row_end;
          C->col[row_end] = c;
          C->val[row_end] = v;
          ++row_end;
        }
        else {
          C->val[marker[c]] += v;
        }
      }

      if (sort)
        Alina::detail::sort_row(C->col + row_beg, C->val + row_beg, row_end - row_beg);
    }
  });

  return C;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Scale matrix values.
template <class Val, class Col, class Ptr, class T> void
scale(CSRMatrix<Val, Col, Ptr>& A, T s)
{
  const ptrdiff_t nb_row = backend::nbRow(A);

  arccoreParallelFor(0, nb_row, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    for (Int32 i = begin; i < (begin + size); ++i) {
      for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j)
        A.val[j] *= s;
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Reduce matrix to a pointwise one
template <class value_type, class col_type, class ptr_type>
std::shared_ptr<CSRMatrix<typename math::scalar_of<value_type>::type, col_type, ptr_type>>
pointwise_matrix(const CSRMatrix<value_type, col_type, ptr_type>& A, unsigned block_size)
{
  typedef value_type V;
  typedef typename math::scalar_of<V>::type S;

  ARCCORE_ALINA_TIC("pointwise_matrix");
  const ptrdiff_t n = A.nbRow();
  const ptrdiff_t m = A.ncols;
  const ptrdiff_t np = n / block_size;
  const ptrdiff_t mp = m / block_size;

  precondition(np * block_size == n,
               "Matrix size should be divisible by block_size");

  auto ap = std::make_shared<CSRMatrix<S, col_type, ptr_type>>();
  auto& Ap = *ap;

  Ap.set_size(np, mp, true);

  arccoreParallelFor(0, np, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    std::vector<ptr_type> j(block_size);
    std::vector<ptr_type> e(block_size);
    for (Int32 ip = begin; ip < (begin + size); ++ip) {
      ptrdiff_t ia = ip * block_size;
      col_type cur_col = 0;
      bool done = true;

      for (unsigned k = 0; k < block_size; ++k) {
        ptr_type beg = j[k] = A.ptr[ia + k];
        ptr_type end = e[k] = A.ptr[ia + k + 1];

        if (beg == end)
          continue;

        col_type c = A.col[beg];

        if (done) {
          done = false;
          cur_col = c;
        }
        else {
          cur_col = std::min(cur_col, c);
        }
      }

      while (!done) {
        cur_col /= block_size;
        ++Ap.ptr[ip + 1];

        done = true;
        col_type col_end = (cur_col + 1) * block_size;
        for (unsigned k = 0; k < block_size; ++k) {
          ptr_type beg = j[k];
          ptr_type end = e[k];

          while (beg < end) {
            col_type c = A.col[beg++];

            if (c >= col_end) {
              if (done) {
                done = false;
                cur_col = c;
              }
              else {
                cur_col = std::min(cur_col, c);
              }

              break;
            }
          }

          j[k] = beg;
        }
      }
    }
  });

  Ap.set_nonzeros(Ap.scan_row_sizes());

  arccoreParallelFor(0, np, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    std::vector<ptr_type> j(block_size);
    std::vector<ptr_type> e(block_size);
    for (Int32 ip = begin; ip < (begin + size); ++ip) {
      ptrdiff_t ia = ip * block_size;
      col_type cur_col = 0;
      ptr_type head = Ap.ptr[ip];
      bool done = true;

      for (unsigned k = 0; k < block_size; ++k) {
        ptr_type beg = j[k] = A.ptr[ia + k];
        ptr_type end = e[k] = A.ptr[ia + k + 1];

        if (beg == end)
          continue;

        col_type c = A.col[beg];

        if (done) {
          done = false;
          cur_col = c;
        }
        else {
          cur_col = std::min(cur_col, c);
        }
      }

      while (!done) {
        cur_col /= block_size;

        Ap.col[head] = cur_col;

        done = true;
        bool first = true;
        S cur_val = math::zero<S>();

        col_type col_end = (cur_col + 1) * block_size;
        for (unsigned k = 0; k < block_size; ++k) {
          ptr_type beg = j[k];
          ptr_type end = e[k];

          while (beg < end) {
            col_type c = A.col[beg];
            S v = math::norm(A.val[beg]);
            ++beg;

            if (c >= col_end) {
              if (done) {
                done = false;
                cur_col = c;
              }
              else {
                cur_col = std::min(cur_col, c);
              }

              break;
            }

            if (first) {
              first = false;
              cur_val = v;
            }
            else {
              cur_val = std::max(cur_val, v);
            }
          }

          j[k] = beg;
        }

        Ap.val[head++] = cur_val;
      }
    }
  });

  ARCCORE_ALINA_TOC("pointwise_matrix");
  return ap;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Diagonal of a matrix
template <typename V, typename C, typename P> std::shared_ptr<numa_vector<V>>
diagonal(const CSRMatrix<V, C, P>& A, bool invert = false)
{
  const size_t nb_row = A.nbRow();
  auto dia = std::make_shared<numa_vector<V>>(nb_row, false);

  arccoreParallelFor(0, nb_row, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
    for (Int32 i = begin; i < (begin + size); ++i) {
      for (auto a = A.row_begin(i); a; ++a) {
        if (a.col() == i) {
          V d = a.value();
          if (invert) {
            d = math::is_zero(d) ? math::identity<V>() : math::inverse(d);
          }
          (*dia)[i] = d;
          break;
        }
      }
    }
  });

  return dia;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Estimate spectral radius of the matrix.
// Use Gershgorin disk theorem when power_iters == 0,
// Use Power method when power_iters > 0.
// When scale = true, scale the matrix by its inverse diagonal.
template <bool scale, class Matrix>
static typename math::scalar_of<typename backend::value_type<Matrix>::type>::type
spectral_radius(const Matrix& A, int power_iters = 0)
{
  ARCCORE_ALINA_TIC("spectral radius");
  typedef typename backend::value_type<Matrix>::type value_type;
  typedef typename math::rhs_of<value_type>::type rhs_type;
  typedef typename math::scalar_of<value_type>::type scalar_type;

  const ptrdiff_t n = backend::nbRow(A);
  scalar_type radius;

  // TODO: CONCURRENCY Use arccore concurrency for spectral_radius
  if (power_iters <= 0) {
    // Use Gershgorin disk theorem.
    radius = 0;

    // NOTE: It is possible to do non blocking (nowait using OpenMP)
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      scalar_type emax = 0;
      value_type dia = math::identity<value_type>();

      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        scalar_type s = 0;

        for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
          ptrdiff_t c = A.col[j];
          value_type v = A.val[j];

          s += math::norm(v);

          if (scale && c == i)
            dia = v;
        }

        if (scale)
          s *= math::norm(math::inverse(dia));

        emax = std::max(emax, s);
      }

      Accelerator::doAtomic<Accelerator::eAtomicOperation::Max>(&radius, emax);
    });
  }
  else {
    // Power method.
    numa_vector<rhs_type> b0(n, false), b1(n, false);

    // Fill the initial vector with random values.
    // Also extract the inverted matrix diagonal values.
    std::atomic<scalar_type> atomic_b0_norm = 0;
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      const int tid = TaskFactory::currentTaskThreadIndex();
      std::mt19937 rng(tid);
      std::uniform_real_distribution<scalar_type> rnd(-1, 1);

      scalar_type loc_norm = 0;

      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        rhs_type v = math::constant<rhs_type>(rnd(rng));

        b0[i] = v;
        loc_norm += math::norm(math::inner_product(v, v));
      }

      // GG: Not reproducible
      atomic_b0_norm += loc_norm;
    });

    scalar_type b0_norm = atomic_b0_norm;

    // Normalize b0
    b0_norm = 1 / sqrt(b0_norm);
    arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
      for (ptrdiff_t i = begin; i < (begin + size); ++i) {
        b0[i] = b0_norm * b0[i];
      }
    });

    for (int iter = 0; iter < power_iters;) {
      // b1 = scale ? (D^1 * A) * b0 : A * b0
      // b1_norm = ||b1||
      // radius = <b1,b0>
      std::atomic<scalar_type> atomic_b1_norm = 0;
      std::atomic<scalar_type> atomic_radius = 0;
      arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
        scalar_type loc_norm = 0;
        scalar_type loc_radi = 0;
        value_type dia = math::identity<value_type>();

        for (ptrdiff_t i = begin; i < (begin + size); ++i) {
          rhs_type s = math::zero<rhs_type>();

          for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
            ptrdiff_t c = A.col[j];
            value_type v = A.val[j];
            if (scale && c == i)
              dia = v;
            s += v * b0[c];
          }

          if (scale)
            s = math::inverse(dia) * s;

          loc_norm += math::norm(math::inner_product(s, s));
          loc_radi += math::norm(math::inner_product(s, b0[i]));

          b1[i] = s;
        }

        {
          // GG: Not reproducible
          atomic_b1_norm += loc_norm;
          atomic_radius += loc_radi;
        }
      });
      scalar_type b1_norm = atomic_b1_norm;
      radius = atomic_radius;

      if (++iter < power_iters) {
        // b0 = b1 / b1_norm
        b1_norm = 1 / sqrt(b1_norm);
        arccoreParallelFor(0, n, ForLoopRunInfo{}, [&](Int32 begin, Int32 size) {
          for (ptrdiff_t i = begin; i < (begin + size); ++i) {
            b0[i] = b1_norm * b1[i];
          }
        });
      }
    }
  }
  ARCCORE_ALINA_TOC("spectral radius");

  return radius < 0 ? static_cast<scalar_type>(2) : radius;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
