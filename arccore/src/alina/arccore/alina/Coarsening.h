// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Coarsening.h                                                (C) 2000-2026 */
/*                                                                           */
/* Coarsening strategies for AMG hierarchy construction.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_COARSENING_H
#define ARCCORE_ALINA_COARSENING_H
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

#include <tuple>
#include <memory>
#include <vector>
#include <numeric>
#include <limits>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/QRFactorizationImpl.h"
#include "arccore/alina/Adapters.h"
#include "arccore/alina/ValueTypeInterface.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::detail
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  Galerkin operator.
 */
template <class Matrix>
std::shared_ptr<Matrix> galerkin(const Matrix& A, const Matrix& P, const Matrix& R)
{
  return product(R, *product(A, P));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Scaled Galerkin operator.
 */
template <class Matrix>
std::shared_ptr<Matrix> scaled_galerkin(const Matrix& A, const Matrix& P, const Matrix& R, float s)
{
  auto a = galerkin(A, P, R);
  scale(*a, s);
  return a;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::detail

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct nullspace_params
{
  /// Number of vectors in near nullspace.
  int cols;

  /// Near nullspace vectors.
  /**
     * The vectors are represented as columns of a 2D matrix stored in
     * row-major order.
     */
  std::vector<double> B;

  nullspace_params()
  : cols(0)
  {}

  nullspace_params(const PropertyTree& p)
  : cols(p.get("cols", nullspace_params().cols))
  {
    double* b = 0;
    b = p.get("B", b);

    if (b) {
      Int32 rows = 0;
      rows = p.get("rows", rows);

      precondition(cols > 0, "Error in nullspace parameters: "
                             "B is set, but cols is not");

      precondition(rows > 0, "Error in nullspace parameters: "
                             "B is set, but rows is not");

      B.assign(b, b + rows * cols);
    }
    else {
      precondition(cols == 0, "Error in nullspace parameters: "
                              "cols > 0, but B is empty");
    }

    p.check_params( { "cols", "rows", "B" });
  }

  void get(PropertyTree&, const std::string&) const {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Plain aggregation.
 *
 * Modification of a greedy aggregation scheme from \cite Vanek1996.
 * Connectivity is defined in a symmetric way, that is, two variables \f$i\f$
 * and \f$j\f$ are considered to be connected to each other if
 * \f$a_{ij}^2/a_{ii}a_{jj} > \varepsilon_{strong}\f$. Variables without
 * neighbours (resulting, e.g., from Dirichlet conditions) are excluded from
 * aggregation process. The aggregation is completed in a single pass over
 * variables: variables adjacent to a new aggregate are temporarily marked as
 * beloning to this aggregate. Later they may be claimed by other aggregates;
 * if nobody claims them, then they just stay in their initial aggregate.
 *
 * \ingroup aggregates
 */
struct plain_aggregates
{
  /// Aggregation parameters.
  struct params
  {
    /// Parameter \f$\varepsilon_{strong}\f$ defining strong couplings.
    /*!
     * Connectivity is defined in a symmetric way, that is, two variables
     * \f$i\f$ and \f$j\f$ are considered to be connected to each other if
     * \f$a_{ij}^2/a_{ii}a_{jj} > \varepsilon_{strong}\f$ with fixed \f$0 <
     * \varepsilon_{strong} < 1.\f$
     */
    float eps_strong;

    params()
    : eps_strong(0.08f)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, eps_strong)
    {
      p.check_params( { "eps_strong", "block_size" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, eps_strong);
    }
  };

  static const ptrdiff_t undefined = -1;
  static const ptrdiff_t removed = -2;

  /// Number of aggregates.
  size_t count;

  /*!
   * \brief Strong connectivity matrix.
   *
   * This is just 'values' part of CRS matrix. 'col' and 'ptr' arrays are
   * borrowed from the system matrix.
   */
  std::vector<char> strong_connection;

  /*!
   * \brief Aggerate id that each fine-level variable belongs to.
   *
   * When id[i] < 0, then variable i stays at the fine level (this could be
   * the case for a Dirichelt condition variable).
   */
  std::vector<ptrdiff_t> id;

  /*!
   * \brief Constructs aggregates for a given matrix.
   *
   * \param A   The system matrix.
   * \param prm Aggregation parameters.
   */
  template <class Matrix>
  plain_aggregates(const Matrix& A, const params& prm)
  : count(0)
  , strong_connection(backend::nonzeros(A))
  , id(backend::nbRow(A))
  {
    typedef typename backend::value_type<Matrix>::type value_type;
    typedef typename math::scalar_of<value_type>::type scalar_type;

    scalar_type eps_squared = prm.eps_strong * prm.eps_strong;

    const size_t n = backend::nbRow(A);

    /* 1. Get strong connections */
    auto dia = diagonal(A);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      value_type eps_dia_i = eps_squared * (*dia)[i];

      for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
        ptrdiff_t c = A.col[j];
        value_type v = A.val[j];

        strong_connection[j] = (c != i) && (eps_dia_i * (*dia)[c] < v * v);
      }
    }

    /* 2. Get aggregate ids */

    // Remove lonely nodes.
    size_t max_neib = 0;
    for (size_t i = 0; i < n; ++i) {
      ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1];
      max_neib = std::max<size_t>(max_neib, e - j);

      ptrdiff_t state = removed;
      for (; j < e; ++j)
        if (strong_connection[j]) {
          state = undefined;
          break;
        }

      id[i] = state;
    }

    std::vector<ptrdiff_t> neib;
    neib.reserve(max_neib);

    // Perform plain aggregation
    for (size_t i = 0; i < n; ++i) {
      if (id[i] != undefined)
        continue;

      // The point is not adjacent to a core of any previous aggregate:
      // so its a seed of a new aggregate.
      ptrdiff_t cur_id = static_cast<ptrdiff_t>(count++);
      id[i] = cur_id;

      // (*) Include its neighbors as well.
      neib.clear();
      for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
        ptrdiff_t c = A.col[j];
        if (strong_connection[j] && id[c] != removed) {
          id[c] = cur_id;
          neib.push_back(c);
        }
      }

      // Temporarily mark undefined points adjacent to the new aggregate
      // as members of the aggregate.
      // If nobody claims them later, they will stay here.
      for (ptrdiff_t c : neib) {
        for (ptrdiff_t j = A.ptr[c], e = A.ptr[c + 1]; j < e; ++j) {
          ptrdiff_t cc = A.col[j];
          if (strong_connection[j] && id[cc] == undefined)
            id[cc] = cur_id;
        }
      }
    }

    if (!count)
      throw error::empty_level();

    // Some of the aggregates could potentially vanish during expansion
    // step (*) above. We need to exclude those and renumber the rest.
    std::vector<ptrdiff_t> cnt(count, 0);
    for (ptrdiff_t i : id)
      if (i >= 0)
        cnt[i] = 1;
    std::partial_sum(cnt.begin(), cnt.end(), cnt.begin());

    if (static_cast<ptrdiff_t>(count) > cnt.back()) {
      count = cnt.back();

      for (size_t i = 0; i < n; ++i)
        if (id[i] >= 0)
          id[i] = cnt[id[i]] - 1;
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace detail
{
  struct skip_negative
  {
    const std::vector<ptrdiff_t>& key;
    int block_size;

    skip_negative(const std::vector<ptrdiff_t>& key, int block_size)
    : key(key)
    , block_size(block_size)
    {}

    bool operator()(ptrdiff_t i, ptrdiff_t j) const
    {
      // Cast to unsigned type to keep negative values at the end
      return static_cast<size_t>(key[i]) / block_size <
      static_cast<size_t>(key[j]) / block_size;
    }
  };
} // namespace detail

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tentative prolongation operator.
 *
 * If near nullspace vectors are not provided, returns piecewise-constant
 * prolongation operator. If user provides near nullspace vectors, those are
 * used to improve the prolongation operator.
 * \see \cite Vanek2001
 */
template <class Matrix>
std::shared_ptr<Matrix>
tentative_prolongation(size_t n,
                       size_t naggr,
                       const std::vector<ptrdiff_t> aggr,
                       nullspace_params& nullspace,
                       int block_size)
{
  typedef typename backend::value_type<Matrix>::type value_type;
  typedef typename backend::col_type<Matrix>::type col_type;

  auto P = std::make_shared<Matrix>();

  ARCCORE_ALINA_TIC("tentative");
  if (nullspace.cols > 0) {
    ptrdiff_t nba = naggr / block_size;

    // Sort fine points by aggregate number.
    // Put points not belonging to any aggregate to the end of the list.
    std::vector<ptrdiff_t> order(n);
    for (size_t i = 0; i < n; ++i)
      order[i] = i;
    std::stable_sort(order.begin(), order.end(), detail::skip_negative(aggr, block_size));
    std::vector<ptrdiff_t> aggr_ptr(nba + 1, 0);
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      ptrdiff_t a = aggr[order[i]];
      if (a < 0)
        break;
      ++aggr_ptr[a / block_size + 1];
    }
    std::partial_sum(aggr_ptr.begin(), aggr_ptr.end(), aggr_ptr.begin());

    // Precompute the shape of the prolongation operator.
    // Each row contains exactly nullspace.cols non-zero entries.
    // Rows that do not belong to any aggregate are empty.
    P->set_size(n, nullspace.cols * nba);
    P->ptr[0] = 0;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
      P->ptr[i + 1] = aggr[i] < 0 ? 0 : nullspace.cols;

    P->scan_row_sizes();
    P->set_nonzeros();

    // Compute the tentative prolongation operator and null-space vectors
    // for the coarser level.
    std::vector<double> Bnew;
    Bnew.resize(nba * nullspace.cols * nullspace.cols);

#pragma omp parallel
    {
      Alina::detail::QRFactorization<double> qr;
      std::vector<double> Bpart;

#pragma omp for
      for (ptrdiff_t i = 0; i < nba; ++i) {
        auto aggr_beg = aggr_ptr[i];
        auto aggr_end = aggr_ptr[i + 1];
        auto d = aggr_end - aggr_beg;

        Bpart.resize(d * nullspace.cols);

        for (ptrdiff_t j = aggr_beg, jj = 0; j < aggr_end; ++j, ++jj) {
          ptrdiff_t ib = nullspace.cols * order[j];
          for (int k = 0; k < nullspace.cols; ++k)
            Bpart[jj + d * k] = nullspace.B[ib + k];
        }

        qr.factorize(d, nullspace.cols, &Bpart[0], Alina::detail::col_major);

        for (int ii = 0, kk = 0; ii < nullspace.cols; ++ii)
          for (int jj = 0; jj < nullspace.cols; ++jj, ++kk)
            Bnew[i * nullspace.cols * nullspace.cols + kk] = qr.R(ii, jj);

        for (ptrdiff_t j = aggr_beg, ii = 0; j < aggr_end; ++j, ++ii) {
          col_type* c = &P->col[P->ptr[order[j]]];
          value_type* v = &P->val[P->ptr[order[j]]];

          for (int jj = 0; jj < nullspace.cols; ++jj) {
            c[jj] = i * nullspace.cols + jj;
            // TODO: this is just a workaround to make non-scalar value
            // types compile. Most probably this won't actually work.
            v[jj] = qr.Q(ii, jj) * math::identity<value_type>();
          }
        }
      }
    }

    std::swap(nullspace.B, Bnew);
  }
  else {
    P->set_size(n, naggr);
    P->ptr[0] = 0;
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
      P->ptr[i + 1] = (aggr[i] >= 0);

    P->set_nonzeros(P->scan_row_sizes());

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      if (aggr[i] >= 0) {
        P->col[P->ptr[i]] = aggr[i];
        P->val[P->ptr[i]] = math::identity<value_type>();
      }
    }
  }
  ARCCORE_ALINA_TOC("tentative");

  return P;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Pointwise aggregation.
 *
 * The system matrix should have block structure. It is reduced to a single
 * value per block and is subjected to coarsening::plain_aggregation.
 *
 * \ingroup aggregates
 */
class pointwise_aggregates
{
 public:

  /// Aggregation parameters.
  struct params : plain_aggregates::params
  {
    /**
     * \brief Block size for the system matrix.
     *
     * When block_size=1, the scheme is equivalent to (and performs on
     * par with) plain_aggregates.
     */
    Int32 block_size = 1;

    params() = default;

    params(const PropertyTree& p)
    : plain_aggregates::params(p)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, block_size)
    {
      p.check_params( { "eps_strong", "block_size" });
    }

    void get(Alina::PropertyTree& p, const std::string& path) const
    {
      plain_aggregates::params::get(p, path);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, block_size);
    }
  };

  static const ptrdiff_t undefined = -1;
  static const ptrdiff_t removed = -2;

  /// \copydoc plain_aggregates::count
  size_t count;

  /// \copydoc plain_aggregates::strong_connection
  std::vector<char> strong_connection;

  /// \copydoc plain_aggregates::id
  std::vector<ptrdiff_t> id;

  /// \copydoc plain_aggregates::plain_aggregates
  template <class Matrix>
  pointwise_aggregates(const Matrix& A, const params& prm, unsigned min_aggregate)
  : count(0)
  {
    if (prm.block_size == 1) {
      plain_aggregates aggr(A, prm);

      remove_small_aggregates(A.nbRow(), 1, min_aggregate, aggr);

      count = aggr.count;
      strong_connection.swap(aggr.strong_connection);
      id.swap(aggr.id);
    }
    else {
      strong_connection.resize(backend::nonzeros(A));
      id.resize(backend::nbRow(A));

      auto ap = pointwise_matrix(A, prm.block_size);
      auto& Ap = *ap;

      plain_aggregates pw_aggr(Ap, prm);

      remove_small_aggregates(Ap.nbRow(), prm.block_size, min_aggregate, pw_aggr);

      count = pw_aggr.count * prm.block_size;

#pragma omp parallel
      {
        std::vector<ptrdiff_t> j(prm.block_size);
        std::vector<ptrdiff_t> e(prm.block_size);

#pragma omp for
        for (ptrdiff_t ip = 0; ip < static_cast<ptrdiff_t>(Ap.nbRow()); ++ip) {
          ptrdiff_t ia = ip * prm.block_size;

          for (unsigned k = 0; k < prm.block_size; ++k, ++ia) {
            id[ia] = prm.block_size * pw_aggr.id[ip] + k;

            j[k] = A.ptr[ia];
            e[k] = A.ptr[ia + 1];
          }

          for (ptrdiff_t jp = Ap.ptr[ip], ep = Ap.ptr[ip + 1]; jp < ep; ++jp) {
            ptrdiff_t cp = Ap.col[jp];
            bool sp = (cp == ip) || pw_aggr.strong_connection[jp];

            ptrdiff_t col_end = (cp + 1) * prm.block_size;

            for (unsigned k = 0; k < prm.block_size; ++k) {
              ptrdiff_t beg = j[k];
              ptrdiff_t end = e[k];

              while (beg < end && A.col[beg] < col_end) {
                strong_connection[beg] = sp && A.col[beg] != (ia + k);
                ++beg;
              }

              j[k] = beg;
            }
          }
        }
      }
    }
  }

  static void remove_small_aggregates(size_t n, unsigned block_size, unsigned min_aggregate,
                                      plain_aggregates& aggr)
  {
    if (min_aggregate <= 1)
      return; // nothing to do

    // Count entries in each of the aggregates
    std::vector<ptrdiff_t> count(aggr.count, 0);

    for (size_t i = 0; i < n; ++i) {
      ptrdiff_t id = aggr.id[i];
      if (id != removed)
        ++count[id];
    }

    // If any aggregate has less entries than required, remove it.
    // Renumber the rest of the aggregates to leave no gaps.
    size_t m = 0;
    for (size_t i = 0; i < aggr.count; ++i) {
      if (block_size * count[i] < min_aggregate) {
        count[i] = removed;
      }
      else {
        count[i] = m++;
      }
    }

    // Update aggregate count and aggregate ids.
    aggr.count = m;

    for (size_t i = 0; i < n; ++i) {
      ptrdiff_t id = aggr.id[i];
      if (id != removed)
        aggr.id[i] = count[id];
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \defgroup coarsening Coarsening strategies
 * \brief Coarsening strategies for AMG hirarchy construction.
 *
 * A coarsener is a class that takes a system matrix and returns three
 * operators:
 *
 * 1. Restriction operator R that downsamples the residual error to a
 *    coarser level in AMG hierarchy,
 * 2. Prolongation operator P that interpolates a correction computed on a
 *    coarser grid into a finer grid,
 * 3. System matrix \f$A^H\f$ at a coarser level that is usually computed as a
 *    Galerkin operator \f$A^H = R A^h P\f$.
 *
 * The AMG hierarchy is constructed by recursive invocation of the selected
 * coarsener.
 */
/*!
 * \brief Non-smoothed aggregation.
 *
 * \ingroup coarsening
 */
template <class Backend>
struct AggregationCoarsening
{
  typedef pointwise_aggregates Aggregates;

  /// Coarsening parameters.
  struct params
  {
    /// Aggregation parameters.
    Aggregates::params aggr;

    /// Near nullspace parameters.
    nullspace_params nullspace;

    /*!
     * \brief Over-interpolation factor \f$\alpha\f$.
     *
     * In case of aggregation coarsening, coarse-grid
     * correction of smooth error, and by this the overall convergence, can
     * often be substantially improved by using "over-interpolation", that is,
     * by multiplying the actual correction (corresponding to piecewise
     * constant interpolation) by some factor \f$\alpha > 1\f$. Equivalently,
     * this means that the coarse-level Galerkin operator is re-scaled by
     * \f$1 / \alpha\f$:
     * \f[I_h^HA_hI_H^h \to \frac{1}{\alpha}I_h^HA_hI_H^h.\f]
     *
     * \sa  \cite Stuben1999, Section 9.1 "Re-scaling of the Galerkin operator".
     */
    float over_interp;

    params()
    : over_interp(math::static_rows<typename Backend::value_type>::value == 1 ? 1.5f : 2.0f)
    {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, aggr)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, nullspace)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, over_interp)
    {
      p.check_params( { "aggr", "nullspace", "over_interp" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, aggr);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, nullspace);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, over_interp);
    }
  } prm;

  explicit AggregationCoarsening(const params& prm = params())
  : prm(prm)
  {}

  /*!
   * \brief Creates transfer operators for the given system matrix.
   *
   * \param A   The system matrix.
   * \param prm Coarsening parameters.
   * \returns   A tuple of prolongation and restriction operators.
   */
  template <class Matrix>
  std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>
  transfer_operators(const Matrix& A)
  {
    const size_t n = backend::nbRow(A);

    ARCCORE_ALINA_TIC("aggregates");
    Aggregates aggr(A, prm.aggr, prm.nullspace.cols);
    ARCCORE_ALINA_TOC("aggregates");

    ARCCORE_ALINA_TIC("interpolation");
    auto P = tentative_prolongation<Matrix>(
    n, aggr.count, aggr.id, prm.nullspace, prm.aggr.block_size);
    ARCCORE_ALINA_TOC("interpolation");

    return std::make_tuple(P, transpose(*P));
  }

  /*!
   * \brief Creates system matrix for the coarser level.
   *
   * \param A The system matrix at the finer level.
   * \param P Prolongation operator returned by transfer_operators().
   * \param R Restriction operator returned by transfer_operators().
   * \returns System matrix for the coarser level.
   */
  template <class Matrix>
  std::shared_ptr<Matrix>
  coarse_operator(const Matrix& A, const Matrix& P, const Matrix& R) const
  {
    return detail::scaled_galerkin(A, P, R, 1 / prm.over_interp);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Apply a scalar coarsening on a block matrix.
 *
 * Takes a block matrix as input, converts it to the scalar format,
 * applies the base coarsening, converts the results back to block format.
 */
template <template <class> class Coarsening>
struct AsScalarCoarsening
{
  template <class Backend>
  struct type
  {
    typedef math::element_of<typename Backend::value_type>::type Scalar;
    typedef BuiltinBackend<Scalar, typename Backend::col_type, typename Backend::ptr_type> BaseBackend;
    typedef Coarsening<BaseBackend> Base;

    typedef typename Base::params params;
    Base base;

    type(const params& prm = params())
    : base(prm) {};

    template <class Matrix>
    typename std::enable_if<backend::coarsening_is_supported<BaseBackend, Coarsening>::value &&
                            (math::static_rows<typename backend::value_type<Matrix>::type>::value > 1),
                            std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>>::type
    transfer_operators(const Matrix& B)
    {
      typedef typename backend::value_type<Matrix>::type Block;
      auto T = base.transfer_operators(*adapter::unblock_matrix(B));

      auto& P = *std::get<0>(T);
      auto& R = *std::get<1>(T);

      sort_rows(P);
      sort_rows(R);

      return std::make_tuple(
      std::make_shared<Matrix>(adapter::block_matrix<Block>(P)),
      std::make_shared<Matrix>(adapter::block_matrix<Block>(R)));
    }

    template <class Matrix>
    typename std::enable_if<backend::coarsening_is_supported<BaseBackend, Coarsening>::value &&
                            (math::static_rows<typename backend::value_type<Matrix>::type>::value == 1),
                            std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>>::type
    transfer_operators(const Matrix& A)
    {
      return base.transfer_operators(A);
    }

    template <class Matrix>
    typename std::enable_if<!backend::coarsening_is_supported<BaseBackend, Coarsening>::value,
                            std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>>::type
    transfer_operators(const Matrix&)
    {
      throw std::logic_error("The coarsening is not supported by the backend");
    }

    template <class Matrix>
    std::shared_ptr<Matrix>
    coarse_operator(const Matrix& A, const Matrix& P, const Matrix& R) const
    {
      return base.coarse_operator(A, P, R);
    }
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Create rigid body modes from coordinate vector.
// To be used as near-nullspace vectors with aggregation coarsening
// for 2D or 3D elasticity problems.
// The output matrix B may be transposed on demand
// (to be used as a set of deflation vectors).
template <class Vector>
int rigid_body_modes(int ndim, const Vector& coo, std::vector<double>& B, bool transpose = false)
{
  precondition(ndim == 2 || ndim == 3, "Only 2D or 3D problems are supported");
  precondition(coo.size() % ndim == 0, "Coordinate vector size should be divisible by ndim");

  size_t n = coo.size();
  int nmodes = (ndim == 2 ? 3 : 6);
  B.resize(n * nmodes, 0.0);

  const int stride1 = transpose ? 1 : nmodes;
  const int stride2 = transpose ? n : 1;

  double sn = 1 / sqrt(n);

  if (ndim == 2) {
    for (size_t i = 0; i < n; ++i) {
      size_t nod = i / ndim;
      size_t dim = i % ndim;

      double x = coo[nod * 2 + 0];
      double y = coo[nod * 2 + 1];

      // Translation
      B[i * stride1 + dim * stride2] = sn;

      // Rotation
      switch (dim) {
      case 0:
        B[i * stride1 + 2 * stride2] = -y;
        break;
      case 1:
        B[i * stride1 + 2 * stride2] = x;
        break;
      }
    }
  }
  else if (ndim == 3) {
    for (size_t i = 0; i < n; ++i) {
      size_t nod = i / ndim;
      size_t dim = i % ndim;

      double x = coo[nod * 3 + 0];
      double y = coo[nod * 3 + 1];
      double z = coo[nod * 3 + 2];

      // Translation
      B[i * stride1 + dim * stride2] = sn;

      // Rotation
      switch (dim) {
      case 0:
        B[i * stride1 + 3 * stride2] = y;
        B[i * stride1 + 5 * stride2] = z;
        break;
      case 1:
        B[i * stride1 + 3 * stride2] = -x;
        B[i * stride1 + 4 * stride2] = -z;
        break;
      case 2:
        B[i * stride1 + 4 * stride2] = y;
        B[i * stride1 + 5 * stride2] = -x;
        break;
      }
    }
  }

  // Orthonormalization
  std::array<double, 6> dot;
  for (int i = ndim; i < nmodes; ++i) {
    std::fill(dot.begin(), dot.end(), 0.0);
    for (size_t j = 0; j < n; ++j) {
      for (int k = 0; k < i; ++k)
        dot[k] += B[j * stride1 + k * stride2] * B[j * stride1 + i * stride2];
    }
    double s = 0.0;
    for (size_t j = 0; j < n; ++j) {
      for (int k = 0; k < i; ++k)
        B[j * stride1 + i * stride2] -= dot[k] * B[j * stride1 + k * stride2];
      s += B[j * stride1 + i * stride2] * B[j * stride1 + i * stride2];
    }
    s = sqrt(s);
    for (size_t j = 0; j < n; ++j)
      B[j * stride1 + i * stride2] /= s;
  }

  return nmodes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classic Ruge-Stuben coarsening with direct interpolation.
 *
 * \ingroup coarsening
 * \sa \cite Stuben1999
 */
template <class Backend>
struct RugeStubenCoarsening
{
  /// Coarsening parameters.
  struct params
  {
    /*!
     * \brief Parameter \f$\varepsilon_{str}\f$ defining strong couplings.
     *
     * Variable \f$i\f$ is defined to be strongly negatively coupled to
     * another variable, \f$j\f$, if \f[-a_{ij} \geq
     * \varepsilon_{str}\max\limits_{a_{ik}<0}|a_{ik}|\quad \text{with
     * fixed} \quad 0 < \varepsilon_{str} < 1.\f] In practice, a value of
     * \f$\varepsilon_{str}=0.25\f$ is usually taken.
     */
    float eps_strong = 0.25f;

    /*!
     * \brief Truncate prolongation operator?
     *
     * Interpolation operators, and, hence coarse operators may increase
     * substabtially towards coarser levels. Without truncation, this may
     * become too costly. Truncation ignores all interpolatory connections
     * which are smaller (in absolute value) than the largest one by a
     * factor of \f$\varepsilon_{tr}\f$. The remaining weights are rescaled
     * so that the total sum remains unchanged. In practice, a value of
     * \f$\varepsilon_{tr}=0.2\f$ is usually taken.
     */
    bool do_trunc = true;

    /// Truncation parameter \f$\varepsilon_{tr}\f$.
    float eps_trunc = 0.2f;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, eps_strong)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, do_trunc)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, eps_trunc)
    {
      p.check_params( { "eps_strong", "do_trunc", "eps_trunc" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, eps_strong);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, do_trunc);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, eps_trunc);
    }
  } prm;

  explicit RugeStubenCoarsening(const params& prm = params())
  : prm(prm)
  {}

  template <class Matrix>
  std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>
  transfer_operators(const Matrix& A) const
  {
    typedef typename backend::value_type<Matrix>::type Val;
    typedef typename backend::col_type<Matrix>::type Col;
    typedef typename backend::ptr_type<Matrix>::type Ptr;
    typedef typename math::scalar_of<Val>::type Scalar;

    const size_t n = backend::nbRow(A);

    static const Scalar eps = Alina::detail::eps<Scalar>(1);

    static const Val zero = math::zero<Val>();

    std::vector<char> cf(n, 'U');
    CSRMatrix<char, Col, Ptr> S;

    ARCCORE_ALINA_TIC("C/F split");
    connect(A, prm.eps_strong, S, cf);
    cfsplit(A, S, cf);
    ARCCORE_ALINA_TOC("C/F split");

    ARCCORE_ALINA_TIC("interpolation");
    size_t nc = 0;
    std::vector<ptrdiff_t> cidx(n);
    for (size_t i = 0; i < n; ++i)
      if (cf[i] == 'C')
        cidx[i] = static_cast<ptrdiff_t>(nc++);

    if (!nc)
      throw error::empty_level();

    auto P = std::make_shared<Matrix>();
    P->set_size(n, nc, true);

    std::vector<Val> Amin, Amax;

    if (prm.do_trunc) {
      Amin.resize(n);
      Amax.resize(n);
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      if (cf[i] == 'C') {
        ++P->ptr[i + 1];
        continue;
      }

      if (prm.do_trunc) {
        Val amin = zero, amax = zero;

        for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
          if (!S.val[j] || cf[A.col[j]] != 'C')
            continue;

          amin = std::min(amin, A.val[j]);
          amax = std::max(amax, A.val[j]);
        }

        Amin[i] = (amin *= prm.eps_trunc);
        Amax[i] = (amax *= prm.eps_trunc);

        for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
          if (!S.val[j] || cf[A.col[j]] != 'C')
            continue;

          if (A.val[j] < amin || amax < A.val[j])
            ++P->ptr[i + 1];
        }
      }
      else {
        for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j)
          if (S.val[j] && cf[A.col[j]] == 'C')
            ++P->ptr[i + 1];
      }
    }

    P->set_nonzeros(P->scan_row_sizes());

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      ptrdiff_t row_head = P->ptr[i];

      if (cf[i] == 'C') {
        P->col[row_head] = cidx[i];
        P->val[row_head] = math::identity<Val>();
        continue;
      }

      Val dia = zero;
      Val a_num = zero, a_den = zero;
      Val b_num = zero, b_den = zero;
      Val d_neg = zero, d_pos = zero;

      for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
        ptrdiff_t c = A.col[j];
        Val v = A.val[j];

        if (c == i) {
          dia = v;
          continue;
        }

        if (v < zero) {
          a_num += v;
          if (S.val[j] && cf[c] == 'C') {
            a_den += v;
            if (prm.do_trunc && Amin[i] < v)
              d_neg += v;
          }
        }
        else {
          b_num += v;
          if (S.val[j] && cf[c] == 'C') {
            b_den += v;
            if (prm.do_trunc && v < Amax[i])
              d_pos += v;
          }
        }
      }

      Scalar cf_neg = 1;
      Scalar cf_pos = 1;

      if (prm.do_trunc) {
        if (math::norm(static_cast<Val>(a_den - d_neg)) > eps)
          cf_neg = math::norm(a_den) / math::norm(static_cast<Val>(a_den - d_neg));

        if (math::norm(static_cast<Val>(b_den - d_pos)) > eps)
          cf_pos = math::norm(b_den) / math::norm(static_cast<Val>(b_den - d_pos));
      }

      if (zero < b_num && math::norm(b_den) < eps)
        dia += b_num;

      Scalar alpha = math::norm(a_den) > eps ? -cf_neg * math::norm(a_num) / (math::norm(dia) * math::norm(a_den)) : 0;
      Scalar beta = math::norm(b_den) > eps ? -cf_pos * math::norm(b_num) / (math::norm(dia) * math::norm(b_den)) : 0;

      for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
        ptrdiff_t c = A.col[j];
        Val v = A.val[j];

        if (!S.val[j] || cf[c] != 'C')
          continue;
        if (prm.do_trunc && Amin[i] <= v && v <= Amax[i])
          continue;

        P->col[row_head] = cidx[c];
        P->val[row_head] = (v < zero ? alpha : beta) * v;
        ++row_head;
      }
    }
    ARCCORE_ALINA_TOC("interpolation");

    return std::make_tuple(P, transpose(*P));
  }

  template <class Matrix>
  std::shared_ptr<Matrix>
  coarse_operator(const Matrix& A, const Matrix& P, const Matrix& R) const
  {
    return detail::galerkin(A, P, R);
  }

 private:

  //-------------------------------------------------------------------
  // On return S will hold both strong connection matrix (in S.val, which
  // is piggybacking A.ptr and A.col), and its transposition (in S.ptr
  // and S.val).
  //
  // Variables that have no positive connections are marked as F(ine).
  //-------------------------------------------------------------------
  template <typename Val, typename Col, typename Ptr>
  static void connect(CSRMatrix<Val, Col, Ptr> const& A, float eps_strong,
                      CSRMatrix<char, Col, Ptr>& S,
                      std::vector<char>& cf)
  {
    typedef typename math::scalar_of<Val>::type Scalar;

    const size_t n = backend::nbRow(A);
    const size_t nnz = backend::nonzeros(A);
    const Scalar eps = Alina::detail::eps<Scalar>(1);

    S.setNbRow(n);
    S.ncols = n;
    S.ptr.resize(n + 1);
    S.val.resize(nnz);
    S.ptr[0] = 0;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      S.ptr[i + 1] = 0;

      Val a_min = math::zero<Val>();

      for (auto a = backend::row_begin(A, i); a; ++a)
        if (a.col() != i)
          a_min = std::min(a_min, a.value());

      if (math::norm(a_min) < eps) {
        cf[i] = 'F';
        continue;
      }

      a_min *= eps_strong;

      for (Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j)
        S.val[j] = (A.col[j] != i && A.val[j] < a_min);
    }

    // Transposition of S:
    for (size_t i = 0; i < nnz; ++i)
      if (S.val[i])
        ++(S.ptr[A.col[i] + 1]);

    S.scan_row_sizes();
    S.col.resize(S.ptr[n]);

    for (size_t i = 0; i < n; ++i)
      for (Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j)
        if (S.val[j])
          S.col[S.ptr[A.col[j]]++] = i;

    std::rotate(S.ptr.data(), S.ptr.data() + n, S.ptr.data() + n + 1);
    S.ptr[0] = 0;
  }

  // Split variables into C(oarse) and F(ine) sets.
  template <typename Val, typename Col, typename Ptr>
  static void cfsplit(CSRMatrix<Val, Col, Ptr> const& A,
                      CSRMatrix<char, Col, Ptr> const& S,
                      std::vector<char>& cf)
  {
    const size_t n = A.nbRow();

    std::vector<Col> lambda(n);

    // Initialize lambdas:
    for (size_t i = 0; i < n; ++i) {
      Col temp = 0;
      for (Ptr j = S.ptr[i], e = S.ptr[i + 1]; j < e; ++j)
        temp += (cf[S.col[j]] == 'U' ? 1 : 2);
      lambda[i] = temp;
    }

    // Keep track of variable groups with equal lambda values.
    // ptr - start of a group;
    // cnt - size of a group;
    // i2n - variable number;
    // n2i - vaiable position in a group.
    std::vector<Ptr> ptr(n + 1, 0);
    std::vector<Ptr> cnt(n, 0);
    std::vector<Ptr> i2n(n);
    std::vector<Ptr> n2i(n);

    for (size_t i = 0; i < n; ++i)
      ++ptr[lambda[i] + 1];

    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

    for (size_t i = 0; i < n; ++i) {
      Col lam = lambda[i];
      Ptr idx = ptr[lam] + cnt[lam]++;
      i2n[idx] = i;
      n2i[i] = idx;
    }

    // Process variables by decreasing lambda value.
    // 1. The vaiable with maximum value of lambda becomes next C-variable.
    // 2. Its neighbours from S' become F-variables.
    // 3. Keep lambda values in sync.
    for (size_t top = n; top-- > 0;) {
      Ptr i = i2n[top];
      Col lam = lambda[i];

      if (lam == 0) {
        std::replace(cf.begin(), cf.end(), 'U', 'C');
        break;
      }

      // Remove tne variable from its group.
      --cnt[lam];

      if (cf[i] == 'F')
        continue;

      // Mark the variable as 'C'.
      cf[i] = 'C';

      // Its neighbours from S' become F-variables.
      for (Ptr j = S.ptr[i], e = S.ptr[i + 1]; j < e; ++j) {
        Col c = S.col[j];

        if (cf[c] != 'U')
          continue;

        cf[c] = 'F';

        // Increase lambdas of the newly created F's neighbours.
        for (Ptr aj = A.ptr[c], ae = A.ptr[c + 1]; aj < ae; ++aj) {
          if (!S.val[aj])
            continue;

          Col ac = A.col[aj];
          Col lam_a = lambda[ac];

          if (cf[ac] != 'U' || static_cast<size_t>(lam_a) + 1 >= n)
            continue;

          Ptr old_pos = n2i[ac];
          Ptr new_pos = ptr[lam_a] + cnt[lam_a] - 1;

          n2i[i2n[old_pos]] = new_pos;
          n2i[i2n[new_pos]] = old_pos;

          std::swap(i2n[old_pos], i2n[new_pos]);

          --cnt[lam_a];
          ++cnt[lam_a + 1];
          ptr[lam_a + 1] = ptr[lam_a] + cnt[lam_a];

          lambda[ac] = lam_a + 1;
        }
      }

      // Decrease lambdas of the newly create C's neighbours.
      for (Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; j++) {
        if (!S.val[j])
          continue;

        Col c = A.col[j];
        Col lam = lambda[c];

        if (cf[c] != 'U' || lam == 0)
          continue;

        Ptr old_pos = n2i[c];
        Ptr new_pos = ptr[lam];

        n2i[i2n[old_pos]] = new_pos;
        n2i[i2n[new_pos]] = old_pos;

        std::swap(i2n[old_pos], i2n[new_pos]);

        --cnt[lam];
        ++cnt[lam - 1];
        ++ptr[lam];
        lambda[c] = lam - 1;
      }
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Smoothed aggregation coarsening.
 *
 * \ingroup coarsening
 * \sa \cite Vanek1996
 */
template <class Backend>
struct SmoothedAggregationCoarserning
{
  typedef pointwise_aggregates Aggregates;

  /// Coarsening parameters
  struct params
  {
    /// Aggregation parameters.
    Aggregates::params aggr;

    /// Near nullspace parameters.
    nullspace_params nullspace;

    /*!
     * \brief Relaxation factor.
     *
     * Used as a scaling for the damping factor omega.
     * When estimate_spectral_radius is set, then
     *   omega = relax * (4/3) / rho.
     * Otherwise
     *   omega = relax * (2/3).
     *
     * Piecewise constant prolongation \f$\tilde P\f$ from non-smoothed
     * aggregation is improved by a smoothing to get the final prolongation
     * matrix \f$P\f$. Simple Jacobi smoother is used here, giving the
     * prolongation matrix
     * \f[P = \left( I - \omega D^{-1} A^F \right) \tilde P.\f]
     * Here \f$A^F = (a_{ij}^F)\f$ is the filtered matrix given by
     * \f[
     * a_{ij}^F =
     * \begin{cases}
     * a_{ij} \quad \text{if} \; j \in N_i\     \
     * 0 \quad \text{otherwise}
     * \end{cases}, \quad \text{if}\; i \neq j,
     * \quad a_{ii}^F = a_{ii} - \sum\limits_{j=1,j\neq i}^n
     * \left(a_{ij} - a_{ij}^F \right),
     * \f]
     * where \f$N_i\f$ is the set of variables, strongly coupled to
     * variable \f$i\f$, and \f$D\f$ denotes the diagonal of \f$A^F\f$.
     */
    float relax = 1.0f;

    // Estimate the matrix spectral radius.
    // This usually improves convergence rate and results in faster solves,
    // but costs some time during setup.
    bool estimate_spectral_radius = false;

    // Number of power iterations to apply for the spectral radius
    // estimation. Use Gershgorin disk theorem when power_iters = 0.
    int power_iters = 0;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, aggr)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, nullspace)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, relax)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, estimate_spectral_radius)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, power_iters)
    {
      p.check_params({ "aggr", "nullspace", "relax", "estimate_spectral_radius", "power_iters" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, aggr);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, nullspace);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, relax);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, estimate_spectral_radius);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, power_iters);
    }
  };

  SmoothedAggregationCoarserning(const params& prm = params())
  : prm(prm)
  {}

  template <class Matrix>
  std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>
  transfer_operators(const Matrix& A)
  {
    typedef typename backend::value_type<Matrix>::type value_type;
    typedef typename math::scalar_of<value_type>::type scalar_type;

    const size_t n = backend::nbRow(A);

    ARCCORE_ALINA_TIC("aggregates");
    Aggregates aggr(A, prm.aggr, prm.nullspace.cols);
    prm.aggr.eps_strong *= 0.5;
    ARCCORE_ALINA_TOC("aggregates");

    auto P_tent = tentative_prolongation<Matrix>(n, aggr.count, aggr.id, prm.nullspace, prm.aggr.block_size);

    auto P = std::make_shared<Matrix>();
    P->set_size(backend::nbRow(*P_tent), backend::nbColumn(*P_tent), true);

    scalar_type omega = prm.relax;
    if (prm.estimate_spectral_radius) {
      omega *= static_cast<scalar_type>(4.0 / 3) / spectral_radius<true>(A, prm.power_iters);
    }
    else {
      omega *= static_cast<scalar_type>(2.0 / 3);
    }

    ARCCORE_ALINA_TIC("smoothing");
#pragma omp parallel
    {
      std::vector<ptrdiff_t> marker(P->ncols, -1);

      // Count number of entries in P.
#pragma omp for
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        for (ptrdiff_t ja = A.ptr[i], ea = A.ptr[i + 1]; ja < ea; ++ja) {
          ptrdiff_t ca = A.col[ja];

          // Skip weak off-diagonal connections.
          if (ca != i && !aggr.strong_connection[ja])
            continue;

          for (ptrdiff_t jp = P_tent->ptr[ca], ep = P_tent->ptr[ca + 1]; jp < ep; ++jp) {
            ptrdiff_t cp = P_tent->col[jp];

            if (marker[cp] != i) {
              marker[cp] = i;
              ++(P->ptr[i + 1]);
            }
          }
        }
      }
    }

    P->scan_row_sizes();
    P->set_nonzeros();

#pragma omp parallel
    {
      std::vector<ptrdiff_t> marker(P->ncols, -1);

      // Fill the interpolation matrix.
#pragma omp for
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {

        // Diagonal of the filtered matrix is the original matrix
        // diagonal minus its weak connections.
        value_type dia = math::zero<value_type>();
        for (ptrdiff_t j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
          if (A.col[j] == i || !aggr.strong_connection[j])
            dia += A.val[j];
        }
        if (!math::is_zero(dia))
          dia = -omega * math::inverse(dia);

        ptrdiff_t row_beg = P->ptr[i];
        ptrdiff_t row_end = row_beg;
        for (ptrdiff_t ja = A.ptr[i], ea = A.ptr[i + 1]; ja < ea; ++ja) {
          ptrdiff_t ca = A.col[ja];

          // Skip weak off-diagonal connections.
          if (ca != i && !aggr.strong_connection[ja])
            continue;

          value_type va = (ca == i)
          ? static_cast<value_type>(static_cast<scalar_type>(1 - omega) * math::identity<value_type>())
          : dia * A.val[ja];

          for (ptrdiff_t jp = P_tent->ptr[ca], ep = P_tent->ptr[ca + 1]; jp < ep; ++jp) {
            ptrdiff_t cp = P_tent->col[jp];
            value_type vp = P_tent->val[jp];

            if (marker[cp] < row_beg) {
              marker[cp] = row_end;
              P->col[row_end] = cp;
              P->val[row_end] = va * vp;
              ++row_end;
            }
            else {
              P->val[marker[cp]] += va * vp;
            }
          }
        }
      }
    }
    ARCCORE_ALINA_TOC("smoothing");

    return std::make_tuple(P, transpose(*P));
  }

  template <class Matrix>
  std::shared_ptr<Matrix>
  coarse_operator(const Matrix& A, const Matrix& P, const Matrix& R) const
  {
    return detail::galerkin(A, P, R);
  }

  params prm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Smoothed aggregation with energy minimization.
 *
 * \ingroup coarsening
 * \sa \cite Sala2008
 */
template <class Backend>
struct SmoothedAggregationEnergyMinCoarsening
{
  typedef pointwise_aggregates Aggregates;

  /// Coarsening parameters.
  struct params
  {
    /// Aggregation parameters.
    Aggregates::params aggr;

    /// Near nullspace parameters.
    nullspace_params nullspace;

    params() {}

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, aggr)
    , ARCCORE_ALINA_PARAMS_IMPORT_CHILD(p, nullspace)
    {
      p.check_params( { "aggr", "nullspace" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, aggr);
      ARCCORE_ALINA_PARAMS_EXPORT_CHILD(p, path, nullspace);
    }
  } prm;

  SmoothedAggregationEnergyMinCoarsening(const params& prm = params())
  : prm(prm)
  {}

  template <class Matrix>
  std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>
  transfer_operators(const Matrix& A)
  {
    typedef typename backend::value_type<Matrix>::type Val;
    typedef typename backend::col_type<Matrix>::type Col;
    typedef typename backend::ptr_type<Matrix>::type Ptr;
    typedef ptrdiff_t Idx;

    ARCCORE_ALINA_TIC("aggregates");
    Aggregates aggr(A, prm.aggr, prm.nullspace.cols);
    prm.aggr.eps_strong *= 0.5;
    ARCCORE_ALINA_TOC("aggregates");

    ARCCORE_ALINA_TIC("interpolation");
    auto P_tent = tentative_prolongation<Matrix>(backend::nbRow(A), aggr.count, aggr.id, prm.nullspace, prm.aggr.block_size);

    // Filter the system matrix
    CSRMatrix<Val, Col, Ptr> Af;
    Af.set_size(backend::nbRow(A), backend::nbColumn(A));
    Af.ptr[0] = 0;

    std::vector<Val> dia(Af.nbRow());

#pragma omp parallel for
    for (Idx i = 0; i < static_cast<Idx>(Af.nbRow()); ++i) {
      Idx row_begin = A.ptr[i];
      Idx row_end = A.ptr[i + 1];
      Idx row_width = row_end - row_begin;

      Val D = math::zero<Val>();
      for (Idx j = row_begin; j < row_end; ++j) {
        Idx c = A.col[j];
        Val v = A.val[j];

        if (c == i)
          D += v;
        else if (!aggr.strong_connection[j]) {
          D += v;
          --row_width;
        }
      }

      dia[i] = D;
      Af.ptr[i + 1] = row_width;
    }

    Af.set_nonzeros(Af.scan_row_sizes());

#pragma omp parallel for
    for (Idx i = 0; i < static_cast<Idx>(Af.nbRow()); ++i) {
      Idx row_begin = A.ptr[i];
      Idx row_end = A.ptr[i + 1];
      Idx row_head = Af.ptr[i];

      for (Idx j = row_begin; j < row_end; ++j) {
        Idx c = A.col[j];

        if (c == i) {
          Af.col[row_head] = i;
          Af.val[row_head] = dia[i];
          ++row_head;
        }
        else if (aggr.strong_connection[j]) {
          Af.col[row_head] = c;
          Af.val[row_head] = A.val[j];
          ++row_head;
        }
      }
    }

    std::vector<Val> omega;

    auto P = interpolation(Af, dia, *P_tent, omega);
    auto R = restriction(Af, dia, *P_tent, omega);
    ARCCORE_ALINA_TOC("interpolation");

    return std::make_tuple(P, R);
  }

  template <class Matrix>
  std::shared_ptr<Matrix>
  coarse_operator(const Matrix& A, const Matrix& P, const Matrix& R) const
  {
    return detail::galerkin(A, P, R);
  }

 private:

  template <class AMatrix, typename Val, typename Col, typename Ptr>
  static std::shared_ptr<CSRMatrix<Val, Col, Ptr>>
  interpolation(const AMatrix& A, const std::vector<Val>& Adia,
                const CSRMatrix<Val, Col, Ptr>& P_tent,
                std::vector<Val>& omega)
  {
    const size_t n = backend::nbRow(P_tent);
    const size_t nc = backend::nbColumn(P_tent);

    auto AP = product(A, P_tent, /*sort rows: */ true);

    omega.resize(nc, math::zero<Val>());
    std::vector<Val> denum(nc, math::zero<Val>());

#pragma omp parallel
    {
      std::vector<ptrdiff_t> marker(nc, -1);

      // Compute A * Dinv * AP row by row and compute columnwise
      // scalar products necessary for computation of omega. The
      // actual results of matrix-matrix product are not stored.
      std::vector<Col> adap_col(128);
      std::vector<Val> adap_val(128);

#pragma omp for
      for (ptrdiff_t ia = 0; ia < static_cast<ptrdiff_t>(n); ++ia) {
        adap_col.clear();
        adap_val.clear();

        // Form current row of ADAP matrix.
        for (auto a = A.row_begin(ia); a; ++a) {
          Col ca = a.col();
          Val va = math::inverse(Adia[ca]) * a.value();

          for (auto p = AP->row_begin(ca); p; ++p) {
            Col c = p.col();
            Val v = va * p.value();

            if (marker[c] < 0) {
              marker[c] = adap_col.size();
              adap_col.push_back(c);
              adap_val.push_back(v);
            }
            else {
              adap_val[marker[c]] += v;
            }
          }
        }

        Alina::detail::sort_row(
        &adap_col[0], &adap_val[0], adap_col.size());

        // Update columnwise scalar products (AP,ADAP) and (ADAP,ADAP).
        // 1. (AP, ADAP)
        for (
        Ptr ja = AP->ptr[ia], ea = AP->ptr[ia + 1],
            jb = 0, eb = adap_col.size();
        ja < ea && jb < eb;) {
          Col ca = AP->col[ja];
          Col cb = adap_col[jb];

          if (ca < cb)
            ++ja;
          else if (cb < ca)
            ++jb;
          else /*ca == cb*/ {
            Val v = AP->val[ja] * adap_val[jb];
#pragma omp critical
            omega[ca] += v;
            ++ja;
            ++jb;
          }
        }

        // 2. (ADAP, ADAP) (and clear marker)
        for (size_t j = 0, e = adap_col.size(); j < e; ++j) {
          Col c = adap_col[j];
          Val v = adap_val[j];
#pragma omp critical
          denum[c] += v * v;
          marker[c] = -1;
        }
      }
    }

    for (size_t i = 0, m = omega.size(); i < m; ++i)
      omega[i] = math::inverse(denum[i]) * omega[i];

    // Update AP to obtain P: P = (P_tent - D^-1 A P Omega)
    /*
     * Here we use the fact that if P(i,j) != 0,
     * then with necessity AP(i,j) != 0:
     *
     * AP(i,j) = sum_k(A_ik P_kj), and A_ii != 0.
     */
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
      Val dia = math::inverse(Adia[i]);

      for (Ptr ja = AP->ptr[i], ea = AP->ptr[i + 1],
               jp = P_tent.ptr[i], ep = P_tent.ptr[i + 1];
           ja < ea; ++ja) {
        Col ca = AP->col[ja];
        Val va = -dia * AP->val[ja] * omega[ca];

        for (; jp < ep; ++jp) {
          Col cp = P_tent.col[jp];
          if (cp > ca)
            break;

          if (cp == ca) {
            va += P_tent.val[jp];
            break;
          }
        }

        AP->val[ja] = va;
      }
    }

    return AP;
  }

  template <typename AMatrix, typename Val, typename Col, typename Ptr>
  static std::shared_ptr<CSRMatrix<Val, Col, Ptr>>
  restriction(const AMatrix& A, const std::vector<Val>& Adia,
              const CSRMatrix<Val, Col, Ptr>& P_tent,
              const std::vector<Val>& omega)
  {
    const size_t nc = backend::nbColumn(P_tent);

    auto R_tent = transpose(P_tent);
    sort_rows(*R_tent);

    auto RA = product(*R_tent, A, /*sort rows: */ true);

    // Compute R = R_tent - Omega R_tent A D^-1.
    /*
     * Here we use the fact that if R(i,j) != 0,
     * then with necessity RA(i,j) != 0:
     *
     * RA(i,j) = sum_k(R_ik A_kj), and A_jj != 0.
     */
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(nc); ++i) {
      Val w = omega[i];

      for (Ptr ja = RA->ptr[i], ea = RA->ptr[i + 1],
               jr = R_tent->ptr[i], er = R_tent->ptr[i + 1];
           ja < ea; ++ja) {
        Col ca = RA->col[ja];
        Val va = -w * math::inverse(Adia[ca]) * RA->val[ja];

        for (; jr < er; ++jr) {
          Col cr = R_tent->col[jr];
          if (cr > ca)
            break;

          if (cr == ca) {
            va += R_tent->val[jr];
            break;
          }
        }

        RA->val[ja] = va;
      }
    }

    return RA;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::backend
{

template <class Backend>
struct coarsening_is_supported<Backend, RugeStubenCoarsening,
                               typename std::enable_if<!std::is_arithmetic<typename backend::value_type<Backend>::type>::value>::type> : std::false_type
{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::backend

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
