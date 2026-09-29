// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CLJPCoarsening.h                                            (C) 2000-2026 */
/*                                                                           */
/* CLJP coarsening for AMG hierarchy construction.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_CLJPCOARSENING_H
#define ARCCORE_ALINA_CLJPCOARSENING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * The CLJP (Cleary-Luby-Jones-Plassman) splitting is adapted from the
 * BoomerAMG solver of hypre (hypre_BoomerAMGCoarsen and
 * hypre_BoomerAMGIndepSet), available in the 'extras/hypre-3.2.0'
 * subdirectory of this repository.
 *
 * Copyright (c) 1998-2026, Lawrence Livermore National Security, LLC
 * SPDX-License-Identifier: Apache-2.0
 *
 * The shared infrastructure used here (strength connections, direct
 * interpolation and Galerkin operator, defined in Coarsening.h) is based
 * on the work on AMGCL library (version march 2026) which can be found
 * at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <cstdint>

#include "arccore/alina/Coarsening.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief CLJP coarsening with direct interpolation.
 *
 * Randomized independent set C/F splitting, following the CLJP
 * (Cleary-Luby-Jones-Plassman) algorithm as implemented in hypre's
 * BoomerAMG (hypre_BoomerAMGCoarsen and hypre_BoomerAMGIndepSet,
 * coarsen type 0 in hypre).
 *
 * The measure of a variable is the number of variables strongly
 * depending on it, augmented with a pseudo random value to break
 * ties. Each iteration of the splitting selects an independent set
 * of variables with maximal measure as coarse points, removes the
 * edges incident to the new coarse points and updates the measures
 * of their strong neighbours. A variable whose strong dependencies
 * are all accounted for (either coarse points, or neighbours sharing
 * a common coarse point) and which no variable depends on anymore
 * becomes a fine point. Compared to the deterministic Ruge-Stueben
 * splitting, the randomized selection avoids long sequential
 * dependency chains, which is the property making CLJP suitable for
 * parallel coarsening in BoomerAMG.
 *
 * \ingroup coarsening
 */
template <class Backend>
struct CLJPCoarsening
{
  /// Coarsening parameters.
  struct params
  {
    /// Parameter \f$\varepsilon_{str}\f$ defining strong couplings.
    /*!
     * Variable \f$i\f$ is defined to be strongly negatively coupled to
     * another variable, \f$j\f$, if \f[-a_{ij} \geq
     * \varepsilon_{str}\max\limits_{a_{ik}<0}|a_{ik}|\quad \text{with
     * fixed} \quad 0 < \varepsilon_{str} < 1.\f] In practice, a value of
     * \f$\varepsilon_{str}=0.25\f$ is usually taken.
     */
    float eps_strong = 0.25f;

    /// Truncate prolongation operator?
    bool do_trunc = true;

    /// Truncation parameter \f$\varepsilon_{tr}\f$.
    float eps_trunc = 0.2f;

    /// Seed for the pseudo random measure augmentation.
    /*!
     * The randomization of the measures is what turns the greedy
     * selection into an independent set selection. A fixed seed (as in
     * hypre, which uses 2747 for the reproducible variant) makes the
     * splitting deterministic for a given matrix.
     */
    int seed = 2747;

    params() = default;

    params(const PropertyTree& p)
    : ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, eps_strong)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, do_trunc)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, eps_trunc)
    , ARCCORE_ALINA_PARAMS_IMPORT_VALUE(p, seed)
    {
      p.check_params( { "eps_strong", "do_trunc", "eps_trunc", "seed" });
    }

    void get(PropertyTree& p, const std::string& path) const
    {
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, eps_strong);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, do_trunc);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, eps_trunc);
      ARCCORE_ALINA_PARAMS_EXPORT_VALUE(p, path, seed);
    }
  } prm;

  explicit CLJPCoarsening(const params& prm = params())
  : prm(prm)
  {}

  template <class Matrix>
  std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>
  transfer_operators(const Matrix& A) const
  {
    typedef typename backend::col_type<Matrix>::type Col;
    typedef typename backend::ptr_type<Matrix>::type Ptr;

    const size_t n = backend::nbRow(A);

    UniqueArray<char> cf(n, 'U');
    CSRMatrix<char, Col, Ptr> S;

    ARCCORE_ALINA_TIC("C/F split");
    detail::strong_connections(A, prm.eps_strong, S, cf);
    cljp_split(A, S, cf, prm.seed);
    ARCCORE_ALINA_TOC("C/F split");

    auto P = detail::direct_interpolation(A, S, cf, prm.do_trunc, prm.eps_trunc);

    return std::make_tuple(P, transpose(*P));
  }

  template <class Matrix>
  std::shared_ptr<Matrix>
  coarse_operator(const Matrix& A, const Matrix& P, const Matrix& R) const
  {
    return detail::galerkin(A, P, R);
  }

 private:

  /*!
   * \brief Reproducible pseudo random value in [0.5, 1) for point \a i.
   *
   * The measure of a point with \a k active dependents lies in
   * [k + 0.5, k + 1), so that "no dependent" (measure < 1) and "at
   * least one dependent" (measure > 1) are decided without ambiguity.
   * This removes the (probability zero, but possible) deadlock of
   * hypre's [0, 1) augmentation when the random part is exactly zero.
   */
  static double random_measure_part(int seed, size_t i)
  {
    std::uint64_t z = static_cast<std::uint64_t>(seed) +
    0x9E3779B97F4A7C15ull * (static_cast<std::uint64_t>(i) + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    // Map the 53 high bits to [0, 1), then shift to [0.5, 1).
    return 0.5 + 0.5 * (static_cast<double>(z >> 11) / 9007199254740992.0);
  }

  /*!
   * \brief CLJP splitting of the variables into C(oarse) and F(ine) sets.
   *
   * Sequential adaptation of hypre_BoomerAMGCoarsen: the strength
   * matrix S (with its transposition in S.ptr and S.col) is not
   * modified; the edge bookkeeping of the original algorithm (edges
   * negated in the column array of S) is kept in the \a active array
   * instead, so that the strength information remains available for
   * the interpolation.
   */
  template <typename Val, typename Col, typename Ptr>
  static void cljp_split(CSRMatrix<Val, Col, Ptr> const& A,
                         CSRMatrix<char, Col, Ptr> const& S,
                         UniqueArray<char>& cf, int seed)
  {
    const size_t n = A.nbRow();

    // The measure of a variable is the number of variables strongly
    // depending on it (the row size of the transposed strength matrix),
    // augmented with a pseudo random value in [0.5, 1).
    UniqueArray<double> measure(n);
    for (size_t i = 0; i < n; ++i) {
      double influences = static_cast<double>(S.ptr[i + 1] - S.ptr[i]);
      measure[i] = influences + random_measure_part(seed, i);
    }

    // State of the strength edges: an edge is deactivated once the
    // corresponding dependency has been accounted for.
    UniqueArray<char> active;
    {
      const size_t nnz = backend::nonzeros(A);
      active.resize(nnz);
      std::copy(S.val.data(), S.val.data() + nnz, active.data());
    }

    // Undecided points.
    UniqueArray<Col> graph(n);
    size_t graph_size = 0;
    for (size_t i = 0; i < n; ++i)
      if (cf[i] == 'U')
        graph[graph_size++] = static_cast<Col>(i);

    // is[i] is set when i is picked in the current independent set.
    UniqueArray<char> is(n, 0);

    while (graph_size > 0) {
      // 1. Variables with no remaining influence become fine points,
      //    provided all their dependencies are accounted for.
      for (size_t ig = 0; ig < graph_size;) {
        Col i = graph[ig];

        if (cf[i] == 'U' && measure[i] < 1.0) {
          bool resolved = true;
          for (Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
            if (active[j]) {
              resolved = false;
              break;
            }
          }
          if (resolved)
            cf[i] = 'F';
        }

        if (cf[i] != 'U') {
          // Take the variable out of the undecided set.
          graph[ig] = graph[--graph_size];
          measure[i] = 0.0;
          is[i] = 0;
          continue;
        }

        ++ig;
      }

      if (graph_size == 0)
        break;

      // 2. Pick an independent set of variables with maximal measure.
      //    Ties are broken by variable number so that the result is a
      //    proper independent set even for equal measures.
      for (size_t ig = 0; ig < graph_size; ++ig) {
        Col i = graph[ig];
        is[i] = (measure[i] > 1.0);
      }

      for (size_t ig = 0; ig < graph_size; ++ig) {
        Col i = graph[ig];
        if (!is[i])
          continue;

        for (Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
          if (!S.val[j])
            continue;
          Col c = A.col[j];
          if (!is[c])
            continue;

          if (measure[i] > measure[c] || (measure[i] == measure[c] && i < c))
            is[c] = 0;
          else {
            is[i] = 0;
            break;
          }
        }
      }

      // 3. The independent set becomes coarse points; the measures of
      //    their strong neighbours are decremented.
      for (size_t ig = 0; ig < graph_size; ++ig) {
        Col i = graph[ig];
        if (!is[i])
          continue;

        cf[i] = 'C';

        for (Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
          if (!active[j])
            continue;
          active[j] = 0;
          Col c = A.col[j];
          if (cf[c] == 'U')
            measure[c] -= 1.0;
        }
      }

      // 4. Heuristics for the remaining undecided variables: a
      //    dependency on a new coarse point, or on a neighbour sharing
      //    a common new coarse point, is accounted for.
      UniqueArray<Col> common_c;
      for (size_t ig = 0; ig < graph_size; ++ig) {
        Col i = graph[ig];
        if (is[i] || cf[i] != 'U')
          continue;

        // 4a. Remove the edges to the new coarse points, remembering
        //     them as common coarse point candidates.
        common_c.clear();
        for (Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
          if (active[j] && is[A.col[j]]) {
            common_c.push_back(A.col[j]);
            active[j] = 0;
          }
        }

        // 4b. Remove the edges to neighbours strongly depending on one
        //     of the common coarse points of i.
        for (Ptr j = A.ptr[i], e = A.ptr[i + 1]; j < e; ++j) {
          if (!active[j])
            continue;
          Col c = A.col[j];

          bool has_common = false;
          for (Ptr k = A.ptr[c], ec = A.ptr[c + 1]; k < ec; ++k) {
            if (!S.val[k])
              continue;
            for (Col cc : common_c) {
              if (A.col[k] == cc) {
                has_common = true;
                break;
              }
            }
            if (has_common)
              break;
          }

          if (has_common) {
            active[j] = 0;
            if (cf[c] == 'U')
              measure[c] -= 1.0;
          }
        }
      }
    }
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
struct coarsening_is_supported<Backend, CLJPCoarsening,
                               typename std::enable_if<!std::is_arithmetic<typename backend::value_type<Backend>::type>::value>::type> : std::false_type
{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::backend

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
