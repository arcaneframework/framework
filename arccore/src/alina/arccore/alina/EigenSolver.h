// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EigenAdapter.h                                              (C) 2000-2026 */
/*                                                                           */
/* Wrapper around eigen direct solvers.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_EIGENSOLVER_H
#define ARCCORE_ALINA_EIGENSOLVER_H
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

// Remove warnings for Eigen
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wint-in-bool-context"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <type_traits>

#include "arccore/alina/BuiltinBackend.h"
#include "arccore/alina/AlinaUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Wrapper around eigen direct solvers.
 */
template <class Solver>
class EigenSolver
{
 public:

  typedef typename Solver::MatrixType MatrixType;
  typedef typename Solver::Scalar value_type;

  typedef Alina::detail::empty_params params;

  static size_t coarse_enough()
  {
    return 3000 / math::static_rows<value_type>::value;
  }

  template <class Matrix>
  EigenSolver(const Matrix& A, const params& = params())
  : n(backend::nbRow(A))
  {
    typedef typename std::remove_const<typename std::remove_pointer<typename backend::col_data_impl<Matrix>::type>::type>::type col_type;
    typedef typename std::remove_const<typename std::remove_pointer<typename backend::ptr_data_impl<Matrix>::type>::type>::type ptr_type;

    S.compute(MatrixType(Eigen::Map<Eigen::SparseMatrix<value_type, Eigen::RowMajor, ptrdiff_t>>(
    backend::nbRow(A), backend::nbColumn(A), backend::nonzeros(A),
    const_cast<ptr_type*>(backend::ptr_data(A)),
    const_cast<col_type*>(backend::col_data(A)),
    const_cast<value_type*>(backend::val_data(A)))));
  }

  template <class Vec1, class Vec2>
  void operator()(const Vec1& rhs, Vec2& x) const
  {
    Eigen::Map<Eigen::Matrix<value_type, Eigen::Dynamic, 1>>
    RHS(const_cast<value_type*>(&rhs[0]), n), X(&x[0], n);

    X = S.solve(RHS);
  }

  friend std::ostream& operator<<(std::ostream& os, const EigenSolver& s)
  {
    return os << "eigen: " << s.n << " unknowns";
  }

 private:

  ptrdiff_t n;
  Solver S;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
