// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

namespace Alien
{
template <typename AlgebraT>
class DiagPreconditioner
{
 public:
  // clang-format off
    typedef AlgebraT                        AlgebraType ;
    typedef typename AlgebraType::Matrix    MatrixType;
    typedef typename AlgebraType::Vector    VectorType;
    typedef typename MatrixType::ValueType  ValueType;
  // clang-format on

  DiagPreconditioner(AlgebraType& algebra,
                     MatrixType const& matrix)
  : m_algebra(algebra)
  , m_matrix(matrix)
  {}

  virtual ~DiagPreconditioner(){};

  //! operator preparation
  void init()
  {
    m_algebra.allocate(AlgebraType::resource(m_matrix), m_inv_diag);
    m_algebra.assign(m_inv_diag, 1.);
    m_algebra.computeInvDiag(m_matrix, m_inv_diag);
  }

  void update()
  {
    // update value from m_matrix
  }

  template <typename AlgebraType>
  void solve(AlgebraType& algebra,
             VectorType const& x,
             VectorType& y) const
  {
    algebra.pointwiseMult(m_inv_diag, x, y);
  }

 private:
  AlgebraType& m_algebra;
  MatrixType const& m_matrix;
  VectorType m_inv_diag;
};

} // namespace Alien
