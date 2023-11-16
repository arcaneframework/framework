// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup expression
 * \brief Schur elimination operator on a linear system
 */
class ALIEN_EXPORT SchurOp
{
 public:
  //! Type of algorithm

  //! Type of the error
  enum eErrorType
  {
    NoError,
    WithErrors
  };

  //! Type of the matrix implementation
  typedef SimpleCSRMatrix<Arccore::Real> MatrixImpl;
  //! Type of the vector implementation
  typedef SimpleCSRVector<Arccore::Real> VectorImpl;

  //! Constructor
  SchurOp(IMatrix& A, IVector& b);

  //! Free resources
  virtual ~SchurOp() {}

  /*!
   * \brief Shur the linear system
   * \param[in] matrix The matrix
   * \param[in] vector The vector
   * \returns The eventual error
   */
  eErrorType computePrimarySystem(IMatrix& pA, IVector& pb) const;

  eErrorType computeSolutionFromPrimaryUnknowns(IVector const& pX, IVector& sX) const;

 private:
  eErrorType _apply_schur(Integer block_size,
                          MatrixImpl& A,
                          VectorImpl& B,
                          Integer p_block_size,
                          MatrixImpl& pA,
                          VectorImpl& pB) const;

  eErrorType _apply_schur(Integer block_size,
                          MatrixImpl& A,
                          VectorImpl& B,
                          VBlock const* p_vblock,
                          MatrixImpl& pA,
                          VectorImpl& pB) const;

  eErrorType _apply_schur(VBlock const* vblock,
                          MatrixImpl& A,
                          VectorImpl& B,
                          Integer p_block_size,
                          MatrixImpl& pA,
                          VectorImpl& pB) const;

  eErrorType _apply_schur(VBlock const* vblock,
                          MatrixImpl& A,
                          VectorImpl& B,
                          VBlock const* p_vblock,
                          MatrixImpl& pA,
                          VectorImpl& pB) const;

  eErrorType _compute_solution(VBlock const* vblock,
                               MatrixImpl const& A,
                               VectorImpl const& B,
                               Integer p_block_size,
                               VectorImpl const& px,
                               VectorImpl& x) const;

  eErrorType _compute_solution(VBlock const* vblock,
                               MatrixImpl const& A,
                               VectorImpl const& B,
                               VBlock const* p_vblock,
                               VectorImpl const& px,
                               VectorImpl& x) const;

  void _copy(ConstArrayView<Real> in, ArrayView<Real> out) const
  {
    for (Integer i = 0; i < out.size(); ++i)
      out(i) = in(i);
  }

  void _copy(ConstArray2View<Real> in, Array2View<Real> out) const
  {
    for (Integer i = 0; i < out.dim1Size(); ++i)
      for (Integer j = 0; j < out.dim2Size(); ++j)
        out(i, j) = in(i, j);
  }

  IMatrix& m_A;

  IVector& m_B;

  mutable UniqueArray<Real> m_ghost_diag_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
