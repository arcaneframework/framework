// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/kernels/sycl/SYCLPrecomp.h>
#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>

namespace Alien::SYCLInternal
{

/*@! Classe amie de SYCLMatrix pour externaliser plus rapidement (mais moins
 * proprement)
 * le produit matrice vecteur */
template <typename ValueT>
class SYCLBEllPackMatrixMultT
{
 public:
  //! Template parameter
  // clang-format off
  typedef ValueT                        ValueType;
  typedef SYCLBEllPackMatrix<ValueType> MatrixType;
  typedef SYCLVector<ValueType>         VectorType;
  // clang-format on

 public:
  //! Constructeur de la classe
  SYCLBEllPackMatrixMultT(const MatrixType& matrix);

  //! Destructeur de la classe
  virtual ~SYCLBEllPackMatrixMultT() {}

 public:
  //! Matrix vector product
  void mult(const VectorType& x, VectorType& y) const;
  void mult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const;

  void addLMult(Real alpha, const VectorType& x, VectorType& y) const;
  void addUMult(Real alpha, const VectorType& x, VectorType& y) const;

  void multDiag(VectorType const& y,VectorType& z) const;

  void computeDiag(VectorType& y) const;
  void multDiag(VectorType& y) const;

  void computeInvDiag(VectorType& y) const;
  void multInvDiag(VectorType& y) const;

 private:
  void _parallelMult(const VectorType& x, VectorType& y) const;
  void _parallelMult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const;

  void _seqMult(const VectorType& x, VectorType& y) const;
  void _seqMult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const;

 private:
  const MatrixType& m_matrix_impl;
};

} // namespace Alien::SYCLInternal

#include "SYCLBEllPackMatrixMultT.h"
