// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

namespace Alien::SimpleCSRInternal
{

/*@! Classe amie de SimpleCSRMatrix pour externaliser plus rapidement (mais moins
 * proprement)
 * le produit matrice vecteur */
template <typename ValueT>
class SimpleCSRMatrixMultT
{
 public:
  //! Template parameter
  typedef ValueT ValueType;
  typedef SimpleCSRMatrix<ValueType> MatrixType;
  typedef SimpleCSRVector<ValueType> VectorType;

 public:
  //! Constructeur de la classe
  SimpleCSRMatrixMultT(const MatrixType& matrix);

  //! Destructeur de la classe
  virtual ~SimpleCSRMatrixMultT() {}

 public:
  void synchronize(VectorType& x) const;

  //! Matrix vector product
  void mult(const VectorType& x, VectorType& y) const;
  void mult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const;

  void addLMult(Real alpha, const VectorType& x, VectorType& y) const;
  void addUMult(Real alpha, const VectorType& x, VectorType& y) const;

  void computeDiag(VectorType& y) const;
  void multDiag(VectorType& y) const;
  void computeInvDiag(VectorType& y) const;
  void multInvDiag(VectorType& y) const;

 private:
  void _synchronize(VectorType& x) const;
  void _parallelMult(const VectorType& x, VectorType& y) const;
  void _parallelMult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const;

  void _seqMult(const VectorType& x, VectorType& y) const;
  void _seqMult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const;

  void _seqAddLMult(Real alpha, const VectorType& x, VectorType& y) const;
  void _seqAddUMult(Real alpha, const VectorType& x, VectorType& y) const;

  void _synchronizeBlock(VectorType& x) const;
  void _parallelMultBlock(const VectorType& x, VectorType& y) const;

  void _seqMultBlock(const VectorType& x, VectorType& y) const;

  void _synchronizeVariableBlock(VectorType& x) const;
  void _parallelMultVariableBlock(const VectorType& x, VectorType& y) const;

  void _seqMultVariableBlock(const VectorType& x, VectorType& y) const;

 private:
  const MatrixType& m_matrix_impl;
};

} // namespace Alien::SimpleCSRInternal

#include "SimpleCSRMatrixMultT.h"
