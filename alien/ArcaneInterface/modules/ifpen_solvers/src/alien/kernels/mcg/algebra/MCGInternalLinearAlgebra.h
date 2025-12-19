// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/utils/Precomp.h>

#include <alien/kernels/mcg/MCGBackEnd.h>
#include <alien/core/backend/IInternalLinearAlgebraT.h>
#include <alien/expression/solver/ILinearAlgebra.h>

namespace Alien {

class MCGInternalLinearAlgebra : public ILinearAlgebra
{
 private:
  typedef Alien::IMatrix MatrixType;
  typedef Alien::IVector VectorType;
  typedef MCGMatrix<Real,MCGInternal::eMemoryDomain::Host> MatrixImpl;
  typedef MCGVector<Real,MCGInternal::eMemoryDomain::Host> VectorImpl;

 public:
  MCGInternalLinearAlgebra();

  virtual ~MCGInternalLinearAlgebra();

 public:
  Real norm0(const VectorType& x) const;
  Real norm1(const VectorType& x) const;
  Real norm2(const VectorType& x) const;
  Real normInf(const VectorType& x) const;
  void mult(const MatrixType& a, const VectorType& x, VectorType& r) const;
  void axpy(Real alpha, const VectorType& x, VectorType& r) const;
  void copy(const VectorType& x, VectorType& r) const;
  Real dot(const VectorType& x, const VectorType& y) const;

 public:
  Real norm0(const VectorImpl& x) const;
  Real norm1(const VectorImpl& x) const;
  Real norm2(const VectorImpl& x) const;
  Real normInf(const VectorImpl& x) const;
  void mult(const MatrixImpl& a, const VectorImpl& x, VectorImpl& r) const;
  void axpy(Real alpha, const VectorImpl& x, VectorImpl& r) const;
  void copy(const VectorImpl& x, VectorImpl& r) const;
  Real dot(const VectorImpl& x, const VectorImpl& y) const;
};
} // namespace Alien

