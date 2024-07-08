﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ALIEN_KERNELS_HYPRE_ALGEBRA_HYPREINTERNALLINEARALGEBRA_H
#define ALIEN_KERNELS_HYPRE_ALGEBRA_HYPREINTERNALLINEARALGEBRA_H

#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/core/backend/IInternalLinearAlgebraT.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNAL_PACKAGES_EXPORT HypreInternalLinearAlgebra
    : public IInternalLinearAlgebra<HypreMatrix, HypreVector>
{
 public:
  HypreInternalLinearAlgebra();
  virtual ~HypreInternalLinearAlgebra();

 public:
  // IInternalLinearAlgebra interface.
  Arccore::Real norm0(const Vector& x) const;
  Arccore::Real norm1(const Vector& x) const;
  Arccore::Real norm2(const Vector& x) const;
  Arccore::Real normInf(const Vector& x) const;
  void mult(const Matrix& a, const Vector& x, Vector& r) const;
  void axpy(Real alpha, const Vector& x, Vector& r) const;
  void aypx(Real alpha, Vector& y, const Vector& x) const;
  void copy(const Vector& x, Vector& r) const;
  Arccore::Real dot(const Vector& x, const Vector& y) const;
  void scal(Real alpha, Vector& x) const;
  void diagonal(const Matrix& a, Vector& x) const;
  void reciprocal(Vector& x) const;
  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const;

  void mult(const Matrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  void axpy(Real alpha, const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  void aypx(Real alpha, UniqueArray<Real>& y, const UniqueArray<Real>& x) const;
  void copy(const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  Real dot(
      Integer local_size, const UniqueArray<Real>& x, const UniqueArray<Real>& y) const;
  void scal(Real alpha, UniqueArray<Real>& x) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_KERNELS_HYPRE_ALGEBRA_HYPREINTERNALLINEARALGEBRA_H */
