/*
 * Copyright 2020 IFPEN-CEA
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * SPDX-License-Identifier: Apache-2.0
 */

#include "matrix.h"
#include "vector.h"

#include <cmath>

#include <petscksp.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
#include <arccore/base/NotImplementedException.h>
#include <arccore/base/TraceInfo.h>

#include <alien/core/backend/LinearAlgebraT.h>

#include <alien/petsc/backend.h>
#include <alien/petsc/export.h>


namespace Alien
{
  template class ALIEN_PETSC_EXPORT LinearAlgebra<BackEnd::tag::petsc>;
} // namespace Alien

namespace Alien::PETSc
{
class ALIEN_PETSC_EXPORT InternalLinearAlgebra
: public IInternalLinearAlgebra<Matrix, Vector>
{
 public:
  InternalLinearAlgebra() {}

  virtual ~InternalLinearAlgebra() {}

 public:
  // IInternalLinearAlgebra interface.
  Arccore::Real norm0(const Vector& x) const;

  Arccore::Real norm1(const Vector& x) const;

  Arccore::Real norm2(const Vector& x) const;

  void mult(const Matrix& a, const Vector& x, Vector& r) const;

  void axpy(const Arccore::Real& alpha, const Vector& x, Vector& r) const;

  void aypx(const Arccore::Real& alpha, Vector& y, const Vector& x) const;

  void copy(const Vector& x, Vector& r) const;

  Arccore::Real dot(const Vector& x, const Vector& y) const;

  void scal(const Arccore::Real& alpha, Vector& x) const;

  void diagonal(const Matrix& a, Vector& x) const;

  void reciprocal(Vector& x) const;

  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const;
};

Arccore::Real
InternalLinearAlgebra::norm0(const Vector& vx ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::norm0 not implemented");
}

Arccore::Real
InternalLinearAlgebra::norm1(const Vector& vx ALIEN_UNUSED_PARAM) const
{
  PetscScalar norm;
  VecNorm(vx.internal(),NORM_1,&norm);
  return static_cast<Arccore::Real>(norm);
}

Arccore::Real
InternalLinearAlgebra::norm2(const Vector& vx) const
{
  PetscScalar norm;
  VecNorm(vx.internal(),NORM_2,&norm);
  return static_cast<Arccore::Real>(norm);
}

void InternalLinearAlgebra::mult(const Matrix& ma, const Vector& vx, Vector& vr) const
{
  MatMult(ma.internal(),vx.internal(),vr.internal());
}

void InternalLinearAlgebra::axpy(
const Arccore::Real& alpha ALIEN_UNUSED_PARAM, const Vector& vx ALIEN_UNUSED_PARAM, Vector& vr ALIEN_UNUSED_PARAM) const
{
  VecAXPY(vr.internal(), alpha, vx.internal()); //  vr = alpha.vx + vr 
}

void InternalLinearAlgebra::copy(const Vector& vx /*src*/, Vector& vr /*dest*/) const
{
  VecCopy(vx.internal(),vr.internal());
}

Arccore::Real
InternalLinearAlgebra::dot(const Vector& vx, const Vector& vy) const
{
  PetscScalar dot_prod;
  VecDot(vx.internal(),vy.internal(), &dot_prod);
  return static_cast<Arccore::Real>(dot_prod);
}

void InternalLinearAlgebra::diagonal(Matrix const& m ALIEN_UNUSED_PARAM, Vector& v ALIEN_UNUSED_PARAM) const
{
  MatGetDiagonal(m.internal(),v.internal());
}

void InternalLinearAlgebra::reciprocal(Vector& v ALIEN_UNUSED_PARAM) const
{
  VecReciprocal(v.internal()); 
}

void InternalLinearAlgebra::aypx(
const double& alpha ALIEN_UNUSED_PARAM, Vector& y ALIEN_UNUSED_PARAM, const Vector& x ALIEN_UNUSED_PARAM) const
{
  VecAYPX(y.internal(),alpha,x.internal()); // y = x + alpha y
}

void InternalLinearAlgebra::pointwiseMult(
const Vector& x ALIEN_UNUSED_PARAM, const Vector& y ALIEN_UNUSED_PARAM, Vector& w ALIEN_UNUSED_PARAM) const
{
  VecPointwiseMult(w.internal(), x.internal(), y.internal()); //Computes the componentwise multiplication w = x*y. 
}

void InternalLinearAlgebra::scal(const Arccore::Real& alpha, Vector& x) const
{
  VecScale(x.internal(), alpha);
}

ALIEN_PETSC_EXPORT
IInternalLinearAlgebra<PETSc::Matrix, PETSc::Vector>*
InternalLinearAlgebraFactory()
{
  return new PETSc::InternalLinearAlgebra();
}
} // namespace Alien::Petsc
