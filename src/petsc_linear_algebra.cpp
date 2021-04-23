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

// FIXME: use public API for Hypre !
//#include <_hypre_parcsr_mv.h>
#include <petscksp.h> // checkerror ... ?

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
#include <arccore/base/NotImplementedException.h>
#include <arccore/base/TraceInfo.h>

#include <alien/core/backend/LinearAlgebraT.h>

#include <alien/petsc/backend.h>
#include <alien/petsc/export.h>


namespace Alien
{
namespace
{/*
  HYPRE_ParVector hypre_implem(const Hypre::Vector& v)
  {
    HYPRE_ParVector res;
    HYPRE_IJVectorGetObject(v.internal(), reinterpret_cast<void**>(&res));
    return res;
  }
*/

  /*HYPRE_ParCSRMatrix hypre_implem(const Hypre::Matrix& m)
  {
    HYPRE_ParCSRMatrix res;
    HYPRE_IJMatrixGetObject(m.internal(), reinterpret_cast<void**>(&res));
    return res;
  }*/  
} // namespace

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
  throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::norm1 not implemented");
}

Arccore::Real
InternalLinearAlgebra::norm2(const Vector& vx) const
{
  return std::sqrt(dot(vx, vx));
}

void InternalLinearAlgebra::mult(const Matrix& ma, const Vector& vx, Vector& vr) const
{
  MatMult(ma.internal(),vx.internal(),vr.internal());
}

void InternalLinearAlgebra::axpy(
const Arccore::Real& alpha ALIEN_UNUSED_PARAM, const Vector& vx ALIEN_UNUSED_PARAM, Vector& vr ALIEN_UNUSED_PARAM) const
{
 // HYPRE_ParVectorAxpy(alpha, hypre_implem(vx), hypre_implem(vr));
   VecAXPY(vr.internal(), alpha, vx.internal()); //  vr = alpha.vx + vr 
   //throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::axpy not implemented");
}

void InternalLinearAlgebra::copy(const Vector& vx /*src*/, Vector& vr /*dest*/) const
{
 // HYPRE_ParVectorCopy(hypre_implem(vx), hypre_implem(vr));
    //PetscErrorCode ierr;
    /*ierr = */VecCopy(vx.internal(),vr.internal());
    //CHKERRQ(ierr);    
    //throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::copy not implemented");
}

Arccore::Real
InternalLinearAlgebra::dot(const Vector& vx, const Vector& vy) const
{
/*  double dot_prod = 0;
  HYPRE_ParVectorInnerProd(hypre_implem(vx), hypre_implem(vy), &dot_prod);
  return static_cast<Arccore::Real>(dot_prod);*/
  
  PetscScalar dot_prod;
  VecDot(vx.internal(),vy.internal(), &dot_prod);
  return static_cast<Arccore::Real>(dot_prod);
    
  //throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::dot not implemented");  
}

void InternalLinearAlgebra::diagonal(Matrix const& m ALIEN_UNUSED_PARAM, Vector& v ALIEN_UNUSED_PARAM) const
{
  /*throw Arccore::NotImplementedException(
  A_FUNCINFO, "HypreLinearAlgebra::diagonal not implemented");*/
  throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::diagonal not implemented");

}

void InternalLinearAlgebra::reciprocal(Vector& v ALIEN_UNUSED_PARAM) const
{
  /*throw Arccore::NotImplementedException(
  A_FUNCINFO, "HypreLinearAlgebra::reciprocal not implemented");*/
  throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::reciprocal not implemented");
}

void InternalLinearAlgebra::aypx(
const double& alpha ALIEN_UNUSED_PARAM, Vector& y ALIEN_UNUSED_PARAM, const Vector& x ALIEN_UNUSED_PARAM) const
{
  /*throw Arccore::NotImplementedException(A_FUNCINFO, "HypreLinearAlgebra::aypx not implemented");*/
  throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::aypx not implemented");  
}

void InternalLinearAlgebra::pointwiseMult(
const Vector& x ALIEN_UNUSED_PARAM, const Vector& y ALIEN_UNUSED_PARAM, Vector& w ALIEN_UNUSED_PARAM) const
{
  /*throw Arccore::NotImplementedException(
  A_FUNCINFO, "HypreLinearAlgebra::pointwiseMult not implemented");*/
  throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::pointwiseMult not implemented");
}

void InternalLinearAlgebra::scal(const Arccore::Real& alpha, Vector& x) const
{
  /*HYPRE_ParVectorScale(static_cast<double>(alpha), hypre_implem(x));*/
  throw Arccore::NotImplementedException(A_FUNCINFO, "PetscLinearAlgebra::scal not implemented");
}

ALIEN_PETSC_EXPORT
IInternalLinearAlgebra<PETSc::Matrix, PETSc::Vector>*
InternalLinearAlgebraFactory()
{
  return new PETSc::InternalLinearAlgebra();
}
} // namespace Alien::Petsc
