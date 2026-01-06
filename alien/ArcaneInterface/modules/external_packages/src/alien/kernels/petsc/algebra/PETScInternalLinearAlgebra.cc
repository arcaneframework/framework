// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "PETScInternalLinearAlgebra.h"

#include <alien/kernels/petsc/PETScBackEnd.h>

#include <alien/core/backend/LinearAlgebraT.h>
#include <alien/data/Space.h>
#include <alien/kernels/petsc/data_structure/PETScInit.h>
#include <alien/kernels/petsc/data_structure/PETScInternal.h>
#include <alien/kernels/petsc/data_structure/PETScMatrix.h>
#include <alien/kernels/petsc/data_structure/PETScVector.h>

#include <alien/kernels/petsc/linear_solver/PETScInternalLinearSolver.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

template class ALIEN_EXTERNAL_PACKAGES_EXPORT LinearAlgebra<BackEnd::tag::petsc>;

extern ALIEN_EXTERNAL_PACKAGES_EXPORT IInternalLinearAlgebra<PETScMatrix, PETScVector>*
PETScInternalLinearAlgebraFactory(Arccore::MessagePassing::IMessagePassingMng* pm)
{
  return new PETScInternalLinearAlgebra(pm);
}

/*---------------------------------------------------------------------------*/

PETScInternalLinearAlgebra::PETScInternalLinearAlgebra(
    Arccore::MessagePassing::IMessagePassingMng* pm)
{
  if(not PETScInternalLinearSolver::m_library_plugin_is_initialized)
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSC Library should be initialized first");

  if (pm != nullptr)
  {
    auto mpi_mng = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(pm);
    if(mpi_mng)
      PETSC_COMM_WORLD = *static_cast<const MPI_Comm*>(mpi_mng->getMPIComm());

  }
}

/*---------------------------------------------------------------------------*/

PETScInternalLinearAlgebra::~PETScInternalLinearAlgebra()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arccore::Real
PETScInternalLinearAlgebra::norm0(const PETScVector& vx) const
{
  Arccore::Real result = 0;
  VecNorm(vx.internal()->m_internal, NORM_INFINITY, &result);
  return result;
}

/*---------------------------------------------------------------------------*/

Arccore::Real
PETScInternalLinearAlgebra::norm1(const PETScVector& vx) const
{
  Arccore::Real result = 0;
  VecNorm(vx.internal()->m_internal, NORM_1, &result);
  return result;
}

/*---------------------------------------------------------------------------*/

Arccore::Real
PETScInternalLinearAlgebra::norm2(const PETScVector& vx) const
{
  Arccore::Real result = 0;
  VecNorm(vx.internal()->m_internal, NORM_2, &result);
  return result;
}


Arccore::Real
PETScInternalLinearAlgebra::normInf(const PETScVector& vx) const
{
  Arccore::Real result = 0;
  VecNorm(vx.internal()->m_internal, NORM_INFINITY, &result);
  return result;
}
/*---------------------------------------------------------------------------*/

void
PETScInternalLinearAlgebra::mult(
    const PETScMatrix& ma, const PETScVector& vx, PETScVector& vr) const
{
  MatMult(
      ma.internal()->m_internal, vx.internal()->m_internal, vr.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

void
PETScInternalLinearAlgebra::axpy(
    Real alpha, const PETScVector& vx, PETScVector& vr) const
{
  VecAXPY(vr.internal()->m_internal, alpha, vx.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

void
PETScInternalLinearAlgebra::aypx(
    Real alpha, PETScVector& vy, const PETScVector& vx) const
{
  VecAYPX(vy.internal()->m_internal, alpha, vx.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

void
PETScInternalLinearAlgebra::copy(const PETScVector& vx, PETScVector& vr) const
{
  VecCopy(vx.internal()->m_internal, vr.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

Arccore::Real
PETScInternalLinearAlgebra::dot(const PETScVector& vx, const PETScVector& vy) const
{
  Arccore::Real result = 0;
  VecDot(vx.internal()->m_internal, vy.internal()->m_internal, &result);
  return result;
}

/*---------------------------------------------------------------------------*/

void
PETScInternalLinearAlgebra::scal(Real alpha, PETScVector& vx) const
{
  VecScale(vx.internal()->m_internal, alpha);
}

/*---------------------------------------------------------------------------*/

void
PETScInternalLinearAlgebra::diagonal(const PETScMatrix& a, PETScVector& x) const
{
  MatGetDiagonal(a.internal()->m_internal, x.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

void
PETScInternalLinearAlgebra::reciprocal(PETScVector& x) const
{
  VecReciprocal(x.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

void
PETScInternalLinearAlgebra::pointwiseMult(
    const PETScVector& x, const PETScVector& y, PETScVector& w) const
{
  VecPointwiseMult(
      w.internal()->m_internal, x.internal()->m_internal, y.internal()->m_internal);
}

/*---------------------------------------------------------------------------*/

void
PETScInternalLinearAlgebra::mult(const Matrix&, const UniqueArray<Real>&, UniqueArray<Real>&) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::mult not implemented");
}
void
PETScInternalLinearAlgebra::axpy(Real, const UniqueArray<Real>&, UniqueArray<Real>&) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::axpy not implemented");
}
void
PETScInternalLinearAlgebra::aypx(Real, UniqueArray<Real>&, const UniqueArray<Real>&) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::aypx not implemented");
}
void
PETScInternalLinearAlgebra::copy(const UniqueArray<Real>&, UniqueArray<Real>&) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::copy not implemented");
}
Real
PETScInternalLinearAlgebra::dot(Integer, const UniqueArray<Real>&, const UniqueArray<Real>&) const
{
  throw NotImplementedException(A_FUNCINFO, "LinearAlgebra::dot not implemented");
}
void
PETScInternalLinearAlgebra::scal(Real, UniqueArray<Real>&) const
{
  throw NotImplementedException(A_FUNCINFO, "HypreLinearAlgebra::scal not implemented");
}

} // END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
