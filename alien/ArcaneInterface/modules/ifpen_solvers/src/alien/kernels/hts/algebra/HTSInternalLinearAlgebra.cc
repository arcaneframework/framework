// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "HTSInternalLinearAlgebra.h"

#include <alien/kernels/hts/HTSBackEnd.h>

#include <alien/core/backend/LinearAlgebraT.h>
#include <alien/kernels/simple_csr/algebra/CBLASMPIKernel.h>

#include <alien/data/Space.h>

#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>

#include <alien/kernels/hts/data_structure/HTSMatrix.h>

#include <arccore/base/NotImplementedException.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

template class ALIEN_IFPEN_SOLVERS_EXPORT LinearAlgebra<BackEnd::tag::htssolver>;
template class ALIEN_IFPEN_SOLVERS_EXPORT
    LinearAlgebra<BackEnd::tag::hts, BackEnd::tag::simplecsr>;

/*---------------------------------------------------------------------------*/
IInternalLinearAlgebra<SimpleCSRMatrix<Real>, SimpleCSRVector<Real>>*
HTSSolverInternalLinearAlgebraFactory()
{
  return new HTSSolverInternalLinearAlgebra();
}

IInternalLinearAlgebra<HTSMatrix<Real>, SimpleCSRVector<Real>>*
HTSInternalLinearAlgebraFactory()
{
  return new HTSInternalLinearAlgebra();
}

/*---------------------------------------------------------------------------*/
HTSInternalLinearAlgebra::HTSInternalLinearAlgebra()
{
  // Devrait faire le HTSInitialize qui est actuellement dans le solveur
  // Attention, cette initialisation serait globale et non restreinte à cet objet
}

/*---------------------------------------------------------------------------*/

HTSInternalLinearAlgebra::~HTSInternalLinearAlgebra()
{
}

/*---------------------------------------------------------------------------*/

Real
HTSInternalLinearAlgebra::norm0(const Vector& x) const
{
  return CBLASMPIKernel::nrm0(x.distribution(), x);
}

/*---------------------------------------------------------------------------*/

Real
HTSInternalLinearAlgebra::norm1(const Vector& x) const
{
  return CBLASMPIKernel::nrm1(x.distribution(), x);
}

/*---------------------------------------------------------------------------*/

Real
HTSInternalLinearAlgebra::norm2(const Vector& x) const
{
  return CBLASMPIKernel::nrm2(x.distribution(), x);
}

Real
HTSInternalLinearAlgebra::normInf(const Vector& x) const
{
  return CBLASMPIKernel::nrmInf(x.distribution(), x);
}


/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::mult(const Matrix& a, const Vector& x, Vector& r) const
{
  a.mult(x.getDataPtr(), r.getDataPtr());
}
void
HTSInternalLinearAlgebra::mult(
    const Matrix& a, const UniqueArray<Real>& vx, UniqueArray<Real>& vr) const
{
  a.mult(dataPtr(vx), dataPtr(vr));
}

/*---------------------------------------------------------------------------*/
void
HTSInternalLinearAlgebra::axpy(
    Real alpha, const UniqueArray<Real>& x, UniqueArray<Real>& r) const
{
  cblas::axpy(x.size(), alpha, dataPtr(x), 1, dataPtr(r), 1);
}

void
HTSInternalLinearAlgebra::axpy(Real alpha, const Vector& x, Vector& r) const
{
  CBLASMPIKernel::axpy(x.distribution(), alpha, x, r);
}

/*---------------------------------------------------------------------------*/
void
HTSInternalLinearAlgebra::aypx(
    Real alpha, UniqueArray<Real>& y, const UniqueArray<Real>& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::aypx not implemented");
}

void
HTSInternalLinearAlgebra::aypx(Real alpha, Vector& y, const Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/
void
HTSInternalLinearAlgebra::copy(const UniqueArray<Real>& x, UniqueArray<Real>& r) const
{
  cblas::copy(x.size(), dataPtr(x), 1, dataPtr(r), 1);
}

void
HTSInternalLinearAlgebra::copy(const Vector& x, Vector& r) const
{
  CBLASMPIKernel::copy(x.distribution(), x, r);
}

/*---------------------------------------------------------------------------*/
Real
HTSInternalLinearAlgebra::dot(
    Integer local_size, const UniqueArray<Real>& vx, const UniqueArray<Real>& vy) const
{
  return cblas::dot(local_size, dataPtr(vx), 1, dataPtr(vy), 1);
}

Real
HTSInternalLinearAlgebra::dot(const Vector& x, const Vector& y) const
{
  return CBLASMPIKernel::dot(x.distribution(), x, y);
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::scal(
    Real alpha ALIEN_UNUSED_PARAM, UniqueArray<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::scal not implemented");
}

void
HTSInternalLinearAlgebra::scal(Real alpha, Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::scal not implemented");
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::diagonal(const Matrix& a, Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::diagonal not implemented");
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::reciprocal(Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::reciprocal not implemented");
}

/*---------------------------------------------------------------------------*/

void
HTSInternalLinearAlgebra::pointwiseMult(const Vector& x, const Vector& y, Vector& w) const
{
  throw NotImplementedException(
      A_FUNCINFO, "HTSInternalLinearAlgebra::pointwiseMult not implemented");
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
