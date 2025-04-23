// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "alien/kernels/trilinos/TrilinosPrecomp.h"
#include <alien/kernels/trilinos/TrilinosBackEnd.h>

#include <alien/kernels/simple_csr/algebra/CBLASMPIKernel.h>
#include <alien/core/backend/LinearAlgebraT.h>

#include <alien/data/Space.h>

#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
#include <alien/kernels/trilinos/data_structure/TrilinosVector.h>
#include <alien/kernels/trilinos/algebra/TrilinosInternalLinearAlgebra.h>

#include <arccore/base/NotImplementedException.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
#ifdef KOKKOS_ENABLE_SERIAL
template class ALIEN_TRILINOS_EXPORT LinearAlgebra<BackEnd::tag::tpetraserial>;
#endif
IInternalLinearAlgebra<TrilinosMatrixType, TrilinosVectorType>*
TrilinosInternalLinearAlgebraFactory(Arccore::MessagePassing::IMessagePassingMng* pm)
{
  return new TrilinosInternalLinearAlgebra(pm);
}

#ifdef KOKKOS_ENABLE_OPENMP
template class ALIEN_TRILINOS_EXPORT LinearAlgebra<BackEnd::tag::tpetraomp>;

IInternalLinearAlgebra<TpetraOmpMatrixType, TpetraOmpVectorType>*
TpetraOmpInternalLinearAlgebraFactory(Arccore::MessagePassing::IMessagePassingMng* pm)
{
  return new TpetraOmpInternalLinearAlgebra(pm);
}
#endif

#ifdef KOKKOS_ENABLE_THREADS
template class ALIEN_TRILINOS_EXPORT LinearAlgebra<BackEnd::tag::tpetrapth>;

IInternalLinearAlgebra<TpetraPthMatrixType, TpetraPthVectorType>*
TpetraPthInternalLinearAlgebraFactory(Arccore::MessagePassing::IMessagePassingMng* pm)
{
  return new TpetraPthInternalLinearAlgebra(pm);
}
#endif
/*---------------------------------------------------------------------------*/
#ifdef KOKKOS_ENABLE_CUDA
template class ALIEN_TRILINOS_EXPORT LinearAlgebra<BackEnd::tag::tpetracuda>;

IInternalLinearAlgebra<TpetraCudaMatrixType, TpetraCudaVectorType>*
TpetraCudaInternalLinearAlgebraFactory(Arccore::MessagePassing::IMessagePassingMng* pm)
{
  return new TpetraCudaInternalLinearAlgebra(pm);
}
#endif

/*---------------------------------------------------------------------------*/

Real
TrilinosInternalLinearAlgebra::norm0([[maybe_unused]] const Vector& x) const
{
  return 0.;
}

/*---------------------------------------------------------------------------*/

Real
TrilinosInternalLinearAlgebra::norm1(const Vector& x) const
{
  return x.norm1();
}

/*---------------------------------------------------------------------------*/

Real
TrilinosInternalLinearAlgebra::norm2(const Vector& x) const
{
  return x.norm2();
}

Real
TrilinosInternalLinearAlgebra::normInf([[maybe_unused]] const Vector& x) const
{
  return 0.;
}
/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::mult(const Matrix& a, const Vector& x, Vector& r) const
{
  a.mult(x, r);
}
void
TrilinosInternalLinearAlgebra::mult(
    const Matrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const
{
  a.mult(dataPtr(x), dataPtr(r));
}

/*---------------------------------------------------------------------------*/
void
TrilinosInternalLinearAlgebra::axpy(
    Real alpha, const UniqueArray<Real>& x, UniqueArray<Real>& r) const
{
  cblas::axpy(x.size(), alpha, dataPtr(x), 1, dataPtr(r), 1);
}

void
TrilinosInternalLinearAlgebra::axpy([[maybe_unused]] Real alpha,[[maybe_unused]] const Vector& x,[[maybe_unused]] Vector& r) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/
void
TrilinosInternalLinearAlgebra::aypx(
    [[maybe_unused]] Real alpha, [[maybe_unused]] UniqueArray<Real>& y,[[maybe_unused]] const UniqueArray<Real>& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
}

void
TrilinosInternalLinearAlgebra::aypx([[maybe_unused]] Real alpha,[[maybe_unused]] Vector& y,[[maybe_unused]] const Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::copy(
    [[maybe_unused]] const UniqueArray<Real>& x,[[maybe_unused]] UniqueArray<Real>& r) const
{
  cblas::copy(x.size(), dataPtr(x), 1, dataPtr(r), 1);
}
void
TrilinosInternalLinearAlgebra::copy([[maybe_unused]] const Vector& x,[[maybe_unused]] Vector& r) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/
Real
TrilinosInternalLinearAlgebra::dot(
    Integer local_size, const UniqueArray<Real>& vx, const UniqueArray<Real>& vy) const
{
  return cblas::dot(local_size, dataPtr(vx), 1, dataPtr(vy), 1);
}
Real
TrilinosInternalLinearAlgebra::dot(const Vector& x, const Vector& y) const
{
  return x.dot(y);
}

/*---------------------------------------------------------------------------*/
void
TrilinosInternalLinearAlgebra::scal([[maybe_unused]] Real alpha,[[maybe_unused]]  UniqueArray<Real>& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::scal not implemented");
}

void
TrilinosInternalLinearAlgebra::scal([[maybe_unused]] Real alpha,[[maybe_unused]] Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::scal not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::diagonal([[maybe_unused]] const Matrix& a,[[maybe_unused]]  Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::diagonal not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::reciprocal([[maybe_unused]] Vector& x) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::reciprocal not implemented");
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalLinearAlgebra::pointwiseMult(
    [[maybe_unused]] const Vector& x,[[maybe_unused]]  const Vector& y,[[maybe_unused]]  Vector& w) const
{
  throw NotImplementedException(
      A_FUNCINFO, "TrilinosInternalLinearAlgebra::pointwiseMult not implemented");
}

void
TrilinosInternalLinearAlgebra::dump(Matrix const& a, std::string const& filename) const
{
  a.dump(filename);
}

void
TrilinosInternalLinearAlgebra::dump(Vector const& x, std::string const& filename) const
{
  x.dump(filename);
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
