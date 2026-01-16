// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#include <cmath>
#include <tuple>

#include "SimpleCSRInternalLinearAlgebra.h"
#include <alien/utils/Precomp.h>

#include <arccore/base/NotImplementedException.h>
#include <arccore/base/TraceInfo.h>

#include <alien/core/backend/LinearAlgebraExprT.h>
#include <alien/core/backend/LinearAlgebraT.h>

#include "CBLASMPIKernel.h"
#include "SimpleCSRMatrixMult.h"

/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

template class ALIEN_EXPORT LinearAlgebra<BackEnd::tag::simplecsr>;
template class ALIEN_EXPORT LinearAlgebraExpr<BackEnd::tag::simplecsr>;

ALIEN_EXPORT IInternalLinearAlgebra<SimpleCSRMatrix<Real>, SimpleCSRVector<Real>>*
SimpleCSRInternalLinearAlgebraFactory()
{
  return new SimpleCSRInternalLinearAlgebra();
}

ALIEN_EXPORT IInternalLinearAlgebraExpr<SimpleCSRMatrix<Real>, SimpleCSRVector<Real>>*
SimpleCSRInternalLinearAlgebraExprFactory()
{
  return new SimpleCSRInternalLinearAlgebraExpr();
}

/*---------------------------------------------------------------------------*/

namespace Internal = SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

SimpleCSRInternalLinearAlgebra::SimpleCSRInternalLinearAlgebra()
: IInternalLinearAlgebra<CSRMatrix, CSRVector>()
{}

/*---------------------------------------------------------------------------*/

SimpleCSRInternalLinearAlgebra::~SimpleCSRInternalLinearAlgebra()
{
#ifdef ALIEN_USE_PERF_TIMER
  m_timer.printInfo("SIMPLECSR-ALGEBRA");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
SimpleCSRInternalLinearAlgebra::ResourceType
SimpleCSRInternalLinearAlgebra::resource(Matrix const& A)
{
  return std::make_tuple(&A.distribution().rowDistribution(),A.blockSize());
}

void SimpleCSRInternalLinearAlgebra::allocate(ResourceType resource, Vector& v)
{
  v.init(*std::get<0>(resource),std::get<1>(resource), true);
}

void SimpleCSRInternalLinearAlgebra::free(Vector& v)
{
  v.clear();
}

Real SimpleCSRInternalLinearAlgebra::norm0(const CSRVector& vx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-NORM0");
#endif

  return CBLASMPIKernel::nrm0(vx.distribution(), vx);
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebra::norm1(const CSRVector& vx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-NORM1");
#endif

  return CBLASMPIKernel::nrm1(vx.distribution(), vx);
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebra::norm2(const CSRVector& vx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-NORM2");
#endif
  return CBLASMPIKernel::nrm2(vx.distribution(), vx);
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebra::normInf(const CSRVector& vx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-NORMINF");
#endif
  return CBLASMPIKernel::nrmInf(vx.distribution(), vx);
}

/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebra::synchronize(const CSRMatrix& ma,
                                                 CSRVector& vx) const
{
  Internal::SimpleCSRMatrixMultT<Real>(ma).synchronize(vx);
}

void SimpleCSRInternalLinearAlgebra::mult(const CSRMatrix& ma,
                                          const CSRVector& vx,
                                          CSRVector& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-SPMV");
#endif
  Internal::SimpleCSRMatrixMultT<Real>(ma).mult(vx, vr);
}

void SimpleCSRInternalLinearAlgebra::addLMult(Real alpha,
                                              const CSRMatrix& ma,
                                              const CSRVector& vx,
                                              CSRVector& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-AddLMult");
#endif
  Internal::SimpleCSRMatrixMultT<Real>(ma).addLMult(alpha, vx, vr);
}

void SimpleCSRInternalLinearAlgebra::addUMult(Real alpha,
                                              const CSRMatrix& ma,
                                              const CSRVector& vx,
                                              CSRVector& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-AddUMult");
#endif
  Internal::SimpleCSRMatrixMultT<Real>(ma).addUMult(alpha, vx, vr);
}

void SimpleCSRInternalLinearAlgebra::multInvDiag(const CSRMatrix& ma, CSRVector& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-MULTINVDIAG");
#endif
  Internal::SimpleCSRMatrixMultT<Real>(ma).multInvDiag(vr);
}

void SimpleCSRInternalLinearAlgebra::computeInvDiag(const CSRMatrix& ma, CSRVector& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-INVDIAG");
#endif
  Internal::SimpleCSRMatrixMultT<Real>(ma).computeInvDiag(vr);
}

/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebra::axpy(Real alpha, const CSRVector& vx, CSRVector& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-AXPY");
#endif
  CBLASMPIKernel::axpy(vx.distribution(), alpha, vx, vr);
}

/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebra::aypx(Real alpha ALIEN_UNUSED_PARAM,
                                          CSRVector& y ALIEN_UNUSED_PARAM,
                                          const CSRVector& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebra::copy(const CSRVector& vx, CSRVector& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-COPY");
#endif
  CBLASMPIKernel::copy(vx.distribution(), vx, vr);
}

void SimpleCSRInternalLinearAlgebra::axpy(Real alpha,
                                          const CSRVector& vx,
                                          Integer stride_x,
                                          CSRVector& vr,
                                          Integer stride_r) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-AXPY");
#endif
  CBLASMPIKernel::axpy(vx.distribution(), alpha, vx, stride_x, vr, stride_r);
}

/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebra::aypx([[maybe_unused]] Real alpha,
                                          [[maybe_unused]] CSRVector& y,
                                          [[maybe_unused]] Integer stride_y,
                                          [[maybe_unused]] const CSRVector& x,
                                          [[maybe_unused]] Integer stride_x) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

void SimpleCSRInternalLinearAlgebra::copy(const Vector& vx,
                                          Integer stride_x,
                                          Vector& vr,
                                          Integer stride_r) const
{
  CBLASMPIKernel::copy(vx.distribution(), vx, stride_x, vr, stride_r);
}
/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebra::dot(const CSRVector& vx, const CSRVector& vy) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-DOT");
#endif
  return CBLASMPIKernel::dot(vx.distribution(), vx, vy);
}

void SimpleCSRInternalLinearAlgebra::dot(const CSRVector& vx,
                                         const CSRVector& vy,
                                         SimpleCSRInternalLinearAlgebra::FutureType& res) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-DOT-F");
#endif
  res() = CBLASMPIKernel::dot(vx.distribution(), vx, vy);
}

/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebra::scal(Real alpha, CSRVector& vx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-SCAL");
#endif
  CBLASMPIKernel::scal(vx.distribution(), alpha, vx);
}

void SimpleCSRInternalLinearAlgebra::diagonal(
const CSRMatrix& a ALIEN_UNUSED_PARAM, CSRVector& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

void SimpleCSRInternalLinearAlgebra::reciprocal(CSRVector& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

void SimpleCSRInternalLinearAlgebra::pointwiseMult(const CSRVector& vx,
                                                   const CSRVector& vy,
                                                   CSRVector& vz) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-XYZ");
#endif
  CBLASMPIKernel::pointwiseMult(vx.distribution(), vx, vy, vz);
}

void SimpleCSRInternalLinearAlgebra::assign(CSRVector& vx, Real alpha) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "CSR-ASSIGN");
#endif
  CBLASMPIKernel::assign(vx.distribution(), alpha, vx);
}

Integer SimpleCSRInternalLinearAlgebra::computeCxr(const CSRMatrix& a, CSRMatrix& cxr_a) const
{
  auto block_size = a.blockSize() ;
  cxr_a.setBlockSize(1) ;
  cxr_a.copy(a) ;
  return block_size ;
}


SimpleCSRInternalLinearAlgebraExpr::SimpleCSRInternalLinearAlgebraExpr()
: IInternalLinearAlgebraExpr<CSRMatrix, CSRVector>()
{}

/*---------------------------------------------------------------------------*/

SimpleCSRInternalLinearAlgebraExpr::~SimpleCSRInternalLinearAlgebraExpr() {}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebraExpr::norm0(const CSRVector& vx) const
{
  return CBLASMPIKernel::nrm0(vx.distribution(), vx);
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebraExpr::norm1(const CSRVector& vx) const
{
  return CBLASMPIKernel::nrm1(vx.distribution(), vx);
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebraExpr::norm2(const CSRVector& vx) const
{
  return CBLASMPIKernel::nrm2(vx.distribution(), vx);
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebraExpr::normInf(const CSRVector& vx) const
{
  return CBLASMPIKernel::nrmInf(vx.distribution(), vx);
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebraExpr::norm2(const CSRMatrix& mx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "MATRIX-CSR-NORM2");
#endif
  return CBLASMPIKernel::matrix_nrm2(mx.distribution(), mx);
}
/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebraExpr::mult(
const CSRMatrix& ma, const CSRVector& vx, CSRVector& vr) const
{
  Internal::SimpleCSRMatrixMultT<Real>(ma).mult(vx, vr);
}

void SimpleCSRInternalLinearAlgebraExpr::mult(
const CSRMatrix& ma, const UniqueArray<Real>& vx, UniqueArray<Real>& vr) const
{
  Internal::SimpleCSRMatrixMultT<Real>(ma).mult(vx, vr);
}

/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebraExpr::axpy(Real alpha, const UniqueArray<Real>& vx, UniqueArray<Real>& vr) const
{
  cblas::axpy(vx.size(), alpha, dataPtr(vx), 1, dataPtr(vr), 1);
}

void SimpleCSRInternalLinearAlgebraExpr::axpy(Real alpha, const CSRVector& vx, CSRVector& vr) const
{
  CBLASMPIKernel::axpy(vx.distribution(), alpha, vx, vr);
}

/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebraExpr::aypx(Real alpha, UniqueArray<Real>& vy, UniqueArray<Real> const& vx) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

void SimpleCSRInternalLinearAlgebraExpr::aypx(Real alpha ALIEN_UNUSED_PARAM,
                                              CSRVector& y ALIEN_UNUSED_PARAM, const CSRVector& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void SimpleCSRInternalLinearAlgebraExpr::copy(
const UniqueArray<Real>& vx, UniqueArray<Real>& vr) const
{
  cblas::copy(vx.size(), dataPtr(vx), 1, dataPtr(vr), 1);
}

void SimpleCSRInternalLinearAlgebraExpr::copy(const CSRVector& vx, CSRVector& vr) const
{
  CBLASMPIKernel::copy(vx.distribution(), vx, vr);
}

void SimpleCSRInternalLinearAlgebraExpr::copy(const CSRMatrix& ma, CSRMatrix& mr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "MATRIX-CSR-COPY");
#endif
  mr.copy(ma);
}

void SimpleCSRInternalLinearAlgebraExpr::add(const CSRMatrix& ma, CSRMatrix& mr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "MATRIX-CSR-ADD");
#endif

  cblas::axpy(mr.getProfile().getNnz(), 1, (CSRMatrix::ValueType*)ma.data(), 1, mr.data(), 1);
}

void SimpleCSRInternalLinearAlgebraExpr::scal(Real alpha, CSRMatrix& mr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "MATRIX-CSR-SCAL");
#endif

  cblas::scal(mr.getProfile().getNnz(), alpha, mr.data(), 1);
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebraExpr::dot(
Integer local_size, const UniqueArray<Real>& vx, const UniqueArray<Real>& vy) const
{
  return cblas::dot(local_size, dataPtr(vx), 1, dataPtr(vy), 1);
}

Real SimpleCSRInternalLinearAlgebraExpr::dot(const CSRVector& vx, const CSRVector& vy) const
{
  return CBLASMPIKernel::dot(vx.distribution(), vx, vy);
}

/*---------------------------------------------------------------------------*/
void SimpleCSRInternalLinearAlgebraExpr::scal(Real alpha ALIEN_UNUSED_PARAM, UniqueArray<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::scal not implemented");
}

void SimpleCSRInternalLinearAlgebraExpr::scal(Real alpha ALIEN_UNUSED_PARAM, CSRVector& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

void SimpleCSRInternalLinearAlgebraExpr::diagonal(
const CSRMatrix& a ALIEN_UNUSED_PARAM, CSRVector& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

void SimpleCSRInternalLinearAlgebraExpr::reciprocal(CSRVector& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

void SimpleCSRInternalLinearAlgebraExpr::pointwiseMult(const CSRVector& x ALIEN_UNUSED_PARAM,
                                                       const CSRVector& y ALIEN_UNUSED_PARAM, CSRVector& w ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::aypx not implemented");
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
