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
SimpleCSRInternalLinearAlgebra::ResourceType const&
SimpleCSRInternalLinearAlgebra::resource(Matrix const& A)
{
  return A.distribution().rowDistribution();
}

void SimpleCSRInternalLinearAlgebra::allocate(ResourceType const& resource, Vector& v)
{
  v.init(resource, true);
}

void SimpleCSRInternalLinearAlgebra::free(Vector& v)
{
  v.clear();
}

Real SimpleCSRInternalLinearAlgebra::norm0(const CSRVector& vx ALIEN_UNUSED_PARAM) const
{
  // return CBLASMPIKernel::nrm0(x.space().structInfo(),vx);
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::norm0 not implemented");
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

SimpleCSRInternalLinearAlgebraExpr::SimpleCSRInternalLinearAlgebraExpr()
: IInternalLinearAlgebraExpr<CSRMatrix, CSRVector>()
{}

/*---------------------------------------------------------------------------*/

SimpleCSRInternalLinearAlgebraExpr::~SimpleCSRInternalLinearAlgebraExpr() {}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebraExpr::norm0(const CSRVector& vx ALIEN_UNUSED_PARAM) const
{
  // return CBLASMPIKernel::nrm0(x.space().structInfo(),vx);
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::norm0 not implemented");
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebraExpr::norm1(const CSRVector& vx ALIEN_UNUSED_PARAM) const
{
  // return CBLASMPIKernel::nrm1(x.space().structInfo(),vx);
  throw NotImplementedException(
  A_FUNCINFO, "SimpleCSRLinearAlgebra::norm1 not implemented");
}

/*---------------------------------------------------------------------------*/

Real SimpleCSRInternalLinearAlgebraExpr::norm2(const CSRVector& vx) const
{
  return CBLASMPIKernel::nrm2(vx.distribution(), vx);
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
