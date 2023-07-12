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

#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/Request.h"
#include "arccore/message_passing/IStat.h"
#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing_mpi/MpiAdapter.h"
#include "arccore/message_passing_mpi/MpiLock.h"
#include "arccore/message_passing_mpi/MpiRequest.h"
#include "arccore/message_passing_mpi/MpiTypeDispatcher.h"
#include "arccore/message_passing_mpi/MpiDatatype.h"

#include <alien/utils/Precomp.h>

#include <arccore/base/NotImplementedException.h>
#include <arccore/base/TraceInfo.h>

#include <alien/core/backend/LinearAlgebraExprT.h>
#include <alien/core/backend/LinearAlgebraT.h>

#include "alien/kernels/sycl/data/SYCLEnv.h"
#include "alien/kernels/sycl/data/SYCLEnvInternal.h"

#include "alien/kernels/sycl/data/SYCLVectorInternal.h"
#include "alien/kernels/sycl/data/SYCLVector.h"

#include <alien/kernels/sycl/data/SYCLBEllPackInternal.h>
#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>

#include "alien/kernels/sycl/algebra/SYCLKernelInternal.h"
#include "alien/kernels/sycl/algebra/SYCLInternalLinearAlgebra.h"

#include "alien/kernels/sycl/algebra/SYCLBEllPackMatrixMult.h"
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

template class ALIEN_EXPORT LinearAlgebra<BackEnd::tag::sycl>;
template class ALIEN_EXPORT LinearAlgebraExpr<BackEnd::tag::sycl>;

ALIEN_EXPORT IInternalLinearAlgebra<SYCLBEllPackMatrix<Real>, SYCLVector<Real>>*
SYCLInternalLinearAlgebraFactory()
{
  return new SYCLInternalLinearAlgebra();
}

ALIEN_EXPORT IInternalLinearAlgebraExpr<SYCLBEllPackMatrix<Real>, SYCLVector<Real>>*
SYCLInternalLinearAlgebraExprFactory()
{
  return new SYCLInternalLinearAlgebraExpr();
}

/*---------------------------------------------------------------------------*/

namespace Internal = SYCLInternal;

/*---------------------------------------------------------------------------*/

SYCLInternalLinearAlgebra::SYCLInternalLinearAlgebra()
: IInternalLinearAlgebra<SYCLBEllPackMatrix<Real>, SYCLVector<Real>>()
{
  m_internal.reset(new SYCLInternal::KernelInternal());
}

/*---------------------------------------------------------------------------*/

SYCLInternalLinearAlgebra::~SYCLInternalLinearAlgebra()
{
#ifdef ALIEN_USE_PERF_TIMER
  m_timer.printInfo("SYCL-ALGEBRA");
#endif
}

void SYCLInternalLinearAlgebra::setDotAlgo(int dot_algo)
{
  m_internal->setDotAlgo(dot_algo);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
SYCLInternalLinearAlgebra::ResourceType const&
SYCLInternalLinearAlgebra::resource(Matrix const& A)
{
  return A.distribution().rowDistribution();
}

void SYCLInternalLinearAlgebra::allocate(ResourceType const& resource, Vector& v)
{
  v.init(resource, true);
}

void SYCLInternalLinearAlgebra::free(Vector& v)
{
  v.clear();
}

Real SYCLInternalLinearAlgebra::norm0(const SYCLVector<Real>& vx ALIEN_UNUSED_PARAM) const
{
  // return SYCLBLASKernel::nrm0(x.space().structInfo(),vx);
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::norm0 not implemented");
}

/*---------------------------------------------------------------------------*/

Real SYCLInternalLinearAlgebra::norm1(const SYCLVector<Real>& vx ALIEN_UNUSED_PARAM) const
{
  // return SYCLBLASKernel::nrm1(x.space().structInfo(),vx);
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::norm1 not implemented");
}

/*---------------------------------------------------------------------------*/

Real SYCLInternalLinearAlgebra::norm2(const SYCLVector<Real>& vx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-NORM2");
#endif
  return std::sqrt(m_internal->dot(vx.internal()->values(), vx.internal()->values()));
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebra::mult(const SYCLBEllPackMatrix<Real>& ma,
                                     const SYCLVector<Real>& vx,
                                     SYCLVector<Real>& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-SPMV");
#endif
  Internal::SYCLBEllPackMatrixMultT<Real>(ma).mult(vx, vr);
}

void SYCLInternalLinearAlgebra::addLMult(Real alpha,
                                         const SYCLBEllPackMatrix<Real>& ma,
                                         const SYCLVector<Real>& vx,
                                         SYCLVector<Real>& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-ADDLMULT");
#endif
  Internal::SYCLBEllPackMatrixMultT<Real>(ma).addLMult(alpha, vx, vr);
}

void SYCLInternalLinearAlgebra::addUMult(Real alpha,
                                         const SYCLBEllPackMatrix<Real>& ma,
                                         const SYCLVector<Real>& vx,
                                         SYCLVector<Real>& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-ADDUMULT");
#endif
  Internal::SYCLBEllPackMatrixMultT<Real>(ma).addUMult(alpha, vx, vr);
}

void SYCLInternalLinearAlgebra::
multInvDiag(const SYCLBEllPackMatrix<Real>& ma,
            SYCLVector<Real>& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-MULTINVDIAG");
#endif
  Internal::SYCLBEllPackMatrixMultT<Real>(ma).multInvDiag(vr);
}

void SYCLInternalLinearAlgebra::
computeInvDiag(const SYCLBEllPackMatrix<Real>& ma,
               SYCLVector<Real>& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-INVDIAG");
#endif
  Internal::SYCLBEllPackMatrixMultT<Real>(ma).computeInvDiag(vr);
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebra::axpy(Real alpha, const SYCLVector<Real>& vx, SYCLVector<Real>& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-AXPY");
#endif
  m_internal->axpy(alpha, vx.internal()->values(), vr.internal()->values());
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebra::aypx(Real alpha ALIEN_UNUSED_PARAM,
                                     SYCLVector<Real>& y ALIEN_UNUSED_PARAM,
                                     const SYCLVector<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebra::copy(const SYCLVector<Real>& vx, SYCLVector<Real>& vr) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-COPY");
#endif
  m_internal->copy(vx.internal()->values(), vr.internal()->values());
}

/*---------------------------------------------------------------------------*/

Real SYCLInternalLinearAlgebra::dot(const SYCLVector<Real>& vx, const SYCLVector<Real>& vy) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-DOT");
#endif
  auto value = m_internal->dot(vx.internal()->values(), vy.internal()->values());
  auto& dist = vx.distribution();
  if (dist.isParallel()) {
    return Arccore::MessagePassing::mpAllReduce(dist.parallelMng(),
                                                Arccore::MessagePassing::ReduceSum,
                                                value);
  }
  return value;
}

void SYCLInternalLinearAlgebra::dot(const SYCLVector<Real>& vx,
                                    const SYCLVector<Real>& vy,
                                    SYCLInternal::Future<Real>& res) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-DOT-F");
#endif
  m_internal->dot(vx.internal()->values(), vy.internal()->values(), res.deviceValue());

  auto& dist = vx.distribution();
  if (dist.isParallel()) {
    using namespace Arccore::MessagePassing::Mpi;
    res.get();
    Real* x = &res();
    auto pm = dist.parallelMng();
    auto type_dispatcher = pm->dispatchers()->dispatcher(x);
    MpiTypeDispatcher<Real>* ptr = dynamic_cast<MpiTypeDispatcher<Real>*>(type_dispatcher);
    if (ptr) {
      auto datatype = ptr->datatype();
      auto op = datatype->reduceOperator(Arccore::MessagePassing::ReduceSum);
      auto request = ptr->adapter()->nonBlockingAllReduce(x, x, 1, datatype->datatype(), op);
      res.addRequest(pm, request);
    }
  }
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebra::scal(Real alpha, SYCLVector<Real>& vx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-SCAL");
#endif
  return m_internal->scal(alpha, vx.internal()->values());
}

void SYCLInternalLinearAlgebra::pointwiseMult(const SYCLVector<Real>& vx,
                                              const SYCLVector<Real>& vy,
                                              SYCLVector<Real>& vz) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-XYZ");
#endif
  return m_internal->pointwiseMult(vx.internal()->values(), vy.internal()->values(), vz.internal()->values());
}

void SYCLInternalLinearAlgebra::assign(SYCLVector<Real>& vx, Real alpha) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-ASSIGN");
#endif
  return m_internal->assign(alpha, vx.internal()->values());
}

void SYCLInternalLinearAlgebra::diagonal(
const SYCLBEllPackMatrix<Real>& a ALIEN_UNUSED_PARAM, SYCLVector<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

void SYCLInternalLinearAlgebra::reciprocal(SYCLVector<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

SYCLInternalLinearAlgebraExpr::SYCLInternalLinearAlgebraExpr()
: IInternalLinearAlgebraExpr<SYCLBEllPackMatrix<Real>, SYCLVector<Real>>()
{
  m_internal.reset(new SYCLInternal::KernelInternal());
}

/*---------------------------------------------------------------------------*/

SYCLInternalLinearAlgebraExpr::~SYCLInternalLinearAlgebraExpr() {}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real SYCLInternalLinearAlgebraExpr::norm0(const SYCLVector<Real>& vx ALIEN_UNUSED_PARAM) const
{
  // return SYCLBLASKernel::nrm0(x.space().structInfo(),vx);
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::norm0 not implemented");
}

/*---------------------------------------------------------------------------*/

Real SYCLInternalLinearAlgebraExpr::norm1(const SYCLVector<Real>& vx ALIEN_UNUSED_PARAM) const
{
  // return SYCLBLASKernel::nrm1(x.space().structInfo(),vx);
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::norm1 not implemented");
}

/*---------------------------------------------------------------------------*/

Real SYCLInternalLinearAlgebraExpr::norm2(const SYCLVector<Real>& vx) const
{
  return std::sqrt(m_internal->dot(vx.internal()->values(), vx.internal()->values()));
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebraExpr::mult(
const SYCLBEllPackMatrix<Real>& ma, const SYCLVector<Real>& vx, SYCLVector<Real>& vr) const
{
  Internal::SYCLBEllPackMatrixMultT<Real>(ma).mult(vx, vr);
}

void SYCLInternalLinearAlgebraExpr::mult(
const SYCLBEllPackMatrix<Real>& ma, const UniqueArray<Real>& vx, UniqueArray<Real>& vr) const
{
  Internal::SYCLBEllPackMatrixMultT<Real>(ma).mult(vx, vr);
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebraExpr::axpy(Real alpha, const UniqueArray<Real>& vx, UniqueArray<Real>& vr) const
{
  //cblas::axpy(vx.size(), alpha, dataPtr(vx), 1, dataPtr(vr), 1);
}

void SYCLInternalLinearAlgebraExpr::axpy(Real alpha, const SYCLVector<Real>& vx, SYCLVector<Real>& vy) const
{
  m_internal->axpy(alpha, vx.internal()->values(), vy.internal()->values());
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebraExpr::aypx(Real alpha, UniqueArray<Real>& vy, UniqueArray<Real> const& vx) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

void SYCLInternalLinearAlgebraExpr::aypx(Real alpha ALIEN_UNUSED_PARAM,
                                         SYCLVector<Real>& y ALIEN_UNUSED_PARAM, const SYCLVector<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebraExpr::copy(
const UniqueArray<Real>& vx, UniqueArray<Real>& vr) const
{
  //cblas::copy(vx.size(), dataPtr(vx), 1, dataPtr(vr), 1);
}

void SYCLInternalLinearAlgebraExpr::copy(const SYCLVector<Real>& vx, SYCLVector<Real>& vy) const
{
  m_internal->copy(vx.internal()->values(), vy.internal()->values());
}

/*---------------------------------------------------------------------------*/

Real SYCLInternalLinearAlgebraExpr::dot(
Integer local_size, const UniqueArray<Real>& vx, const UniqueArray<Real>& vy) const
{
  return 0.; //cblas::dot(local_size, dataPtr(vx), 1, dataPtr(vy), 1);
}

Real SYCLInternalLinearAlgebraExpr::dot(const SYCLVector<Real>& vx, const SYCLVector<Real>& vy) const
{
  return m_internal->dot(vx.internal()->values(), vy.internal()->values());
}

/*---------------------------------------------------------------------------*/
void SYCLInternalLinearAlgebraExpr::scal(Real alpha ALIEN_UNUSED_PARAM, UniqueArray<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::scal not implemented");
}

void SYCLInternalLinearAlgebraExpr::scal(Real alpha ALIEN_UNUSED_PARAM, SYCLVector<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

void SYCLInternalLinearAlgebraExpr::diagonal(
const SYCLBEllPackMatrix<Real>& a ALIEN_UNUSED_PARAM, SYCLVector<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

void SYCLInternalLinearAlgebraExpr::reciprocal(SYCLVector<Real>& x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

void SYCLInternalLinearAlgebraExpr::pointwiseMult(const SYCLVector<Real>& x ALIEN_UNUSED_PARAM,
                                                  const SYCLVector<Real>& y ALIEN_UNUSED_PARAM, SYCLVector<Real>& w ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

template class ALIEN_EXPORT SYCLInternal::Future<double>;

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
