// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/Request.h"
#include "arccore/message_passing/IStat.h"
#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
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
SYCLInternalLinearAlgebra::ResourceType
SYCLInternalLinearAlgebra::resource(Matrix const& A)
{
  return std::make_tuple(&A.distribution().rowDistribution(),A.blockSize());
}

void SYCLInternalLinearAlgebra::allocate(ResourceType resource, Vector& v)
{
  v.init(*std::get<0>(resource),std::get<1>(resource), true);
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

Real SYCLInternalLinearAlgebra::normInf([[maybe_unused]] const SYCLVector<Real>& vx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-NORMINF");
#endif
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::normInf not implemented");
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


void SYCLInternalLinearAlgebra::axpy(Real alpha,
                                     const SYCLVector<Real>& vx,
                                     Integer stride_x,
                                     SYCLVector<Real>& vr,
                                     Integer stride_r) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-AXPY");
#endif
  m_internal->axpy(alpha, vx.internal()->values(), stride_x, vr.internal()->values(), stride_r);
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebra::aypx(Real alpha ALIEN_UNUSED_PARAM,
                                     SYCLVector<Real>& y ALIEN_UNUSED_PARAM,
                                     Integer stride_y ALIEN_UNUSED_PARAM,
                                     const SYCLVector<Real>& x ALIEN_UNUSED_PARAM,
                                     Integer stride_x ALIEN_UNUSED_PARAM) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::aypx not implemented");
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebra::copy(const SYCLVector<Real>& vx,
                                     Integer stride_x,
                                     SYCLVector<Real>& vr,
                                     Integer stride_r) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-COPY");
#endif
  m_internal->copy(vx.internal()->values(), stride_x, vr.internal()->values(), stride_r);
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
    Real local_value = res() ;
    Real* x = &res();
    auto request = mpNonBlockingAllReduce(dist.parallelMng(),
                                          Arccore::MessagePassing::ReduceSum,
                                          Arccore::ConstArrayView<Real>(1,&local_value),
                                          Arccore::ArrayView<Real>(1,x)) ;
    res.addRequest(dist.parallelMng(), request);
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


void SYCLInternalLinearAlgebra::scal(SYCLVector<Real> const& vx, SYCLBEllPackMatrix<Real>& ma) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-SCAL");
#endif
  return ma.scal(vx);
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


Integer SYCLInternalLinearAlgebra::computeCxr(const Matrix& a, Matrix& cxr_a) const
{
  auto block_size = a.blockSize() ;
  cxr_a.setBlockSize(1) ;
  cxr_a.copy(a) ;
  return block_size ;
}

Integer SYCLInternalLinearAlgebra::computeCxr(const Matrix& a, const Vector& diag, Matrix& cxr_a) const
{
  auto block_size = a.blockSize() ;
  cxr_a.setBlockSize(1) ;
  cxr_a.copy(a) ;
  cxr_a.scal(diag);
  return block_size ;
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

Real SYCLInternalLinearAlgebraExpr::normInf([[maybe_unused]] const SYCLVector<Real>& vx) const
{
#ifdef ALIEN_USE_PERF_TIMER
  SentryType s(m_timer, "SYCL-NORMINF");
#endif
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::normInf not implemented");
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

void SYCLInternalLinearAlgebraExpr::axpy([[maybe_unused]] Real alpha,
                                         [[maybe_unused]] const UniqueArray<Real>& vx,
                                         [[maybe_unused]] UniqueArray<Real>& vr) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::axpy not implemented");
}

void SYCLInternalLinearAlgebraExpr::axpy(Real alpha, const SYCLVector<Real>& vx, SYCLVector<Real>& vy) const
{
  m_internal->axpy(alpha, vx.internal()->values(), vy.internal()->values());
}

/*---------------------------------------------------------------------------*/

void SYCLInternalLinearAlgebraExpr::aypx([[maybe_unused]] Real alpha, [[maybe_unused]] UniqueArray<Real>& vy, [[maybe_unused]] UniqueArray<Real> const& vx) const
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

void SYCLInternalLinearAlgebraExpr::copy([[maybe_unused]] const UniqueArray<Real>& vx,
                                         [[maybe_unused]] UniqueArray<Real>& vr) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::copy not implemented");
}

void SYCLInternalLinearAlgebraExpr::copy(const SYCLVector<Real>& vx, SYCLVector<Real>& vy) const
{
  m_internal->copy(vx.internal()->values(), vy.internal()->values());
}

/*---------------------------------------------------------------------------*/

Real SYCLInternalLinearAlgebraExpr::dot([[maybe_unused]]
Integer local_size, [[maybe_unused]] const UniqueArray<Real>& vx, [[maybe_unused]] const UniqueArray<Real>& vy) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::dot not implemented");
}

Real SYCLInternalLinearAlgebraExpr::dot(const SYCLVector<Real>& vx, const SYCLVector<Real>& vy) const
{
  return m_internal->dot(vx.internal()->values(), vy.internal()->values());
}

Real SYCLInternalLinearAlgebraExpr::norm2([[maybe_unused]] const SYCLBEllPackMatrix<Real>& a) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::notm2 not implemented");
}

void SYCLInternalLinearAlgebraExpr::copy([[maybe_unused]] const SYCLBEllPackMatrix<Real>& a, [[maybe_unused]] SYCLBEllPackMatrix<Real>& r) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::copy not implemented");
}

void SYCLInternalLinearAlgebraExpr::add([[maybe_unused]] const SYCLBEllPackMatrix<Real>& a, [[maybe_unused]] SYCLBEllPackMatrix<Real>& r) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::add not implemented");
}

void SYCLInternalLinearAlgebraExpr::scal([[maybe_unused]] Real alpha, [[maybe_unused]] SYCLBEllPackMatrix<Real>& a) const
{
  throw NotImplementedException(
  A_FUNCINFO, "SYCLLinearAlgebra::scal not implemented");
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

namespace SYCLInternal
{
template <>
sycl::buffer<double>& KernelInternal::getWorkBuffer(std::size_t size)
{
  if (m_double_work == nullptr) {
    m_double_work = new sycl::buffer<double>(size);
    m_double_work->set_final_data(nullptr);
  }
  else {
    if (size > m_double_work->size()) {
      delete m_double_work;
      m_double_work = new sycl::buffer<double>(size);
      m_double_work->set_final_data(nullptr);
    }
  }
  return *m_double_work;
}
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
