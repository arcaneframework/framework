// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <tuple>

#include <alien/utils/Precomp.h>

#include <alien/core/backend/IInternalLinearAlgebraExprT.h>
#include <alien/core/backend/IInternalLinearAlgebraT.h>
#include <alien/kernels/sycl/SYCLBackEnd.h>

#include <alien/utils/ExceptionUtils.h>

#include <alien/utils/StdTimer.h>
/*---------------------------------------------------------------------------*/

namespace Alien
{
namespace SYCLInternal
{

  template <typename T>
  class Future;

  class KernelInternal;
} // namespace SYCLInternal

typedef AlgebraTraits<BackEnd::tag::sycl>::matrix_type SYCLMatrixType;
typedef AlgebraTraits<BackEnd::tag::sycl>::vector_type SYCLVectorType;

class ALIEN_EXPORT SYCLInternalLinearAlgebra
: public IInternalLinearAlgebra<SYCLMatrixType, SYCLVectorType>
{
 public:
  typedef BackEnd::tag::sycl BackEndType;

  //typedef VectorDistribution ResourceType;
  typedef std::tuple<VectorDistribution const*,Integer> ResourceType;

  class NullValueException
  : public Exception::NumericException
  {
   public:
    typedef Exception::NumericException BaseType;
    NullValueException(std::string const& type)
    : BaseType(type, __LINE__)
    {}
  };

  // clang-format off
  typedef SYCLInternal::Future<Real> FutureType ;

  typedef Alien::StdTimer            TimerType ;
  typedef TimerType::Sentry          SentryType ;
  // clang-format on

  SYCLInternalLinearAlgebra();
  virtual ~SYCLInternalLinearAlgebra();

  void setDotAlgo(int dot_algo);

 public:
  // IInternalLinearAlgebra interface.
  Real norm0(const Vector& x) const;
  Real norm1(const Vector& x) const;
  Real norm2(const Vector& x) const;
  Real normInf(const Vector& x) const;

  void mult(const Matrix& a, const Vector& x, Vector& r) const;
  void addLMult(Real alpha, const Matrix& A, const Vector& x, Vector& y) const;
  void addUMult(Real alpha, const Matrix& A, const Vector& x, Vector& y) const;

  void multDiag(const Matrix& A, Vector const& y, Vector& z) const;
  void multDiag(const Vector& diag, Vector const& y, Vector& z) const;

  void multInvDiag(const Matrix& A, Vector& y) const;
  void computeInvDiag(const Matrix& a, Vector& inv_diag) const;

  void axpy(Real alpha, const Vector& x, Vector& r) const;
  void aypx(Real alpha, Vector& y, const Vector& x) const;
  void copy(const Vector& x, Vector& r) const;

  void axpy(Real alpha, const Vector& x, Integer stride_x, Vector& r, Integer stride_r) const;
  void aypx(Real alpha, Vector& y, Integer stride_y, const Vector& x, Integer stride_x) const;
  void copy(const Vector& x, Integer stride_x, Vector& r, Integer stride_r) const;

  Real dot(const Vector& x, const Vector& y) const;
  void dot(const Vector& x, const Vector& y, SYCLInternal::Future<Real>& res) const;

  void scal(Real alpha, Vector& x) const;
  void scal(const Vector& x, Matrix& a) const;
  void diagonal(const Matrix& a, Vector& x) const;
  void reciprocal(Vector& x) const;
  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const;

  void assign(Vector& x, Real alpha) const;

  template <typename LambdaT>
  void assign(Vector& x, LambdaT const& lambda) const
  {
    x.apply(lambda);
    //m_internal->apply(lambda,x) ;
  }

  template <typename PrecondT>
  void exec(PrecondT& precond, Vector const& x, Vector& y)
  {
    return precond.solve(*this, x, y);
  }

  Integer computeCxr(const Matrix& a, Matrix& cxr_a) const ;
  Integer computeCxr(const Matrix& a, Vector const& diag_scal, Matrix& cxr_a) const ;

  static ResourceType resource(Matrix const& A);

  void allocate(ResourceType resource, Vector& v);

  template <typename T0, typename... T>
  void allocate(ResourceType resource, T0& v0, T&... args)
  {
    allocate(resource, v0);
    allocate(resource, args...);
  }

  void free(Vector& v);

  template <typename T0, typename... T>
  void free(T0& v0, T&... args)
  {
    free(v0);
    free(args...);
  }

 private:
  std::unique_ptr<SYCLInternal::KernelInternal> m_internal;
#ifdef ALIEN_USE_PERF_TIMER
  mutable TimerType m_timer;
#endif
};

class SYCLInternalLinearAlgebraExpr
: public IInternalLinearAlgebraExpr<SYCLMatrixType, SYCLVectorType>
{
 public:
  SYCLInternalLinearAlgebraExpr();
  virtual ~SYCLInternalLinearAlgebraExpr();

 public:
  // IInternalLinearAlgebra interface.
  Real norm0(const Vector& x) const;
  Real norm1(const Vector& x) const;
  Real norm2(const Vector& x) const;
  Real normInf(const Vector& x) const;
  void mult(const Matrix& a, const Vector& x, Vector& r) const;
  void axpy(Real alpha, const Vector& x, Vector& r) const;
  void aypx(Real alpha, Vector& y, const Vector& x) const;
  void copy(const Vector& x, Vector& r) const;
  Real dot(const Vector& x, const Vector& y) const;
  void scal(Real alpha, Vector& x) const;
  void diagonal(const Matrix& a, Vector& x) const;
  void reciprocal(Vector& x) const;
  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const;

  Real norm2(const Matrix& x) const;
  void copy(const Matrix& a, Matrix& r) const;
  void add(const Matrix& a, Matrix& r) const;
  void scal(Real alpha, Matrix& a) const;

  // IInternalLinearAlgebra interface.

  void mult(const Matrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  void axpy(Real alpha, UniqueArray<Real> const& x, UniqueArray<Real>& r) const;
  void aypx(Real alpha, UniqueArray<Real>& y, UniqueArray<Real> const& x) const;
  void copy(const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  Real dot(Integer local_size, const UniqueArray<Real>& x, const UniqueArray<Real>& y) const;

  void scal(Real alpha, UniqueArray<Real>& x) const;


 private:
  std::unique_ptr<SYCLInternal::KernelInternal> m_internal;
};

} // namespace Alien

/*---------------------------------------------------------------------------*/
