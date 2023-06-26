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

#pragma once

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

  typedef VectorDistribution ResourceType;

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

  void mult(const Matrix& a, const Vector& x, Vector& r) const;
  void addLMult(Real alpha, const Matrix& A, const Vector& x, Vector& y) const;
  void addUMult(Real alpha, const Matrix& A, const Vector& x, Vector& y) const;

  void multInvDiag(const Matrix& A, Vector& y) const;
  void computeInvDiag(const Matrix& a, Vector& inv_diag) const;

  void axpy(Real alpha, const Vector& x, Vector& r) const;
  void aypx(Real alpha, Vector& y, const Vector& x) const;
  void copy(const Vector& x, Vector& r) const;

  Real dot(const Vector& x, const Vector& y) const;
  void dot(const Vector& x, const Vector& y, SYCLInternal::Future<Real>& res) const;

  void scal(Real alpha, Vector& x) const;
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

  static ResourceType const& resource(Matrix const& A);

  void allocate(ResourceType const& resource, Vector& v);

  template <typename T0, typename... T>
  void allocate(ResourceType const& resource, T0& v0, T&... args)
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
  void mult(const Matrix& a, const Vector& x, Vector& r) const;
  void axpy(Real alpha, const Vector& x, Vector& r) const;
  void aypx(Real alpha, Vector& y, const Vector& x) const;
  void copy(const Vector& x, Vector& r) const;
  Real dot(const Vector& x, const Vector& y) const;
  void scal(Real alpha, Vector& x) const;
  void diagonal(const Matrix& a, Vector& x) const;
  void reciprocal(Vector& x) const;
  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const;

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
