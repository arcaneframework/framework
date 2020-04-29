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
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

typedef AlgebraTraits<BackEnd::tag::simplecsr>::matrix_type CSRMatrix;
typedef AlgebraTraits<BackEnd::tag::simplecsr>::vector_type CSRVector;

class SimpleCSRInternalLinearAlgebra : public IInternalLinearAlgebra<CSRMatrix, CSRVector>
{
 public:
  SimpleCSRInternalLinearAlgebra();
  virtual ~SimpleCSRInternalLinearAlgebra();

 public:
  // IInternalLinearAlgebra interface.
  Real norm0(const Vector& x) const;
  Real norm1(const Vector& x) const;
  Real norm2(const Vector& x) const;
  void mult(const Matrix& a, const Vector& x, Vector& r) const;
  void axpy(const Real& alpha, const Vector& x, Vector& r) const;
  void aypx(const Real& alpha, Vector& y, const Vector& x) const;
  void copy(const Vector& x, Vector& r) const;
  Real dot(const Vector& x, const Vector& y) const;
  void scal(const Real& alpha, Vector& x) const;
  void diagonal(const Matrix& a, Vector& x) const;
  void reciprocal(Vector& x) const;
  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const;

 private:
  // No member.
};

class SimpleCSRInternalLinearAlgebraExpr
: public IInternalLinearAlgebraExpr<CSRMatrix, CSRVector>
{
 public:
  SimpleCSRInternalLinearAlgebraExpr();
  virtual ~SimpleCSRInternalLinearAlgebraExpr();

 public:
  // IInternalLinearAlgebra interface.
  Real norm0(const Vector& x) const;
  Real norm1(const Vector& x) const;
  Real norm2(const Vector& x) const;
  void mult(const Matrix& a, const Vector& x, Vector& r) const;
  void axpy(const Real& alpha, const Vector& x, Vector& r) const;
  void aypx(const Real& alpha, Vector& y, const Vector& x) const;
  void copy(const Vector& x, Vector& r) const;
  Real dot(const Vector& x, const Vector& y) const;
  void scal(const Real& alpha, Vector& x) const;
  void diagonal(const Matrix& a, Vector& x) const;
  void reciprocal(Vector& x) const;
  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const;

  // IInternalLinearAlgebra interface.

  void mult(const Matrix& a, const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  void axpy(const Real& alpha, UniqueArray<Real> const& x, UniqueArray<Real>& r) const;
  void aypx(const Real& alpha, UniqueArray<Real>& y, UniqueArray<Real> const& x) const;
  void copy(const UniqueArray<Real>& x, UniqueArray<Real>& r) const;
  Real dot(
      Integer local_size, const UniqueArray<Real>& x, const UniqueArray<Real>& y) const;

  void scal(const Real& alpha, UniqueArray<Real>& x) const;

 private:
  // No member.
};

} // namespace Alien

/*---------------------------------------------------------------------------*/
