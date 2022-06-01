/*
 * Copyright 2022 IFPEN-CEA
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

#include "trilinos_matrix.h"
#include "trilinos_vector.h"

#include <arccore/base/NotImplementedException.h>
#include <arccore/base/TraceInfo.h>

#include <alien/core/backend/LinearAlgebraT.h>

#include <alien/trilinos/backend.h>
#include <alien/trilinos/export.h>

namespace Alien
{
template class ALIEN_TRILINOS_EXPORT LinearAlgebra<BackEnd::tag::trilinos>;
} // namespace Alien

namespace Alien::Trilinos
{
class ALIEN_TRILINOS_EXPORT InternalLinearAlgebra
: public IInternalLinearAlgebra<Matrix, Vector>
{

 public:
  InternalLinearAlgebra() = default;

  ~InternalLinearAlgebra() final = default;

  // IInternalLinearAlgebra interface.
  Arccore::Real norm0(const Vector& x) const final;

  Arccore::Real norm1(const Vector& x) const final;

  Arccore::Real norm2(const Vector& x) const final;

  void mult(const Matrix& a, const Vector& x, Vector& r) const final;

  void axpy(Arccore::Real alpha, const Vector& x, Vector& r) const final;

  void aypx(Arccore::Real alpha, Vector& y, const Vector& x) const final;

  void copy(const Vector& x, Vector& r) const final;

  Arccore::Real dot(const Vector& x, const Vector& y) const final;

  void scal(Arccore::Real alpha, Vector& x) const final;

  void diagonal(const Matrix& a, Vector& x) const final;

  void reciprocal(Vector& x) const final;

  void pointwiseMult(const Vector& x, const Vector& y, Vector& w) const final;
};

Arccore::Real
InternalLinearAlgebra::norm0(const Vector& vx ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(A_FUNCINFO, "TrilinosLinearAlgebra::norm0 not implemented");
}

Arccore::Real
InternalLinearAlgebra::norm1(const Vector& vx ALIEN_UNUSED_PARAM) const
{
  Teuchos::Array<Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
  vx.internal()->norm1(norms);
  return norms.at(0);
}

Arccore::Real
InternalLinearAlgebra::norm2(const Vector& vx) const
{
  Teuchos::Array<Teuchos::ScalarTraits<SC>::magnitudeType> norms(1);
  vx.internal()->norm2(norms);
  return norms.at(0);
}

void InternalLinearAlgebra::mult(const Matrix& ma, const Vector& vx, Vector& vr) const
{
  ma.internal()->apply(*(vx.internal()), *(vr.internal()), Teuchos::NO_TRANS, 1, 0);
}

void InternalLinearAlgebra::axpy(
Arccore::Real alpha, const Vector& vx, Vector& vr) const
{
  // y = y + ax , vr = vr + alpha * vx
  vr.internal()->update(alpha, *(vx.internal()), 1);
}

void InternalLinearAlgebra::copy(const Vector& vx /*src*/, Vector& vr /*dest*/) const
{
  Tpetra::deep_copy(*(vr.internal()), *(vx.internal()));
}

Arccore::Real
InternalLinearAlgebra::dot(const Vector& vx, const Vector& vy) const
{
  std::vector<SC> results(1); // dot product result
  vx.internal()->dot(*(vx.internal()), results);
}

void InternalLinearAlgebra::diagonal(Matrix const& m ALIEN_UNUSED_PARAM, Vector& v ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(
  A_FUNCINFO, "TrilinosLinearAlgebra::diagonal not implemented");
}

void InternalLinearAlgebra::reciprocal(Vector& v ALIEN_UNUSED_PARAM) const
{
  //element-wise reciprocal values : this(i,j) = 1/A(i,j).
  v.internal()->reciprocal(*(v.internal()));
}

void InternalLinearAlgebra::aypx(
Arccore::Real alpha ALIEN_UNUSED_PARAM, Vector& y ALIEN_UNUSED_PARAM, const Vector& x ALIEN_UNUSED_PARAM) const
{
  y.internal()->update(1, *(x.internal()), alpha); // y = alpha * y
}

void InternalLinearAlgebra::pointwiseMult(
const Vector& x ALIEN_UNUSED_PARAM, const Vector& y ALIEN_UNUSED_PARAM, Vector& w ALIEN_UNUSED_PARAM) const
{
  w.internal()->elementWiseMultiply(1, *(y.internal()->getVector(0)), *(x.internal()), 0);
}

void InternalLinearAlgebra::scal(Arccore::Real alpha, Vector& x) const
{
  x.internal()->scale(alpha);
}

ALIEN_TRILINOS_EXPORT
IInternalLinearAlgebra<Trilinos::Matrix, Trilinos::Vector>*
InternalLinearAlgebraFactory()
{
  return new Trilinos::InternalLinearAlgebra();
}
} // namespace Alien::Trilinos
