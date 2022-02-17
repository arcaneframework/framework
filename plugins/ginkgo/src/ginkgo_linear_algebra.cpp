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

#include "matrix.h"
#include "vector.h"

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
#include <alien/core/backend/LinearAlgebraT.h>

#include <alien/ginkgo/backend.h>
#include <alien/ginkgo/export.h>

#include <ginkgo/ginkgo.hpp>

namespace Alien
{
template class ALIEN_GINKGO_EXPORT LinearAlgebra<BackEnd::tag::ginkgo>;
} // namespace Alien

namespace Alien::Ginkgo
{
class ALIEN_GINKGO_EXPORT InternalLinearAlgebra
: public IInternalLinearAlgebra<Matrix, Vector>
{
 public:
  InternalLinearAlgebra() = default;

  ~InternalLinearAlgebra() override = default;

 public:
  // IInternalLinearAlgebra interface.
  Arccore::Real norm0(const Vector& x) const override;

  Arccore::Real norm1(const Vector& x) const override;

  Arccore::Real norm2(const Vector& x) const override;

  void mult(const Matrix& a, const Vector& x, Vector& r) const override;

  void axpy(Arccore::Real alpha, const Vector& x,
            Vector& r) const override;

  void aypx(Arccore::Real alpha, Vector& y,
            const Vector& x) const override;

  void copy(const Vector& x, Vector& r) const override;

  Arccore::Real dot(const Vector& x, const Vector& y) const override;

  void scal(Arccore::Real alpha, Vector& x) const override;

  void diagonal(const Matrix& a, Vector& x) const override;

  void reciprocal(Vector& x) const override;

  void pointwiseMult(const Vector& x, const Vector& y,
                     Vector& w) const override;
};

Arccore::Real
InternalLinearAlgebra::norm0(const Vector& vx ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(A_FUNCINFO, "GinkgoLinearAlgebra::norm0 not implemented");
}

Arccore::Real
InternalLinearAlgebra::norm1(const Vector& vx) const
{
  throw Arccore::NotImplementedException(A_FUNCINFO, "GinkgoLinearAlgebra::norm1 not implemented");
}

Arccore::Real
InternalLinearAlgebra::norm2(const Vector& vx) const
{
  using vec = gko::matrix::Dense<double>;
  auto exec = vx.internal()->get_executor();
  auto mtx_res = gko::initialize<vec>({ 1.0 }, exec);
  vx.internal()->compute_norm2(gko::lend(mtx_res.get()));
  return mtx_res->get_values()[0];
}

void InternalLinearAlgebra::mult(const Matrix& ma, const Vector& vx, Vector& vr) const
{
  ma.internal()->apply(lend(vx.internal()), lend(vr.internal()));
}

void InternalLinearAlgebra::axpy(
Arccore::Real alpha, const Vector& vx, Vector& vr) const
{
  using vec = gko::matrix::Dense<double>;
  auto exec = vx.internal()->get_executor();
  auto mtx_alpha = gko::initialize<vec>({ alpha }, exec);
  vr.internal()->add_scaled(mtx_alpha.get(), vx.internal());
}

void InternalLinearAlgebra::copy(const Vector& vx, Vector& vr) const
{
  vr.internal()->copy_from(vx.internal());
}

Arccore::Real
InternalLinearAlgebra::dot(const Vector& vx, const Vector& vy) const
{
  using vec = gko::matrix::Dense<double>;
  auto exec = vx.internal()->get_executor();
  auto res = gko::initialize<vec>({ 1.0 }, exec);

  vx.internal()->compute_dot(lend(vy.internal()), lend(res));
  return res->get_values()[0];
}

void InternalLinearAlgebra::diagonal(Matrix const& m ALIEN_UNUSED_PARAM, Vector& v ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(A_FUNCINFO, "GinkgoLinearAlgebra::diagonal not implemented");
}

void InternalLinearAlgebra::reciprocal(Vector& v ALIEN_UNUSED_PARAM) const
{
  throw Arccore::NotImplementedException(A_FUNCINFO, "GinkgoLinearAlgebra::reciprocal not implemented");
}

void InternalLinearAlgebra::aypx(
Arccore::Real alpha, Vector& y, const Vector& x) const
{
  throw Arccore::NotImplementedException(A_FUNCINFO, "GinkgoLinearAlgebra::aypx not implemented");
}

void InternalLinearAlgebra::pointwiseMult(
const Vector& x, const Vector& y, Vector& w) const
{
  throw Arccore::NotImplementedException(A_FUNCINFO, "GinkgoLinearAlgebra::pointwiseMult not implemented");
}

void InternalLinearAlgebra::scal(Arccore::Real alpha, Vector& x) const
{
  using vec = gko::matrix::Dense<double>;
  auto exec = x.internal()->get_executor();
  auto a = gko::initialize<vec>({ alpha }, exec);
  x.internal()->scale(a.get());
}

ALIEN_GINKGO_EXPORT
IInternalLinearAlgebra<Ginkgo::Matrix, Ginkgo::Vector>*
InternalLinearAlgebraFactory()
{
  return new Ginkgo::InternalLinearAlgebra();
}
} // namespace Alien::Ginkgo
