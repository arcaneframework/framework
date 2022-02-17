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

#include <alien/core/backend/BackEnd.h>
#include <alien/core/backend/LinearSolver.h>
#include <alien/core/backend/LinearAlgebra.h>

namespace Alien::Ginkgo
{
// Forward declarations
class Matrix;

class Vector;

class Options;

extern IInternalLinearSolver<Matrix, Vector>* InternalLinearSolverFactory(const Options& options);

extern IInternalLinearSolver<Matrix, Vector>* InternalLinearSolverFactory();

extern IInternalLinearAlgebra<Matrix, Vector>* InternalLinearAlgebraFactory();
} // namespace Alien::Ginkgo

namespace Alien::BackEnd::tag
{
struct ginkgo
{
};
} // namespace Alien::BackEnd::tag

namespace Alien
{
template <>
struct AlgebraTraits<BackEnd::tag::ginkgo>
{
  // types
  using matrix_type = Ginkgo::Matrix;
  using vector_type = Ginkgo::Vector;
  using options_type = Ginkgo::Options;
  using algebra_type = IInternalLinearAlgebra<matrix_type, vector_type>;
  using solver_type = IInternalLinearSolver<matrix_type, vector_type>;

  // factory to build algebra
  static auto algebra_factory()
  {
    return Ginkgo::InternalLinearAlgebraFactory();
  }

  // factories to build solver
  static auto solver_factory(const options_type& options)
  {
    return Ginkgo::InternalLinearSolverFactory(options);
  }

  // factories to build default solver
  static auto solver_factory() { return Ginkgo::InternalLinearSolverFactory(); }

  static BackEndId name() { return "ginkgo"; }
};

} // namespace Alien

// user interface
namespace Alien::Ginkgo
{
using LinearSolver = Alien::LinearSolver<BackEnd::tag::ginkgo>;
using LinearAlgebra = Alien::LinearAlgebra<BackEnd::tag::ginkgo>;
} // namespace Alien::Ginkgo
