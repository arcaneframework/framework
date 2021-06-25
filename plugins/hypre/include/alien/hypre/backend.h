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

namespace Alien::Hypre
{
class Matrix;

class Vector;

class Options;

extern IInternalLinearSolver<Matrix, Vector>* InternalLinearSolverFactory(const Options& options);

extern IInternalLinearSolver<Matrix, Vector>* InternalLinearSolverFactory();

extern IInternalLinearAlgebra<Matrix, Vector>* InternalLinearAlgebraFactory();
} // namespace Alien::Hypre

namespace Alien
{
namespace BackEnd
{
  namespace tag
  {
    struct hypre
    {
    };
  } // namespace tag
} // namespace BackEnd

template <>
struct AlgebraTraits<BackEnd::tag::hypre>
{
  // types
  using matrix_type = Hypre::Matrix;
  using vector_type = Hypre::Vector;
  using options_type = Hypre::Options;
  using algebra_type = IInternalLinearAlgebra<matrix_type, vector_type>;
  using solver_type = IInternalLinearSolver<matrix_type, vector_type>;

  // factory to build algebra
  static auto* algebra_factory()
  {
    return Hypre::InternalLinearAlgebraFactory();
  }

  // factories to build solver
  static auto* solver_factory(const options_type& options)
  {
    return Hypre::InternalLinearSolverFactory(options);
  }

  // factories to build default solver
  static auto* solver_factory()
  {
    return Hypre::InternalLinearSolverFactory();
  }

  static BackEndId name() { return "hypre"; }
};

} // namespace Alien

// user interface
namespace Alien::Hypre
{
using LinearSolver = Alien::LinearSolver<BackEnd::tag::hypre>;
using LinearAlgebra = Alien::LinearAlgebra<BackEnd::tag::hypre>;
} // namespace Alien::Hypre
