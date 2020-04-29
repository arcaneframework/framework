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
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

class DoKLinearSolver;
class DoKMatrix;
class VectorDoK;
class Space;
template <class Matrix, class Vector> class IInternalLinearAlgebra;
template <class Matrix, class Vector> class IInternalLinearSolver;

extern IInternalLinearAlgebra<DoKMatrix, VectorDoK>* DoKLinearAlgebraFactory();

extern IInternalLinearSolver<DoKMatrix, VectorDoK>* DoKLinearSolverFactory(
    IMessagePassingMng* p_mng);

/*---------------------------------------------------------------------------*/

namespace BackEnd {
  namespace tag {
    struct DoK
    {};
  } // namespace tag
} // namespace BackEnd

template <> struct AlgebraTraits<BackEnd::tag::DoK>
{
  typedef DoKMatrix matrix_type;
  typedef VectorDoK vector_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  typedef IInternalLinearSolver<matrix_type, vector_type> solver_type;
  static algebra_type* algebra_factory() { return DoKLinearAlgebraFactory(); }
  static solver_type* solver_factory(IMessagePassingMng* p_mng)
  {
    return DoKLinearSolverFactory(p_mng);
  }
  static BackEndId name() { return "DoK"; }
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
