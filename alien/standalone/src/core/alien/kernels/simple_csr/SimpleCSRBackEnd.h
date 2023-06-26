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

#include <alien/core/backend/IInternalLinearAlgebraExprT.h>
#include <alien/core/backend/IInternalLinearAlgebraT.h>

namespace Arccore::MessagePassing
{
class IMessagePassingMng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MultiVectorImpl;
template <typename T>
class SimpleCSRMatrix;
template <typename T>
class SimpleCSRVector;

template <typename T>
struct SimpleCSRTraits
{
  typedef SimpleCSRMatrix<T> MatrixType;
  typedef SimpleCSRVector<T> VectorType;
  typedef IInternalLinearAlgebra<MatrixType, VectorType> AlgebraType;
  typedef IInternalLinearAlgebraExpr<MatrixType, VectorType> AlgebraExprType;
};

extern SimpleCSRTraits<Real>::AlgebraType* SimpleCSRInternalLinearAlgebraFactory();
extern SimpleCSRTraits<Real>::AlgebraExprType*
SimpleCSRInternalLinearAlgebraExprFactory();

/*---------------------------------------------------------------------------*/

namespace BackEnd
{
  namespace tag
  {
    struct simplecsr
    {};
  } // namespace tag
} // namespace BackEnd

template <>
struct AlgebraTraits<BackEnd::tag::simplecsr>
{
  // clang-format off
  typedef Real                                   value_type;
  typedef SimpleCSRTraits<Real>::MatrixType      matrix_type;
  typedef SimpleCSRTraits<Real>::VectorType      vector_type;
  typedef SimpleCSRTraits<Real>::AlgebraType     algebra_type;
  typedef SimpleCSRTraits<Real>::AlgebraExprType algebra_expr_type;
  // clang-format off
  
  static algebra_type* algebra_factory(
  IMessagePassingMng* p_mng ALIEN_UNUSED_PARAM = nullptr)
  {
    return SimpleCSRInternalLinearAlgebraFactory();
  }
  static algebra_expr_type* algebra_expr_factory(
  IMessagePassingMng* p_mng ALIEN_UNUSED_PARAM = nullptr)
  {
    return SimpleCSRInternalLinearAlgebraExprFactory();
  }

  static BackEndId name() { return "simplecsr"; }
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
