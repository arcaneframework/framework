// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <arccore/message_passing/MessagePassingGlobal.h>
#include <alien/core/backend/BackEnd.h>
#include <alien/utils/Precomp.h>

#include <alien/core/backend/IInternalLinearAlgebraExprT.h>
#include <alien/core/backend/IInternalLinearAlgebraT.h>

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

extern ALIEN_EXPORT SimpleCSRTraits<Real>::AlgebraType* SimpleCSRInternalLinearAlgebraFactory();
extern ALIEN_EXPORT SimpleCSRTraits<Real>::AlgebraExprType*
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
  
  static algebra_type* algebra_factory([[maybe_unused]] IMessagePassingMng* p_mng = nullptr)
  {
    return SimpleCSRInternalLinearAlgebraFactory();
  }
  static algebra_expr_type* algebra_expr_factory([[maybe_unused]] IMessagePassingMng* p_mng = nullptr)
  {
    return SimpleCSRInternalLinearAlgebraExprFactory();
  }

  static BackEndId name() { return "simplecsr"; }
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
