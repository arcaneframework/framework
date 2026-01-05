// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/utils/Precomp.h>
#include <alien/core/backend/BackEnd.h>
#include <arccore/message_passing/IMessagePassingMng.h>

/*---------------------------------------------------------------------------*/

class IOptionsAlienSolver;

namespace Alien {

/*---------------------------------------------------------------------------*/

class MultiVectorImpl;
class MatrixData;
class MatrixExp;
class VectorData;
class VectorData;
class VectorExp;
class ILinearSolver;
class ILinearAlgebra;

template <typename T> class SimpleCSRMatrix;
template <typename T> class SimpleCSRVector;

class Space;

template <class Matrix, class Vector> class IInternalLinearAlgebra;
template <class Matrix, class Vector> class IInternalLinearSolver;

extern IInternalLinearAlgebra<SimpleCSRMatrix<Real>, SimpleCSRVector<Real>>*
AlienSolverInternalLinearAlgebraFactory();

// extern IInternalLinearSolver<SimpleCSRMatrix<Real>, SimpleCSRVector<Real>>*
extern ILinearSolver* AlienInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsAlienSolver* options);

extern IInternalLinearAlgebra<HTSMatrix<Real>, SimpleCSRVector<Real>>*
AlienInternalLinearAlgebraFactory();

/*---------------------------------------------------------------------------*/

namespace BackEnd {
  namespace tag {
    struct aliensolver
    {
    };
    struct alien
    {
    };
  }
}

template <> struct AlgebraTraits<BackEnd::tag::aliensolver>
{
  typedef SimpleCSRMatrix<Real> matrix_type;
  typedef SimpleCSRVector<Real> vector_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;
  // typedef IInternalLinearSolver<matrix_type, vector_type>  solver_type;
  typedef ILinearSolver solver_type;
  typedef IOptionsHTSSolver options_type;

  static algebra_type* algebra_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng = nullptr)
  {
    return AlienSolverInternalLinearAlgebraFactory();
  }

  static solver_type* solver_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng, options_type* options)
  {
    return AlienInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "aliensolver"; }
};

template <> struct AlgebraTraits<BackEnd::tag::alien>
{
  typedef SimpleCSRMatrix<Real> matrix_type;
  typedef SimpleCSRVector<Real> vector_type;
  typedef IInternalLinearAlgebra<matrix_type, vector_type> algebra_type;

  static algebra_type* algebra_factory(
      Arccore::MessagePassing::IMessagePassingMng* p_mng = nullptr)
  {
    return AlienInternalLinearAlgebraFactory();
  }

  static BackEndId name() { return "alien"; }
};
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
