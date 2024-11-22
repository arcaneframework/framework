// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once
#include <alien/utils/Precomp.h>
#include <alien/core/backend/BackEnd.h>
#include <arccore/message_passing/IMessagePassingMng.h>

/*---------------------------------------------------------------------------*/

class IOptionsComposyxSolver;

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

template<typename T> class ComposyxMatrix ;
template<typename T> class ComposyxVector ;

template<typename T> class SimpleCSRMatrix ;
template<typename T> class SimpleCSRMatrix ;

class Space;

template <class Matrix, class Vector> class IInternalLinearAlgebra;
template <class Matrix, class Vector> class IInternalLinearSolver;

//extern IInternalLinearAlgebra<SimpleCSRMatrix<Real>, SimpleCSRMatrix<Real>>*
//ComposyxSolverInternalLinearAlgebraFactory();

// extern IInternalLinearSolver<ComposyxMatrix<Real>, ComposyxVector<Real>>*
extern ILinearSolver* ComposyxInternalLinearSolverFactory(
    IMessagePassingMng* p_mng, IOptionsComposyxSolver* options);

extern IInternalLinearAlgebra<SimpleCSRMatrix<Real>, SimpleCSRMatrix<Real>>*
ComposyxInternalLinearAlgebraFactory();

/*---------------------------------------------------------------------------*/

namespace BackEnd {
  namespace tag {
    struct composyx
    {
    };
  }
}

template <> struct AlgebraTraits<BackEnd::tag::composyx>
{
  typedef ComposyxMatrix<Real> matrix_type;
  typedef ComposyxVector<Real> vector_type;

  typedef SimpleCSRMatrix<Real> csr_matrix_type;
  typedef SimpleCSRMatrix<Real> csr_vector_type;
  typedef IInternalLinearAlgebra<csr_matrix_type, csr_vector_type> algebra_type;

  // typedef IInternalLinearSolver<matrix_type, vector_type>  solver_type;
  typedef ILinearSolver solver_type;
  typedef IOptionsComposyxSolver options_type;

  static algebra_type* algebra_factory(IMessagePassingMng* p_mng = nullptr)
  {
    return ComposyxInternalLinearAlgebraFactory();
  }

  static solver_type* solver_factory(IMessagePassingMng* p_mng, options_type* options)
  {
    return ComposyxInternalLinearSolverFactory(p_mng, options);
  }

  static BackEndId name() { return "composyx"; }
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
