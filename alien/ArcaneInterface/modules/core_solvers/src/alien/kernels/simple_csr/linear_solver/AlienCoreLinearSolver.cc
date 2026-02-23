// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include <vector>

#include "alien/kernels/simple_csr/AlienSolverPrecomp.h"

#include <alien/data/Space.h>
#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/ILinearAlgebra.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/expression/solver/SolverStater.h>

#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>

#include <alien/kernels/simple_csr/linear_solver/AlienCoreLinearSolver.h>

#include <alien/expression/solver/ILinearSolver.h>
#include <alien/core/backend/LinearSolverT.h>
#include <alien/core/backend/SolverFabricRegisterer.h>
#include <alien/core/block/ComputeBlockOffsets.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>


#include <alien/expression/krylov/AlienKrylov.h>
#include <alien/utils/StdTimer.h>

#include <arcane/ArcaneVersion.h>
#include <arcane/Timer.h>

/*---------------------------------------------------------------------------*/

namespace Alien {


/*---------------------------------------------------------------------------*/
AlienCoreLinearSolver::
AlienCoreLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
                     IOptionsAlienCoreSolver* options)
: BaseType(parallel_mng,options)
, m_stater(this)
{
}

void
AlienCoreLinearSolver::init(int argc, char const** argv)
{
  SolverStatSentry<AlienCoreLinearSolver> sentry(m_stater, BaseSolverStater::eInit);
  BaseType::init(argc,argv) ;
}

/*---------------------------------------------------------------------------*/

void
AlienCoreLinearSolver::init()
{
  SolverStatSentry<AlienCoreLinearSolver> sentry(m_stater, BaseSolverStater::eInit);
  BaseType::init() ;
}

/*---------------------------------------------------------------------------*/


bool
AlienCoreLinearSolver::solve(CSRMatrixType const& matrixA,
                             CSRVectorType const& vectorB,
                             CSRVectorType& vectorX)
{
  return BaseType::solve(matrixA,vectorB,vectorX) ;
}

const Alien::SolverStatus&
AlienCoreLinearSolver::getStatus() const
{
  if (this->m_output_level > 0) {
    printInfo();
  }
  return m_status;
}



bool
AlienCoreLinearSolver::solve(IMatrix const& A, IVector const& b, IVector& x)
{
  using namespace Alien;

  SolverStatSentry<AlienCoreLinearSolver> sentry(m_stater, BaseSolverStater::ePrepare);
  CSRMatrixType const& matrix = A.impl()->get<BackEnd::tag::simplecsr>();
  CSRVectorType const& rhs = b.impl()->get<BackEnd::tag::simplecsr>();
  CSRVectorType& sol = x.impl()->get<BackEnd::tag::simplecsr>(true);
  sentry.release();

  SolverStatSentry<AlienCoreLinearSolver> sentry2(m_stater, BaseSolverStater::eSolve);
  return solve(matrix, rhs, sol);
}

void
AlienCoreLinearSolver::setDiagScaling(IMatrix const& matrix)
{
  using namespace Alien;

  SolverStatSentry<AlienCoreLinearSolver> sentry(m_stater, BaseSolverStater::ePrepare);
  CSRMatrixType const& matrixA = matrix.impl()->get<BackEnd::tag::simplecsr>();
  BaseType::setDiagScaling(matrixA) ;
}

void
AlienCoreLinearSolver::setDiagScaling(CSRMatrixType const& matrix)
{
  BaseType::setDiagScaling(matrix) ;
}

std::shared_ptr<ILinearAlgebra>
AlienCoreLinearSolver::algebra() const
{
  return std::shared_ptr<ILinearAlgebra>(new Alien::SimpleCSRLinearAlgebra());
}

ILinearSolver*
AlienCoreLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsAlienCoreSolver* options)
{
  return new AlienCoreLinearSolver(p_mng, options);
}

}

