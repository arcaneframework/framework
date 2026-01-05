// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include <vector>

#include "alien/kernels/sycl/AlienSolverSYCLPrecomp.h"

#include <alien/ref/AlienRefSemantic.h>
#include <alien/AlienLegacyConfig.h>

#include <alien/data/Space.h>
#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/ILinearAlgebra.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/expression/solver/SolverStater.h>

#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>

#ifdef ALIEN_USE_SYCL
#include <alien/kernels/sycl/SYCLPrecomp.h>

#include "alien/kernels/sycl/data/SYCLEnv.h"
#include "alien/kernels/sycl/data/SYCLEnvInternal.h"

#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>
#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>

#include "alien/kernels/sycl/data/SYCLVectorInternal.h"
#include <alien/kernels/sycl/data/SYCLBEllPackInternal.h>
#include <alien/kernels/sycl/algebra/SYCLInternalLinearAlgebra.h>

#include "alien/kernels/sycl/data/SYCLSendRecvOp.h"
#include "alien/kernels/sycl/data/SYCLLUSendRecvOp.h"
#include <alien/kernels/sycl/algebra/SYCLKernelInternal.h>
#endif

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>

#include <alien/kernels/sycl/linear_solver/AlienCoreSYCLLinearSolver.h>

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
AlienCoreSYCLLinearSolver::
AlienCoreSYCLLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
                     IOptionsAlienCoreSolver* options)
: BaseType(parallel_mng,options)
, m_stater(this)
{
}

void
AlienCoreSYCLLinearSolver::init(int argc, char const** argv)
{
  SolverStatSentry<AlienCoreSYCLLinearSolver> sentry(m_stater, BaseSolverStater::eInit);
  BaseType::init(argc,argv) ;
}

/*---------------------------------------------------------------------------*/

void
AlienCoreSYCLLinearSolver::init()
{
  SolverStatSentry<AlienCoreSYCLLinearSolver> sentry(m_stater, BaseSolverStater::eInit);
  BaseType::init() ;
}

/*---------------------------------------------------------------------------*/


bool
AlienCoreSYCLLinearSolver::solve(MatrixType const& matrixA,
                                 VectorType const& vectorB,
                                 VectorType& vectorX)
{
  return BaseType::solve(matrixA,vectorB,vectorX) ;
}

const Alien::SolverStatus&
AlienCoreSYCLLinearSolver::getStatus() const
{
  if (this->m_output_level > 0) {
    printInfo();
  }
  return this->m_status;
}



bool
AlienCoreSYCLLinearSolver::solve(IMatrix const& A, IVector const& b, IVector& x)
{
  using namespace Alien;

  SolverStatSentry<AlienCoreSYCLLinearSolver> sentry(m_stater, BaseSolverStater::ePrepare);
  MatrixType const& matrix = A.impl()->get<BackEnd::tag::sycl>();
  VectorType const& rhs = b.impl()->get<BackEnd::tag::sycl>();
  VectorType& sol = x.impl()->get<BackEnd::tag::sycl>(true);
  sentry.release();

  SolverStatSentry<AlienCoreSYCLLinearSolver> sentry2(m_stater, BaseSolverStater::eSolve);
  return solve(matrix, rhs, sol);
}

std::shared_ptr<ILinearAlgebra>
AlienCoreSYCLLinearSolver::algebra() const
{
  return std::shared_ptr<ILinearAlgebra>(new Alien::SYCLLinearAlgebra());
}

ILinearSolver*
AlienCoreSYCLLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsAlienCoreSolver* options)
{
  return new AlienCoreSYCLLinearSolver(p_mng, options);
}

}

