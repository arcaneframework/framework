// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include <vector>

#include "alien/kernels/hts/HTSPrecomp.h"

#ifdef ALIEN_USE_HARTS
#include "HARTS/HARTS.h"
#endif

#ifdef ALIEN_USE_HTSSOLVER
#include "HARTSSolver/HTS.h"
#include "HARTSSolver/MatrixVector/CSR/CSRMatrixImpT.h"
#endif

#include <alien/data/Space.h>
#include <alien/expression/solver/IEigenSolver.h>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/kernels/hts/eigen_solver/HTSInternalEigenSolver.h>
#include <alien/core/backend/EigenSolverT.h>
#include <ALIEN/axl/HTSEigenSolver_IOptions.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
HTSInternalEigenSolver::HTSInternalEigenSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    IOptionsHTSEigenSolver* options)
: m_parallel_mng(parallel_mng)
, m_options(options)
{
}

void
HTSInternalEigenSolver::init(int argc, char const** argv)
{
#ifdef ALIEN_USE_HTSSOLVER
// m_hts_solver.reset(new HartsSolver::HTSSolver()) ;
#endif
}

/*---------------------------------------------------------------------------*/

void
HTSInternalEigenSolver::init()
{
  m_output_level = m_options->output();

#ifdef ALIEN_USE_HTSSOLVER
  m_use_mpi = m_parallel_mng->commSize() > 1;
  m_machine_info.init(m_parallel_mng->commRank() == 0);

  m_hts_solver.reset(new HartsSolver::HTSSolver());
  m_hts_solver->setMachineInfo(&m_machine_info);
#endif
  m_current_ctx_id = m_hts_solver->createNewContext();
  HartsSolver::HTSSolver::ContextType& context =
      m_hts_solver->getContext(m_current_ctx_id);

  m_hts_solver->setCurrentContext(&context);

  m_hts_solver->setParameter<int>("output", m_output_level);
  m_hts_solver->setParameter<int>("max-iteration", m_options->maxIterationNum());
  m_hts_solver->setParameter<double>("tol", m_options->tol());
  m_hts_solver->setParameter<int>("ev-type", m_options->evType());
  m_hts_solver->setParameter<int>("nev", m_options->nev());
  m_hts_solver->setParameter<int>("ev-order", m_options->evOrder());
  m_hts_solver->setParameter<double>("ev-bound", m_options->evBound());
}

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

const Alien::IEigenSolver::Status&
HTSInternalEigenSolver::getStatus() const
{
  return m_status;
}

bool
HTSInternalEigenSolver::solve(EigenProblem& p)
{
  using namespace Alien;

#ifdef ALIEN_USE_HTSSOLVER
// EigenProblemT<BackEnd::tag::simplecsr> true_problem(p) ;
// return solve(true_problem) ;
#endif
  return false;
}

bool
HTSInternalEigenSolver::solve(GeneralizedEigenProblem& p)
{
  using namespace Alien;

#ifdef ALIEN_USE_HTSSOLVER
// EigenProblemT<BackEnd::tag::simplecsr> true_problem(p) ;
// return solve(true_problem) ;
#endif
  return false;
}

IEigenSolver*
HTSInternalEigenSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsHTSEigenSolver* options)
{
  return new HTSInternalEigenSolver(p_mng, options);
}
} // namespace Alien
