// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include <vector>

#include "alien/kernels/trilinos/TrilinosPrecomp.h"
#include <alien/kernels/trilinos/data_structure/TrilinosVector.h>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
#include <alien/kernels/trilinos/algebra/TrilinosLinearAlgebra.h>
#include <alien/kernels/trilinos/algebra/TrilinosInternalLinearAlgebra.h>
#include <alien/kernels/trilinos/data_structure/TrilinosInternal.h>

#include <alien/data/Space.h>
#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>
#include <alien/expression/solver/IEigenSolver.h>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/trilinos/TrilinosPrecomp.h>
#include <alien/core/backend/EigenSolverT.h>
#include <ALIEN/axl/TrilinosEigenSolver_IOptions.h>

#include <alien/kernels/trilinos/eigen_solver/TrilinosInternalEigenSolver.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
TrilinosInternalEigenSolver::TrilinosInternalEigenSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    IOptionsTrilinosEigenSolver* options)
: m_parallel_mng(parallel_mng)
, m_options(options)
{
}

void
TrilinosInternalEigenSolver::init([[maybe_unused]] int argc,[[maybe_unused]] char const** argv)
{
#ifdef ALIEN_USE_TRILINOSSOLVER
// m_hts_solver.reset(new HartsSolver::HTSSolver()) ;
#endif
}

/*---------------------------------------------------------------------------*/

void
TrilinosInternalEigenSolver::init()
{
  m_output_level = m_options->output();
}

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

const Alien::IEigenSolver::Status&
TrilinosInternalEigenSolver::getStatus() const
{
  return m_status;
}

bool
TrilinosInternalEigenSolver::solve([[maybe_unused]] EigenProblem& p)
{
  using namespace Alien;

#ifdef ALIEN_USE_TRILINOSSOLVER
// EigenProblemT<BackEnd::tag::simplecsr> true_problem(p) ;
// return solve(true_problem) ;
#endif
  return false;
}

bool
TrilinosInternalEigenSolver::solve([[maybe_unused]] GeneralizedEigenProblem& p)
{
  using namespace Alien;

#ifdef ALIEN_USE_TRILINOSSOLVER
// EigenProblemT<BackEnd::tag::simplecsr> true_problem(p) ;
// return solve(true_problem) ;
#endif
  return false;
}

IEigenSolver*
TrilinosInternalEigenSolverFactory(Arccore::MessagePassing::IMessagePassingMng* p_mng,
    IOptionsTrilinosEigenSolver* options)
{
  return new TrilinosInternalEigenSolver(p_mng, options);
}
} // namespace Alien
