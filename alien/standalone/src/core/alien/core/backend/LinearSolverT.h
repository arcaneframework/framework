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

/*!
 * \file LinearSolverT.h
 * \brief LinearSolverT.h
 */

#include <alien/core/backend/LinearSolver.h>

#include <alien/expression/solver/SolverStat.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/data/IMatrix.h>
#include <alien/data/IVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
LinearSolver<Tag>::~LinearSolver() {}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
String
LinearSolver<Tag>::getBackEndName() const
{
  return AlgebraTraits<Tag>::name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \todo Check the comment below and fix if necessary
 */
template <class Tag>
bool LinearSolver<Tag>::solve(const IMatrix& A, const IVector& b, IVector& x)
{
  auto dist = A.impl()->distribution();
  // solve is a global call
  // when redistribution, matrix or dist could be null on some subsets of procs
  // so we return true to don't block
  // FIXME: sync result ??
  if (A.impl() == nullptr || dist.parallelMng() == nullptr)
    return true;
  // solver is not parallel but dist is
  if (not m_solver->hasParallelSupport() && dist.isParallel()) {
    throw FatalErrorException("solver is not parallel");
  }
  else {
    m_solver->updateParallelMng(A.impl()->distribution().parallelMng());
  }

  SolverStatSentry<KernelSolver> sentry(m_stater, BaseSolverStater::ePrepare);
  const auto& matrix = A.impl()->get<Tag>();
  const auto& rhs = b.impl()->get<Tag>();
  auto& sol = x.impl()->get<Tag>(true);
  sentry.release();

  SolverStatSentry<KernelSolver> sentry2(m_stater, BaseSolverStater::eSolve);
  return m_solver->solve(matrix, rhs, sol);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
void LinearSolver<Tag>::init()
{
  SolverStatSentry<KernelSolver> sentry(m_stater, BaseSolverStater::eInit);
  m_solver->init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
void LinearSolver<Tag>::end()
{
  m_solver->end();
}

template <class Tag>
void LinearSolver<Tag>::updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm)
{
  m_solver->updateParallelMng(pm);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
const SolverStat&
LinearSolver<Tag>::getSolverStat() const
{
  return m_solver->getSolverStat();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
bool LinearSolver<Tag>::hasParallelSupport() const
{
  return m_solver->hasParallelSupport();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
const SolverStatus&
LinearSolver<Tag>::getStatus() const
{
  return m_solver->getStatus();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
typename LinearSolver<Tag>::KernelSolver*
LinearSolver<Tag>::implem()
{
  return m_solver.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
std::shared_ptr<ILinearAlgebra>
LinearSolver<Tag>::algebra() const
{
  return m_solver->algebra();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
