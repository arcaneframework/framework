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
 * \file EigenSolverT.h
 * \brief EigenSolverT.h
 */

#pragma once

#include <alien/core/backend/EigenSolver.h>

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

template <class TagT, typename VectorT>
Integer
EigenProblemT<TagT, VectorT>::localSize() const
{
  return this->m_A.impl()->distribution().rowDistribution().localSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class TagT, typename VectorT>
typename EigenProblemT<TagT, VectorT>::KernelMatrix const&
EigenProblemT<TagT, VectorT>::getA() const
{
  return this->m_A.impl()->template get<TagT>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class TagT, typename VectorT>
Integer
GeneralizedEigenProblemT<TagT, VectorT>::localSize() const
{
  return this->m_A.impl()->distribution().rowDistribution().localSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class TagT, typename VectorT>
typename GeneralizedEigenProblemT<TagT, VectorT>::KernelMatrix const&
GeneralizedEigenProblemT<TagT, VectorT>::getA() const
{
  return this->m_A.impl()->template get<TagT>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class TagT, typename VectorT>
typename GeneralizedEigenProblemT<TagT, VectorT>::KernelMatrix const&
GeneralizedEigenProblemT<TagT, VectorT>::getB() const
{
  return this->m_B.impl()->template get<TagT>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
EigenSolver<Tag>::EigenSolver(IMessagePassingMng* parallel_mng, IOptions* options)
: m_solver(AlgebraTraits<Tag>::eigen_solver_factory(parallel_mng, options))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
String
EigenSolver<Tag>::getBackEndName() const
{
  return AlgebraTraits<Tag>::name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
void EigenSolver<Tag>::init()
{
  m_solver->init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
bool EigenSolver<Tag>::solve(EigenProblem& p)
{
  return m_solver->solve(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
bool EigenSolver<Tag>::hasParallelSupport() const
{
  return m_solver->hasParallelSupport();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
const IEigenSolver::Status&
EigenSolver<Tag>::getStatus() const
{
  return m_solver->getStatus();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
typename EigenSolver<Tag>::KernelSolver*
EigenSolver<Tag>::implem()
{
  return m_solver.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
GeneralizedEigenSolver<Tag>::GeneralizedEigenSolver(
IMessagePassingMng* parallel_mng, IOptions* options)
: m_solver(AlgebraTraits<Tag>::eigen_solver_factory(parallel_mng, options))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
String
GeneralizedEigenSolver<Tag>::getBackEndName() const
{
  return AlgebraTraits<Tag>::name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
void GeneralizedEigenSolver<Tag>::init()
{
  m_solver->init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
bool GeneralizedEigenSolver<Tag>::solve(GeneralizedEigenProblem& p)
{
  return m_solver->solve(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
const IEigenSolver::Status&
GeneralizedEigenSolver<Tag>::getStatus() const
{
  return m_solver->getStatus();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Tag>
typename GeneralizedEigenSolver<Tag>::KernelSolver*
GeneralizedEigenSolver<Tag>::implem()
{
  return m_solver.get();
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
