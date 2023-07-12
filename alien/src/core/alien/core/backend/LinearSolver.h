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
 * \file LinearSolver.h
 * \brief LinearSolver.h
 */
#pragma once

#include <alien/utils/Precomp.h>

#include <memory>

#include <arccore/base/NotImplementedException.h>
#include <arccore/base/TraceInfo.h>

#include <alien/core/backend/BackEnd.h>
#include <alien/core/backend/IInternalLinearSolverT.h>

#include <alien/expression/solver/ILinearSolver.h>

#include <alien/expression/solver/SolverStater.h>
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Space;
class IVector;
class IMatrix;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Linear solver interface
 *
 * Interface for all linear solver package
 *
 * \tparam Tag The tag of the type of solvers used
 */
template <class Tag>
class LinearSolver : public ILinearSolver
{
 public:
  //! The type of the solver
  typedef typename AlgebraTraits<Tag>::solver_type KernelSolver;

 public:
  /*!
   * \brief Creates a linear solver
   *
   * Creates a linear solver using traits and the linear solver factory
   *
   * \tparam T Variadics type of linear solver
   * \param[in] args Linear solvers
   */
  template <typename... T>
  LinearSolver(T... args)
  : m_solver(AlgebraTraits<Tag>::solver_factory(args...))
  , m_stater(m_solver.get())
  {}

  //! Free resources
  virtual ~LinearSolver();

  /*!
   * \brief Get package back end name
   * \returns Package back end name
   */
  Arccore::String getBackEndName() const;

  //! Initialize the linear solver
  void init();

  //! Finalize the linear solver
  void end();

  /*!
   * \brief update parallel_mng, required for redistribution
   * For some solver libraries, Solver is kind of a global object
   * and must be updated accordingly with Matrices and Vectors (Think PETScInit)
   *
   * \param[in] pm : new parallel mng
   */
  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm);

  /*!
   * \brief Solve the linear system A * x = b
   * \param[in] A The matrix to invert
   * \param[in] b The right hand side
   * \param[in,out] x The solution
   * \returns Solver success or failure
   */
  bool solve(const IMatrix& A, const IVector& b, IVector& x);

  /*!
   * \brief Get statistics on the solve phase
   *
   * Get statistics on the solver phase, such as iteration count, initialization time,
   * solve time, etc.
   *
   * \return Solver statistics
   */
  const SolverStat& getSolverStat() const;

  /*!
   * \brief Indicates if the kernel is parallel
   * \returns Parallel support capability
   */
  bool hasParallelSupport() const;

  /*!
   * \brief Get solver resolution status
   * \returns The solver status
   */
  const SolverStatus& getStatus() const;

  /*!
   * \brief Get compatible linear algebra
   * \returns Linear algebra pointer
   */
  std::shared_ptr<ILinearAlgebra> algebra() const;

  /*!
   * \brief Get kernel solver implementation
   * \return Linear solver actual implementation
   */
  KernelSolver* implem();

  /*!
   * \brief Option to add an extra-equation
   *
   * Option to add an extra-equation to the linear system such as a constraint equation
   *
   * \param[in] flag If the option is activated
   *
   * \todo Implement this method
   */
  virtual void setNullSpaceConstantOption(bool flag ALIEN_UNUSED_PARAM)
  {
    throw NotImplementedException(A_FUNCINFO);
  }

 private:
  //! The linear solver kernel
  std::unique_ptr<KernelSolver> m_solver;
  SolverStater<KernelSolver> m_stater;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
