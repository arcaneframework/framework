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
 * \file IInternalLinearSolverT.h
 * \brief IInternalLinearSolverT.h
 */
#pragma once

#include <alien/utils/Precomp.h>
#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
class IMessagePassingMng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ILinearAlgebra;
class SolverStat;
class SolverStatus;

/*!
 * \ingroup core
 * \brief Internal linear solver interface
 *
 * Internal interface for all linear solver package
 *
 * \tparam Matrix The type of matrix used
 * \tparam Vector The type of vector used
 */
template <class Matrix, class Vector>
class IInternalLinearSolver
{
 public:
  //! Free resources
  virtual ~IInternalLinearSolver() {}

  /*!
   * \brief Update parallel manager
   *
   * Allows to change parallel manager in cases where it changes, like solves in a
   * redistributed environment
   *
   * \param[in] pm The new parallel manager
   */
  virtual void updateParallelMng(ALIEN_UNUSED_PARAM Arccore::MessagePassing::IMessagePassingMng* pm) {}

  //! Initialize the linear solver
  virtual void init() {}

  //! Finalize the linear solver
  virtual void end() {}

  /*!
   * \brief Solve the linear system A * x = b
   * \param[in] A The matrix to invert
   * \param[in] b The right hand side
   * \param[in,out] x The solution
   * \returns Solver success or failure
   */
  virtual bool solve(const Matrix& A, const Vector& b, Vector& x) = 0;

  /*!
   * \brief Get statistics on the solve phase
   *
   * Get statistics on the solver phase, such as iteration count, initialization time,
   * solve time, etc.
   *
   * \return Solver statistics
   */
  virtual SolverStat const& getSolverStat() const = 0;

  /*!
   * \brief Get statistics on the solve phase
   *
   * Get statistics on the solver phase, such as iteration count, initialization time,
   * solve time, etc.
   *
   * \return Solver statistics
   */
  virtual SolverStat& getSolverStat() = 0;

  /*!
   * \brief Indicates if the kernel is parallel
   * \returns Parallel support capability
   */
  virtual bool hasParallelSupport() const = 0;

  /*!
   * \brief Get solver resolution status
   * \returns The solver status
   */
  virtual const SolverStatus& getStatus() const = 0;

  /*!
   * \brief Get compatible linear algebra
   * \returns Linear algebra pointer
   */
  virtual std::shared_ptr<ILinearAlgebra> algebra() const
  {
    return std::shared_ptr<ILinearAlgebra>();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
