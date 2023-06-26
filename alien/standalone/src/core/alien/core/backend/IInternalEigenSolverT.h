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
 * \file IInternalEigenSolverT.h
 * \brief IInternalEigenSolverT.h
 */
#pragma once

#include <alien/utils/Precomp.h>
#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EigenProblem;

/*!
 * \ingroup core
 * \brief Eigen solver interface
 *
 * Interface for all eigen solver package
 *
 * \tparam Matrix The type of matrix used
 * \tparam Vector The type of vector used
 */
template <class Matrix, class Vector>
class IInternalEigenSolver
{
 public:
  //! Free resources
  virtual ~IInternalEigenSolver() {}

  //! Initialize the eigen solver
  virtual void init() = 0;

  /*!
   * \brief Solve the eigen problem
   * \param[in,out] problem The eigen problem
   * \returns Solver success status
   */
  virtual bool solve(const EigenProblem& problem) = 0;

  /*!
   * \brief Indicates if the kernel is parallel
   * \returns Parallel support capability
   */
  virtual bool hasParallelSupport() const = 0;

  /*!
   * \brief Get solver resolution status
   * \returns The solver status
   */
  virtual const IEigenSolver::Status& getStatus() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Generalized eigen solver internal
 *
 * Interface for all generalized eigen solver package
 *
 * \tparam Matrix The type of matrix used
 * \tparam Vector The type of vector used
 */
template <class Matrix, class Vector>
class IInternalGeneralizedEigenSolver
{
 public:
  //! Free resources
  virtual ~IInternalGeneralizedEigenSolver() {}

  //! Initialize the eigen solver
  virtual void init() = 0;

  /*!
   * \brief Solve the eigen problem
   * \param[in,out] problem The eigen problem
   * \returns Solver success status
   */
  virtual bool solve(const GeneralizedEigenProblem& problem) = 0;

  /*!
   * \brief Indicates if the kernel is parallel
   * \returns Parallel support capability
   */
  virtual bool hasParallelSupport() const = 0;

  /*!
   * \brief Get solver resolution status
   * \returns The solver status
   */
  virtual const IEigenSolver::Status& getStatus() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
