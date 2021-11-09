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
 * \file IEigenSolver.h
 * \brief IEigenSolver.h
 */

#pragma once

#include <alien/utils/Precomp.h>
#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMatrix;

class IVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup solver
 * \brief Defines an eigen problem
 */
class EigenProblem
{
 public:
  /*!
   * \brief Constructor
   * \param[in] A The matrix
   */
  EigenProblem(IMatrix const& A)
  : m_A(A)
  {}

  //! Free resources
  virtual ~EigenProblem() {}

  /*!
   * \brief Get the matrix
   * \returns The matrix
   */
  IMatrix const& getA() const { return m_A; }

  /*!
   * \brief Get real eigen values
   * \returns The real eigen values
   */
  std::vector<Arccore::Real> const& getRealEigenValues() const
  {
    return m_real_eigen_values;
  }

  /*!
   * \brief Get real eigen values
   * \returns The real eigen values
   */
  std::vector<Arccore::Real>& getRealEigenValues() { return m_real_eigen_values; }

  /*!
   * \brief Get imaginary eigen values
   * \returns The imaginary eigen values
   */
  std::vector<Arccore::Real> const& getImaginaryEigenValues() const
  {
    return m_imaginary_eigen_values;
  }

  /*!
   * \brief Get imaginary eigen values
   * \returns The imaginary eigen values
   */
  std::vector<Arccore::Real>& getImaginaryEigenValues()
  {
    return m_imaginary_eigen_values;
  }

  /*!
   * \brief Get the number of eigen vectors
   * \returns The number of eigen vectors
   */
  virtual Arccore::Integer getNbEigenVectors() const = 0;

 protected:
  //! The eigen matrix
  IMatrix const& m_A;
  //! The real eigen values
  std::vector<Arccore::Real> m_real_eigen_values;
  //! The imaginary eigen values
  std::vector<Arccore::Real> m_imaginary_eigen_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup solver
 * \brief Defines a generalized eigen problem
 */
class GeneralizedEigenProblem : public EigenProblem
{
 public:
  /*!
   * \brief Constructor
   * \param[in] A The first matrix
   * \param[in] B The second matrix
   */
  GeneralizedEigenProblem(IMatrix const& A, IMatrix const& B)
  : EigenProblem(A)
  , m_B(B)
  {}

  //! Free resources
  virtual ~GeneralizedEigenProblem() {}

  /*!
   * \brief Get the second matrix
   * \returns The second matrix
   */
  IMatrix const& getB() const { return m_B; }

 protected:
  //! The second matrix
  IMatrix const& m_B;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup expression
 * \brief Eigen solver interface
 */
class IEigenSolver
{
 public:
  //! Eigen solver status
  struct Status
  {
    bool m_succeeded = false;
    Integer m_nconv = 0;
    Real m_residual = -1.;
    Integer m_iteration_count = 0;
    Integer m_error = 0;
  };

  //! Eigen values order
  typedef enum
  {
    SmallestMagnitude,
    LargestMagnitude,
    SmallestReal,
    LargestReal,
    SmallestImaginary,
    LargestImaginary
  } eEigenValuesOrder;

 public:
  //! Constructor
  IEigenSolver() {}

  //! Free resources
  virtual ~IEigenSolver(){};

 public:
  /*!
   * \brief Get back end name
   * \returns The back end name
   */
  virtual Arccore::String getBackEndName() const = 0;

  //! Initialization
  virtual void init() = 0;

  /*!
   * \brief Solves an eigen problem
   * \param[in] problem The eigen problem
   * \returns Whether or not the solver succeeded
   */
  virtual bool solve(EigenProblem& problem) = 0;

  /*
   * \brief Whether or not the solver is parallel
   * \returns Parallel support of the solver
   */
  virtual bool hasParallelSupport() const = 0;

  /*!
   * \brief Get solves status
   * \returns Solver status
   */
  virtual const Status& getStatus() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup expression
 * \brief Interface for generalized eigen solver
 */
class IGeneralizedEigenSolver : public IEigenSolver
{
 public:
  //! Constructor
  IGeneralizedEigenSolver()
  : IEigenSolver()
  {}

  //! Free resources
  virtual ~IGeneralizedEigenSolver(){};

 public:
  /*!
   * \brief Solve an eigen problem
   * \param[in] problem The eigen problem
   * \returns Whether or not the problem was solved
   */
  virtual bool solve(EigenProblem& problem) = 0;

  /*!
   * \brief Solve a generalized eigen problem
   * \param[in] problem The generalized eigen problem
   * \returns Whether or not the problem was solved
   */
  virtual bool solve(GeneralizedEigenProblem& problem) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
