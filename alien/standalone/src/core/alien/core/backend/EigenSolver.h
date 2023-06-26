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

#pragma once

#include <alien/utils/Precomp.h>

#include <memory>
#include <vector>

#include <alien/core/backend/BackEnd.h>
#include <alien/expression/solver/IEigenSolver.h>

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

class IVector;
class IMatrix;
class IOptions;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Eigen problem definition
 *
 * Allows to specify the matrix and get the number of eigen vectors as well as the eigen
 * vectors
 *
 * \tparam Tag The type of kernel used to compute eigen values
 * \tparam VectorT The type of kernel used to store eigen vectors
 */
// FIXME: not implemented !
template <class Tag, typename VectorT>
class EigenProblemT : public EigenProblem
{
 public:
  /*!
   * \brief Eigen problem constructor
   * \param[in] A The matrix of the eigen problem
   */
  EigenProblemT(IMatrix const& A)
  : EigenProblem(A)
  {}

  //! Free resources
  virtual ~EigenProblemT() {}

  //! Type of the matrix used
  typedef typename AlgebraTraits<Tag>::matrix_type KernelMatrix;
  //! Type of the vector used
  typedef typename AlgebraTraits<Tag>::vector_type KernelVector;

  /*!
   * \brief Get the local size of the problem
   * \return Local size of the problem
   */
  Arccore::Integer localSize() const;

  /*!
   * \brief Get the eigen matrix
   * \return The eigen matrix
   */
  KernelMatrix const& getA() const;

  /*!
   * \brief Get the number of eigen vectors
   * \return The number of eigen vectors
   */
  Arccore::Integer getNbEigenVectors() const { return m_eigen_vectors.size(); }

  /*!
   * \brief Get the eigen vectors
   * \return The eigen vectors
   */
  std::vector<VectorT>& getEigenVectors() { return m_eigen_vectors; }

  // TOCHECK: Commentaire
  /*
  void setEigenVectors(std::vector<std::vector<Real>>& ev_vectors)
  {
    auto const& vdist = this->m_A.impl()->distribution().rowDistribution() ;
    for(auto const ev : ev_vectors)
    {
      VectorT v(vdist) ;
      Alien::LocalVectorWriter vv(v) ;
      for(auto i=0;i<vdist.localSize();++i)
      {
        vv[i] = ev[i] ;
        m_eigen_vectors.add(v) ;
      }
    }
  }*/

 private:
  //! Eigen vectors
  std::vector<VectorT> m_eigen_vectors;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Generalized eigen problem definition
 *
 * Allows to specify the matrices and get the number of generalized eigen vectors as well
 * as the generalized eigen vectors
 *
 * \tparam Tag The type of kernel used to compute eigen values
 * \tparam VectorT The type of kernel used to store eigen vectors
 */
template <class Tag, typename VectorT>
class GeneralizedEigenProblemT : public GeneralizedEigenProblem
{
 public:
  /*!
   * \brief Generalized eigen problem constructor
   * \param[in] A The first matrix of the generalized eigen problem
   * \param[in] B The second matrix of the generalized eigen problem
   */
  GeneralizedEigenProblemT(IMatrix const& A, IMatrix const& B)
  : GeneralizedEigenProblem(A, B)
  {}

  //! Free resources
  virtual ~GeneralizedEigenProblemT() {}

  //! Type of the matrix used
  typedef typename AlgebraTraits<Tag>::matrix_type KernelMatrix;
  //! Type of the vector used
  typedef typename AlgebraTraits<Tag>::vector_type KernelVector;

  /*!
   * \brief Get the local size of the problem
   * \return Local size of the problem
   */
  Arccore::Integer localSize() const;

  /*!
   * \brief Get the first eigen matrix of the generalized eigen problem
   * \return The first eigen matrix
   */
  KernelMatrix const& getA() const;

  /*!
   * \brief Get the second eigen matrix of the generalized eigen problem
   * \return The second eigen matrix
   */
  KernelMatrix const& getB() const;

  /*!
   * \brief Get the number of eigen vectors
   * \return The number of eigen vectors
   */
  Arccore::Integer getNbEigenVectors() const { return m_eigen_vectors.size(); }

  /*!
   * \brief Get the eigen vectors
   * \return The eigen vectors
   */
  std::vector<VectorT>& getEigenVectors() { return m_eigen_vectors; }

 private:
  //! Eigen vectors
  std::vector<VectorT> m_eigen_vectors;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Eigen solver
 *
 * Solves an eigen problem and retrieve eigen vectors
 *
 * \tparam Tag The type of kernel used to compute eigen values
 */
template <class Tag>
class EigenSolver : public IEigenSolver
{
 public:
  //! Type of the eigen solver used
  typedef typename AlgebraTraits<Tag>::eigen_solver_type KernelSolver;

 public:
  /*!
   * \brief Eigen solver constructor
   * \param[in] parallel_mng The parallel manager for parallel solve
   * \param[in] options Options passed to the eigen solver
   */
  EigenSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
              IOptions* options = nullptr);

  //! Free resources
  virtual ~EigenSolver() {}

  /*!
   * \brief Get kernel back end name
   * \returns The kernel name as a string
   */
  Arccore::String getBackEndName() const;

  //! Initialize the eigen solver
  void init();

  /*!
   * \brief Solve the eigen problem
   * \param[in,out] p The eigen problem
   * \returns Solver success status
   */
  bool solve(EigenProblem& p);

  /*!
   * \brief Indicates if the kernel is parallel
   * \returns Parallel support capability
   */
  bool hasParallelSupport() const;

  /*!
   * \brief Get solver resolution status
   * \returns The solver status
   */
  const IEigenSolver::Status& getStatus() const;

  /*!
   * \brief Get kernel solver implementation
   * \return Eigen solver actual implementation
   */
  KernelSolver* implem();

 private:
  //! Type of the matrix used
  typedef typename AlgebraTraits<Tag>::matrix_type KernelMatrix;
  //! Type of the vector used
  typedef typename AlgebraTraits<Tag>::vector_type KernelVector;
  //! The eigen solver
  std::unique_ptr<KernelSolver> m_solver;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup core
 * \brief Generalized eigen solver
 *
 * Solves a generalized eigen problem and retrieve eigen vectors
 *
 * \tparam Tag The type of kernel used to compute eigen values
 */
template <class Tag>
class GeneralizedEigenSolver : public IGeneralizedEigenSolver
{
 public:
  //! Type of the eigen solver used
  typedef typename AlgebraTraits<Tag>::generalized_eigen_solver_type KernelSolver;

 public:
  /*!
   * \brief Eigen solver constructor
   * \param[in] parallel_mng The parallel manager for parallel solve
   * \param[in] options Options passed to the eigen solver
   */
  GeneralizedEigenSolver(
  IMessagePassingMng* parallel_mng = nullptr, IOptions* options = nullptr);

  //! Free resources
  virtual ~GeneralizedEigenSolver() {}

  /*!
   * \brief Get kernel back end name
   * \returns The kernel name as a string
   */
  Arccore::String getBackEndName() const;

  //! Initialize the eigen solver
  void init();

  /*!
   * \brief Solve the eigen problem
   * \param[in,out] A The generalized eigen problem
   * \returns Solver success status
   */
  bool solve(GeneralizedEigenProblem& A);

  /*!
   * \brief Indicates if the kernel is parallel
   * \returns Parallel support capability
   */
  bool hasParallelSupport() const;

  /*!
   * \brief Get solver resolution status
   * \returns The solver status
   */
  const IEigenSolver::Status& getStatus() const;

  /*!
   * \brief Get kernel solver implementation
   * \return Eigen solver actual implementation
   */
  KernelSolver* implem();

 private:
  //! Type of the matrix used
  typedef typename AlgebraTraits<Tag>::matrix_type KernelMatrix;
  //! Type of the vector used
  typedef typename AlgebraTraits<Tag>::vector_type KernelVector;
  //! The generalized eigen solver
  std::unique_ptr<KernelSolver> m_solver;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
