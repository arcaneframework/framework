// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

/*!
 * \file ILinearSolver.h
 * \brief ILinearSolver.h
 */

#pragma once

#include <arccore/message_passing/MessagePassingGlobal.h>
#include <alien/utils/Precomp.h>

#include <cstdlib>
#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ILinearAlgebra;
class Space;
class SolverStat;
class VectorData;
class IMatrix;
class IVector;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup expression
 * \brief Structure to store a solver status
 */
struct SolverStatus
{
  //! Constructor
  SolverStatus()
  : succeeded(false)
  , residual(0)
  , iteration_count(0)
  , error(0)
  {}

  //! Whether or not the solver succeeded
  bool succeeded;
  //! The residual
  Arccore::Real residual;
  //! The number of iterations
  Arccore::Integer iteration_count;
  //! The error
  Arccore::Integer error;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup expression
 * \brief Linear solver interface
 */
class ILinearSolver
{
 public:
  /*!
   * \brief Type of the solver status
   * \todo Change to a more abstract implementation based on CaseOptions
   */
  typedef SolverStatus Status;

 public:
  //! Constructor
  ILinearSolver() {}

  //! Free resources
  virtual ~ILinearSolver() {}

 public:
  /*
   * \brief Get the back end name
   * \returns The back end name
   */
  virtual Arccore::String getBackEndName() const = 0;

  //! Initialization
  virtual void init() = 0;

  //! Finalization
  virtual void end() = 0;

  /*!
   * \brief update parallel_mng, required for redistribution
   * \param[in] pm : new parallel mng
   */
  virtual void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm) = 0;

  /*!
   * \brief Solves a linear system
   * \param[in] A The matrix
   * \param[in] b The rhs
   * \param[in, out] The solution
   * \returns Whether or not the solver succeeded
   */
  virtual bool solve(const IMatrix& A, const IVector& b, IVector& x) = 0;

  /*
   * \brief Get statistics on the solve process
   * \returns Statistics on the solve process
   */
  virtual SolverStat const& getSolverStat() const = 0;

  /*!
   * \brief Get a compatible linear algebra, i.e. a linear algebra matching the solver
   * kernel \returns A compatible linear algebra
   */
  virtual std::shared_ptr<ILinearAlgebra> algebra() const = 0;

  /*!
   * \brief Whether or not the solver support parallel solve
   * \returns Whether or not the solver is parallel
   */
  virtual bool hasParallelSupport() const = 0;

  /*!
   * \brief Get resolution information
   * \returns Information about the solve process
   */
  virtual const SolverStatus& getStatus() const = 0;

  /*!
   * \brief Option to add an extra-equation
   *
   * Option to add an extra-equation to the linear system such as a constraint equation
   *
   * \param[in] flag If the option is activated
   *
   * \todo Implement this method
   */
  virtual void setNullSpaceConstantOption(bool flag) = 0;

#ifdef USE_MULTI_SOLVER_INSTANCE
  /*!
   * \brief Creates a new linear solver instance
   * \returns A new linear solver instance
   * \todo Maybe returning a nullptr is not such a smart thing ?
   */
  virtual ILinearSolver* create() const { return NULL; }
#endif /* USE_MULTI_SOLVER_INSTANCE */
};

class ILinearSolverWithDiagScaling
{
 public:

  //! Constructor
  ILinearSolverWithDiagScaling() {}

  //! Free resources
  virtual ~ILinearSolverWithDiagScaling() {}

  virtual void setDiagScaling(const IMatrix& matrix) = 0 ;

} ;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
