// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

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

  template <class Tag>
  class KernelSolverT
  {
   public:
    //! The type of the solver
    using MatrixType  = typename AlgebraTraits<Tag>::matrix_type;
    using VectorType  = typename AlgebraTraits<Tag>::vector_type;
    using AlgebraType = typename AlgebraTraits<Tag>::algebra_type;

   public:
    virtual ~KernelSolverT() {}

    virtual void init() = 0;
    virtual void start() = 0 ;

    //! Initialize the linear solver
    virtual void init(MatrixType const& A) = 0;


    /*!
     * \brief Solve the linear system A * x = b
     * \param[in] A The matrix to invert
     * \param[in] b The right hand side
     * \param[in,out] x The solution
     * \returns Solver success or failure
     */
    virtual bool solve(const VectorType& b, VectorType& x) = 0 ;

   private:
  };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
