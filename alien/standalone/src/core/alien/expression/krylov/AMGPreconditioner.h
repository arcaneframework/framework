// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

namespace Alien
{
  template<typename AlgebraT,
           typename MatrixT,
           typename VectorT,
           typename AMGSolverT>

  class AMGPreconditioner
  {
  public:
    using AlgebraType     = AlgebraT;
    using MatrixType      = MatrixT;
    using VectorType      = VectorT;
    using AMGSolverType   = AMGSolverT;


    AMGPreconditioner(AlgebraType& alg,
                      MatrixType const&  matrix,
                      AMGSolverType*     amg_solver,
                      ITraceMng*         trace_mng = nullptr
                     )
    : m_algebra(alg)
    , m_matrix(matrix)
    , m_amg_solver(amg_solver)
    , m_trace_mng(trace_mng)
    {
    }

    virtual ~AMGPreconditioner()
    {
      end() ;
    }

    void init()
    {
      if(m_amg_solver)
      {
        m_amg_solver->init() ;
        m_amg_solver->init(m_matrix) ;
        m_amg_solver->start() ;
      }
    }

    void end()
    {
      if(m_amg_solver)
        m_amg_solver->end() ;
    }

    void solve([[maybe_unused]] AlgebraType& alg,
               VectorType const& y,
               VectorType& x) const
    {
      // x input, y output
      // solve A.Y=X
      if(m_amg_solver)
      {
        // Solve A11.Y_Cpr = R_Cpr
        m_amg_solver->solve(y,x);
      }
    }

  private :
    AlgebraType&        m_algebra ;
    MatrixType const&   m_matrix ;

    AMGSolverType*      m_amg_solver   = nullptr;

    ITraceMng*          m_trace_mng    = nullptr ;
  };

}
