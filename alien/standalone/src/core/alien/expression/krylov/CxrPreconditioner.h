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
           typename CxrSolverT,
           typename CxrOpT,
           typename RelaxSolverT>

  class CxrPreconditioner
  {
  public:
    using AlgebraType     = AlgebraT;
    using MatrixType      = MatrixT;
    using VectorType      = VectorT;
    using RelaxSolverType = RelaxSolverT;
    using CxrSolverType   = CxrSolverT;
    using CxrOpType       = CxrOpT;


    CxrPreconditioner(AlgebraType& alg,
                      MatrixType const&  matrix,
                      CxrOpType*         cxr_op,
                      CxrSolverType*     cxr_solver,
                      RelaxSolverType*   relax_solver,
                      ITraceMng*         trace_mng = nullptr
                     )
    : m_algebra(alg)
    , m_matrix(matrix)
    , m_relax_solver(relax_solver)
    , m_cxr_solver(cxr_solver)
    , m_cxr_op(cxr_op)
    , m_trace_mng(trace_mng)
    {
    }

    virtual ~CxrPreconditioner() {}

     void init()
    {
      auto& cxr_matrix = m_cxr_op->getCxrMatrix() ;
      if(m_relax_solver)
        m_relax_solver->init() ;
      if(m_cxr_solver)
      {
        m_cxr_solver->init() ;
        m_cxr_solver->init(cxr_matrix) ;
        m_cxr_solver->start() ;
      }
      m_algebra.allocate(AlgebraType::resource(cxr_matrix),m_x_Cxr,m_y_Cxr,m_r_Cxr);
    }

    void solve(AlgebraType& alg,
               VectorType const& y,
               VectorType& x) const
    {
      // x input, y output
      /*
          | A11  A12 |    | X1 |    |Y1|
       A= | A12  A22 |  X=| X2 | Y= |Y2|
       */

      // solve A.Y=X
      if(m_relax_solver)
      {
        m_relax_solver->solve(y,x);

        // X_Cpr = A11.Y1 + A12.Y2
        m_cxr_op->apply(alg,x,m_x_Cxr);

        m_cxr_op->get(alg,y,m_r_Cxr);

        // R_Cpr = X1-X_Cpr
        //m_kernel->xmy(m_r_Cxr,m_x_Cxr);
        alg.scal(-1.,m_x_Cxr) ;
        alg.axpy(1.,m_r_Cxr,m_x_Cxr) ;
      }
      else
      {
        alg.copy(y,x) ;
        m_cxr_op->get(alg,x,m_x_Cxr);
      }


      if(m_cxr_solver)
      {
        // Solve A11.Y_Cpr = R_Cpr

        alg.copy(m_x_Cxr,m_y_Cxr) ;
        m_cxr_solver->solve(m_x_Cxr,m_y_Cxr);

        // combine m_y_Cpr with y
        if(m_relax_solver)
          m_cxr_op->combine(alg,m_y_Cxr,x);
        else
          m_cxr_op->copy(alg,m_y_Cxr,x);
      }
    }

  private :
    AlgebraType&        m_algebra ;
    MatrixType const&   m_matrix ;

    mutable VectorType  m_x_Cxr;

    mutable VectorType  m_y;
    mutable VectorType  m_y_Cxr;

    mutable VectorType  m_r;
    mutable VectorType  m_r_Cxr;


    RelaxSolverType*    m_relax_solver = nullptr;
    CxrSolverType*      m_cxr_solver   = nullptr;
    CxrOpType*          m_cxr_op       = nullptr;

    ITraceMng*          m_trace_mng    = nullptr ;
  };

}
