// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once


#include <alien/utils/Precomp.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/expression/solver/SolverStater.h>
#include <alien/core/backend/IInternalLinearSolverT.h>
#include <alien/utils/ObjectWithTrace.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

class IOptionsAlienCoreSolver;

#include <alien/kernels/common/AlienCoreSolverOptionTypes.h>
#include <ALIEN/axl/AlienCoreSolver_IOptions.h>

#include <alien/expression/solver/ILinearSolver.h>
#include "AlienCoreSolverOptionTypes.h"

#include <alien/expression/krylov/AlienKrylov.h>
#include <alien/utils/StdTimer.h>

#ifdef ALIEN_USE_HYPRE
#include <alien/AlienExternalPackages.h>
#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>
#include <alien/kernels/hypre/algebra/HypreLinearAlgebra.h>
#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/converters/SimpleCSR/SimpleCSR_to_Hypre_VectorConverter.h>
#include <alien/kernels/hypre/converters/SimpleCSR/Hypre_to_SimpleCSR_VectorConverter.h>
#endif

namespace Alien {

class SolverStat;

template<typename AlgebraT>
class AlienCoreSolverBaseT
    : public ILinearSolver,
      public ObjectWithTrace
{
 private:
  typedef SolverStatus Status;

 public:
  using AlgebraType = AlgebraT;
  using MatrixType  = typename AlgebraType::Matrix;
  using VectorType  = typename AlgebraType::Vector;
  using BackEndType = typename AlgebraType::BackEndType;

  /** Constructeur de la classe */
  AlienCoreSolverBaseT(Arccore::MessagePassing::IMessagePassingMng* parallel_mng = nullptr,
                       IOptionsAlienCoreSolver* options = nullptr)
  : m_parallel_mng(parallel_mng)
  , m_options(options)
  {}

  /** Destructeur de la classe */
  virtual ~AlienCoreSolverBaseT() {}

  virtual String getName() const = 0 ;

 public:
  //! Initialisation
  void init(int argv, char const** argc)
  {
    m_output_level  = m_options->outputLevel();
    m_max_iteration = m_options->maxIter();
    m_precision     = m_options->tol();
  }

  void init()
  {
      m_output_level  = m_options->outputLevel();
      m_max_iteration = m_options->maxIter();
      m_precision     = m_options->tol();
  }

  void updateParallelMng(Arccore::MessagePassing::IMessagePassingMng* pm)
  {
    m_parallel_mng = pm;
  }

  void updateParameters()
  {
    m_output_level  = m_options->outputLevel();
    m_max_iteration = m_options->maxIter();
    m_precision     = m_options->tol();
  }

  // void setDiagScal(double* Diag, int size);
  //! Finalize
  void end()
  {

  }


  bool solve(const MatrixType& matrixA, const VectorType& vectorB, VectorType& vectorX)
  {

    using StopCriteriaType = Alien::Iteration<AlgebraType> ;

    auto solver_opt   = m_options->solver() ;
    auto precond_opt  = m_options->preconditioner() ;

    auto backend      = m_options->backend() ;
    auto asynch       = m_options->asynch() ;

    m_timer.reset() ;
    AlgebraType alg;
    StopCriteriaType stop_criteria{alg,vectorB,m_precision,m_max_iteration,m_output_level>0?this->traceMng():nullptr} ;

    switch(solver_opt)
    {
      case AlienCoreSolverOptionTypes::CG:
      {
        typedef Alien::CG<AlgebraType> SolverType ;

        SolverType solver{alg,traceMng()} ;
        solver.setOutputLevel(m_output_level) ;
        switch(precond_opt)
        {
          case AlienCoreSolverOptionTypes::ChebyshevPoly:
          {
              this->traceMng()->info()<<"CHEBYSHEV PRECONDITIONER";
              double polynom_factor          = m_options->polyFactor() ;
              int    polynom_order           = m_options->polyOrder() ;
              int    polynom_factor_max_iter = m_options->polyFactorMaxIter() ;

              typedef Alien::ChebyshevPreconditioner<AlgebraType,true> PrecondType ;
              PrecondType      precond{alg,matrixA,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
              precond.setOutputLevel(m_output_level) ;
              {
                Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
                precond.init() ;
              }
              {
                Alien::StdTimer::Sentry ts(m_timer,"Solve");
                if(asynch==0)
                  solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
                else
                  solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
              }
          }
          break;
          case AlienCoreSolverOptionTypes::NeumannPoly:
          {
            this->traceMng()->info()<<"NEUMANN PRECONDITIONER";
            double polynom_factor          = m_options->polyFactor() ;
            int    polynom_order           = m_options->polyOrder() ;
            int    polynom_factor_max_iter = m_options->polyFactorMaxIter() ;

            typedef Alien::NeumannPolyPreconditioner<AlgebraType> PrecondType ;
            PrecondType precond{alg,matrixA,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
            {
              Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
              precond.init() ;
            }
            {
              Alien::StdTimer::Sentry ts(m_timer,"Solve");
              if(asynch==0)
                solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
              else
                solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
            }
          }
          break ;
          case AlienCoreSolverOptionTypes::AMG:
          {
            ILinearSolver* amg_solver = m_options->amgSolver() ;
            auto amg_backend_name = amg_solver->getBackEndName();
            this->traceMng()->info()<<"AMG PRECONDITIONER - BackEnd : "<<amg_backend_name;
            if(amg_backend_name=="hypre")
            {
#ifdef ALIEN_USE_HYPRE
              using AMGBackEndType  = Alien::BackEnd::tag::hypre ;
              using AMGSolverType   = Alien::KernelAMGSolverT<BackEndType,AlgebraType,AMGBackEndType> ;
              using PrecondType     = Alien::AMGPreconditioner<AlgebraType,MatrixType,VectorType,AMGSolverType>  ;

              auto proxy_amg_solver = AMGSolverType{alg,amg_solver} ;
              PrecondType precond{alg,matrixA,&proxy_amg_solver,traceMng()} ;
              {
                Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
                precond.init() ;
              }
              {
                Alien::StdTimer::Sentry ts(m_timer,"Solve");
                if(asynch==0)
                  solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
                else
                  solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
              }
#endif
            }
          }
          break ;
          case AlienCoreSolverOptionTypes::CxrAMG:
          {
            ILinearSolver* amg_solver = m_options->amgSolver() ;
            auto amg_backend_name = amg_solver->getBackEndName();
            this->traceMng()->info()<<"CxrAMG PRECONDITIONER - BackEnd : "<<amg_backend_name;
            if(amg_backend_name=="hypre")
            {
#ifdef ALIEN_USE_HYPRE
              using AMGBackendType   = Alien::BackEnd::tag::hypre ;
              using CxrOpType        = Alien::CxrOperator<MatrixType,VectorType> ;
              using CxrSolverType    = Alien::KernelAMGSolverT<BackEndType,AlgebraType,AMGBackendType> ;
              using RelaxSolverType  = Alien::ILU0Preconditioner<AlgebraType> ;
              using CxrPrecondType   = Alien::CxrPreconditioner<AlgebraType,
                                                                MatrixType,
                                                                VectorType,
                                                                CxrSolverType,
                                                                CxrOpType,
                                                                RelaxSolverType> ;

              auto cxr_op = CxrOpType(matrixA) ;
              cxr_op.computeCxrMatrix(alg) ;
              auto& cxr_matrix = cxr_op.getCxrMatrix() ;

              auto cxr_solver = CxrSolverType{alg,amg_solver} ;
              auto relax_solver = RelaxSolverType{alg,matrixA,traceMng()} ;

              auto precond = CxrPrecondType{alg,
                                            matrixA,
                                            &cxr_op,
                                            &cxr_solver,
                                            &relax_solver,
                                            traceMng()} ;
              {
                Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
                precond.init() ;
              }
              {
                Alien::StdTimer::Sentry ts(m_timer,"Solve");
                if(asynch==0)
                  solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
                else
                  solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
              }
#endif
            }
          }
          break ;
          case AlienCoreSolverOptionTypes::Diag:
          default:
          {
            this->traceMng()->info()<<"DIAG PRECONDITIONER";
            typedef Alien::DiagPreconditioner<AlgebraType> PrecondType ;
            PrecondType      precond{alg,matrixA} ;
            {
              Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
              precond.init() ;
            }
            {
              Alien::StdTimer::Sentry ts(m_timer,"Solve");
              if(asynch==0)
                solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
              else
                solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
            }
          }
          break ;
        }
    }
    break ;
    case AlienCoreSolverOptionTypes::BCGS:
    {
      typedef Alien::BiCGStab<AlgebraType> SolverType ;
      SolverType solver{alg,traceMng()} ;
      solver.setOutputLevel(m_output_level) ;
      switch(precond_opt)
      {
      case AlienCoreSolverOptionTypes::ChebyshevPoly:
       {
         this->traceMng()->info()<<"CHEBYSHEV PRECONDITIONER";
          double polynom_factor          = m_options->polyFactor() ;
          int    polynom_order           = m_options->polyOrder() ;
          int    polynom_factor_max_iter = m_options->polyFactorMaxIter() ;

          typedef Alien::ChebyshevPreconditioner<AlgebraType,true> PrecondType ;
          PrecondType      precond{alg,matrixA,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
          precond.setOutputLevel(m_output_level) ;
          {
            Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
            precond.init() ;
          }
          {
            Alien::StdTimer::Sentry ts(m_timer,"Solve");
            if(asynch==0)
              solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
            else
              solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
          }
        }
       break ;
      case AlienCoreSolverOptionTypes::NeumannPoly:
        {
          this->traceMng()->info()<<"NEUMANN PRECONDITIONER";
          double polynom_factor          = m_options->polyFactor() ;
          int    polynom_order           = m_options->polyOrder() ;
          int    polynom_factor_max_iter = m_options->polyFactorMaxIter() ;

          typedef Alien::NeumannPolyPreconditioner<AlgebraType> PrecondType ;
          PrecondType precond{alg,matrixA,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
          {
            Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
            precond.init() ;
          }
          {
            Alien::StdTimer::Sentry ts(m_timer,"Solve");
            if(asynch==0)
              solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
            else
              solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
          }
        }
        break ;
      case AlienCoreSolverOptionTypes::ILU0:
        {
          this->traceMng()->info()<<"ILU0 PRECONDITIONER";
          typedef Alien::ILU0Preconditioner<AlgebraType> PrecondType ;
          PrecondType precond{alg,matrixA,traceMng()} ;
          {
            Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
            precond.init() ;
          }

          Alien::StdTimer::Sentry ts(m_timer,"Solve");
          if(asynch==0)
            solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
          else
            solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
        }
        break ;
      case AlienCoreSolverOptionTypes::FILU0:
        {
          this->traceMng()->info()<<"FILU0 PRECONDITIONER";
          typedef Alien::FILU0Preconditioner<AlgebraType> PrecondType ;
          PrecondType precond{alg,matrixA,traceMng()} ;
          precond.setParameter("nb-factor-iter",m_options->filuFactorNiter()) ;
          precond.setParameter("nb-solver-iter",m_options->filuSolverNiter()) ;
          precond.setParameter("tol",           m_options->filuTol()) ;
          {
            Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
            precond.init() ;
          }
          {
            Alien::StdTimer::Sentry ts(m_timer,"Solve");
            if(asynch==0)
              solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
            else
              solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
          }
        }
        break ;
      case AlienCoreSolverOptionTypes::Diag:
      default:
        {
          this->traceMng()->info()<<"DIAG PRECONDITIONER";
          typedef Alien::DiagPreconditioner<AlgebraType> PrecondType ;
          PrecondType      precond{alg,matrixA} ;
          {
            Alien::StdTimer::Sentry ts(m_timer,"PrecSetUp");
            precond.init() ;
          }
          {
            Alien::StdTimer::Sentry ts(m_timer,"Solve");
            if(asynch==0)
              solver.solve(precond,stop_criteria,matrixA,vectorB,vectorX) ;
            else
              solver.solve2(precond,stop_criteria,matrixA,vectorB,vectorX) ;
          }
        }
        break ;
      }
    }
    break ;
    default :
      this->traceMng()->fatal()<<"unknown solver";
    }

    if(stop_criteria.getStatus())
    {

      ////////////////////////////////////////////////////
      //
      // ANALIZE STATUS
      m_status.residual = stop_criteria.getValue();
      m_status.iteration_count = stop_criteria();
      m_status.succeeded = true;
      m_status.error = 0;
      m_total_iter_num += m_status.iteration_count ;
      ++m_solve_num ;
      m_total_solve_time += m_timer("Solve") ;
      m_total_prec_setup_time += m_timer("PrecSetUp") ;
      if (m_output_level > 0) {
        alien_info([&] {
          cout() << "Resolution info      :";
          cout() << "Resolution status      : OK";
          cout() << "Residual             : " << m_status.residual;
          cout() << "Number of iterations : " << m_status.iteration_count;
        });
        m_timer.printInfo(traceMng()->info().file(),"\nAlienCoreSolver Perf INFO :") ;
      }
      return true;
    }
    else
    {
      this->traceMng()->info()<<"Solver convergence failed";
      m_status.succeeded = false;
      m_status.error = 1;
      m_total_iter_num += m_status.iteration_count ;
      ++m_solve_num ;
      m_total_solve_time += m_timer("Solve") ;
      m_total_prec_setup_time += m_timer("PrecSetUp") ;
      if (m_output_level > 0) {
        alien_info([&] {
          cout() << "Resolution status      : Error";
          cout() << "Error code             : " << m_status.error;
        });
        m_timer.printInfo(traceMng()->info().file(),"AlienCoreSolver Perf INFO :") ;
      }
      return false;
    }
  }

  //! Indicateur de support de résolution parallèle
  bool hasParallelSupport() const { return true; }

  //! Etat du solveur
  void setNullSpaceConstantOption(bool flag)
  {
    alien_warning([&] { cout() << "Null Space Constant Option not yet implemented"; });
  }

  void printInfo() const
  {
    alien_info([&] {
      cout();
      cout() << "|--------------------------------------------|";
      cout() << "| Linear Solver        : " << this->getName();
      cout() << "|--------------------------------------------|";
      cout() << "| total solver time    : " << m_total_solve_time;
      cout() << "| total prec setup time : " << m_total_prec_setup_time;
      cout() << "| total num of iter    : " << m_total_iter_num;
      cout() << "| solve num            : " << m_solve_num;
      cout() << "|---------------------------------------------|";
      cout();
    });
  }

  void printInfo()
  {
    alien_info([&] {
      cout();
      cout() << "|--------------------------------------------|";
      cout() << "| Linear Solver        : " << this->getName();
      cout() << "|--------------------------------------------|";
      cout() << "| total solver time     : " << m_total_solve_time;
      cout() << "| total prec setup time : " << m_total_prec_setup_time;
      cout() << "| total num of iter    : " << m_total_iter_num;
      cout() << "| solve num            : " << m_solve_num;
      cout() << "|---------------------------------------------|";
      cout();
    });
    if(m_output_level>0)
      m_timer.printInfo(traceMng()->info().file(),"AlienCoreSolver Perf INFO :") ;
  }
  void printCurrentTimeInfo() {}


 private:
  void updateLinearSystem();
  inline void _startPerfCount()
  {

  }

  inline void _endPerfCount()
  {
    m_solve_num++;
    m_total_iter_num += m_status.iteration_count;
  }

 protected:
  //! Structure interne du solveur

  bool m_use_mpi = false;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  Alien::SolverStatus m_status;
  Alien::StdTimer     m_timer;

  int m_current_ctx_id = -1;

  //! Solver parameters
  Integer m_max_iteration = 0;
  Real m_precision        = 0.;
  Integer m_output_level  = 0;



  Integer m_solve_num = 0;
  Integer m_total_iter_num = 0;
  Real m_current_solve_time = 0.;
  Real m_total_solve_time = 0.;
  Real m_current_prec_setup_time = 0.;
  Real m_total_prec_setup_time = 0.;

  IOptionsAlienCoreSolver* m_options = nullptr;
  std::vector<double> m_pressure_diag;
};

} // namespace Alien

