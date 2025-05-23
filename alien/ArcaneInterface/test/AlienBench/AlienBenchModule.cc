﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <mpi.h>

#include <string>
#include <map>
#include <time.h>
#include <vector>
#include <fstream>

#include <arcane/ArcaneVersion.h>
#include <arcane/Timer.h>
#include <arcane/ItemPairGroup.h>
#include <arcane/mesh/ItemFamily.h>
#include <arcane/utils/PlatformUtils.h>
#include <arcane/utils/IMemoryInfo.h>
#include <arcane/utils/OStringStream.h>
#include <arcane/ITimeLoopMng.h>



#include <alien/arcane_tools/accessors/ItemVectorAccessor.h>
#include <alien/core/block/VBlock.h>

#include <alien/arcane_tools/IIndexManager.h>
#include <alien/arcane_tools/indexManager/BasicIndexManager.h>
#include <alien/arcane_tools/indexManager/SimpleAbstractFamily.h>
#include <alien/arcane_tools/distribution/DistributionFabric.h>
#include <alien/arcane_tools/indexSet/IndexSetFabric.h>
#include <alien/arcane_tools/data/Space.h>

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>

#include <alien/ref/AlienRefSemantic.h>

#include <alien/kernels/redistributor/Redistributor.h>
#include <alien/ref/data/scalar/RedistributedVector.h>
#include <alien/ref/data/scalar/RedistributedMatrix.h>
#include <alien/ref/import_export/MatrixMarketSystemWriter.h>

#include <alien/expression/solver/SolverStater.h>
#include <alien/AlienLegacyConfig.h>

#ifdef ALIEN_USE_PETSC
#include <alien/kernels/petsc/io/AsciiDumper.h>
#include <alien/kernels/petsc/algebra/PETScLinearAlgebra.h>
#endif

#ifdef ALIEN_USE_MTL4
#include <alien/kernels/mtl/algebra/MTLLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_HTSSOLVER
#include <alien/kernels/hts/HTSBackEnd.h>
#include <alien/kernels/hts/data_structure/HTSMatrix.h>
#include <alien/kernels/hts/algebra/HTSLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_TRILINOS
#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
#include <alien/kernels/trilinos/algebra/TrilinosLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_HYPRE
#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>
#include <alien/kernels/hypre/algebra/HypreLinearAlgebra.h>
#endif

#ifdef ALIEN_USE_SYCL
#include <arccore/base/Span.h>
#include <alien/kernels/sycl/SYCLPrecomp.h>

#include "alien/kernels/sycl/data/SYCLEnv.h"

#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>
#include <alien/kernels/sycl/data/SYCLVectorInternal.h>
#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>

#include <alien/kernels/sycl/algebra/SYCLInternalLinearAlgebra.h>

#include "alien/kernels/sycl/data/SYCLSendRecvOp.h"
#include <alien/kernels/sycl/algebra/SYCLKernelInternal.h>
#endif

#include <alien/expression/solver/ILinearSolver.h>
#include "AlienCoreSolverOptionTypes.h"

#include "AlienBenchModule.h"

#include <arcane/ItemPairGroup.h>
#include <arcane/IMesh.h>

#include <alien/core/impl/MultiVectorImpl.h>

#include <alien/expression/krylov/AlienKrylov.h>
#include <alien/utils/StdTimer.h>

using namespace Arcane;
using namespace Alien;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
AlienBenchModule::init()
{
  info()<<"ALIEN BENCH INIT";
  Alien::setTraceMng(traceMng());
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);
  m_parallel_mng = subDomain()->parallelMng();
  m_homogeneous = options()->homogeneous();
  m_diag_coeff = options()->diagonalCoefficient();
  m_lambdax = options()->lambdax();
  m_lambday = options()->lambday();
  m_lambdaz = options()->lambdaz();
  m_alpha = options()->alpha();

  m_use_accelerator = options()->useAccelerator() ;
  m_with_usm = false ;
#ifdef ALIEN_USE_SYCL
  if(m_runner)
    if(m_runner->executionPolicy()==Accelerator::eExecutionPolicy::CUDA ||
       m_runner->executionPolicy()==Accelerator::eExecutionPolicy::HIP)
    {
        m_with_usm = options()->useUsm() ;
    }
#endif

  if(options()->linearSolver.size()>0)
  {
    Alien::ILinearSolver* solver = options()->linearSolver[0];
    solver->init();
  }

}

/*---------------------------------------------------------------------------*/
void
AlienBenchModule::test()
{
  info()<<"ALIEN BENCH TEST";

  Timer pbuild_timer(subDomain(), "PBuildPhase", Timer::TimerReal);
  Timer psolve_timer(subDomain(), "PSolvePhase", Timer::TimerReal);
  Timer rbuild_timer(subDomain(), "RBuildPhase", Timer::TimerReal);
  Timer rsolve_timer(subDomain(), "RSolvePhase", Timer::TimerReal);

  CellGroup areaU = allCells();
  CellCellGroup cell_cell_connection(areaU.own(), areaU, m_stencil_kind);
  CellCellGroup all_cell_cell_connection(areaU, areaU, m_stencil_kind);

  Alien::ArcaneTools::BasicIndexManager index_manager(m_parallel_mng);
  index_manager.setTraceMng(traceMng());

  auto indexSetU = index_manager.buildScalarIndexSet("U", areaU);
  index_manager.prepare();

  ///////////////////////////////////////////////////////////////////////////
  //
  // CREATE Space FROM IndexManger
  // CREATE MATRIX ASSOCIATED TO Space
  // CREATE VECTORS ASSOCIATED TO Space
  //

  Arccore::UniqueArray<Arccore::Integer> allUIndex = index_manager.getIndexes(indexSetU);

  Alien::ArcaneTools::Space space(&index_manager, "TestSpace");
  m_mdist = Alien::ArcaneTools::createMatrixDistribution(space);
  m_vdist = Alien::ArcaneTools::createVectorDistribution(space);

  info() << "GLOBAL SIZE : " << m_vdist.globalSize();

  std::stringstream description;
  int Nx = std::pow(m_vdist.globalSize()+1,1./3) ;
  description<<"Laplace problem "<<Nx<<"x"<<Nx<<"x"<<Nx;
  if(options()->homogeneous())
    description<<" HOMOGENEOUS";
  else
  {
    description<<" ALPHA="<<options()->alpha();
    description<<" LAMBDA (X,Y,Z)=("<<options()->lambdax()<<","<<options()->lambdax()<<","<<options()->lambdax()<<")";
    description<<" SIGMA="<<options()->sigma();
    description<<" EPSILON="<<options()->epsilon();
  }

  Alien::Vector vectorB(m_vdist);
  Alien::Vector vectorBB(m_vdist);
  Alien::Vector vectorX(m_vdist);

  Alien::Vector coordX(m_vdist);
  Alien::Vector coordY(m_vdist);
  Alien::Vector coordZ(m_vdist);

  Alien::Matrix matrixA(m_mdist); // local matrix for exact measure without side effect
                                  // (however, you can reuse a matrix with several
                                  // builder)

  info() << "USE ACCELERATOR "<<m_use_accelerator;
#ifdef ALIEN_USE_SYCL
  info() << "ALIEN USE SYCL";
  if(m_use_accelerator)
  {
    info() << "SYCL ASSEMBLY : usm="<<m_with_usm;
    if(m_with_usm)
      _testSYCLWithUSM(pbuild_timer,
                areaU,
                cell_cell_connection,
                all_cell_cell_connection,
                allUIndex,
                vectorB,
                vectorBB,
                vectorX,
                coordX,
                coordY,
                coordZ,
                matrixA) ;
    else
      _testSYCL(pbuild_timer,
                areaU,
                cell_cell_connection,
                all_cell_cell_connection,
                allUIndex,
                vectorB,
                vectorBB,
                vectorX,
                coordX,
                coordY,
                coordZ,
                matrixA) ;

  }
  else
#endif
  {
    info() << " CPU ASSEMPBLY ";

  _test(pbuild_timer,
        areaU,
        cell_cell_connection,
        all_cell_cell_connection,
        allUIndex,
        vectorB,
        vectorBB,
        vectorX,
        coordX,
        coordY,
        coordZ,
        matrixA) ;
  }


#ifdef ALIEN_USE_HTSSOLVER
/*{
  info()<<"HTS";
  Alien::HTSLinearAlgebra htsAlg;
  htsAlg.mult(matrixA,vectorX,vectorB);
  htsAlg.mult(matrixA,vectorX,vectorBB);
  Real normeb = htsAlg.norm2(vectorB) ;
  info()<<"||b||="<<normeb;
}*/
#endif
#ifdef ALIEN_USE_TRILINOS
  /*{
    info() << "Trilinos";
    Alien::TrilinosLinearAlgebra tpetraAlg;
    tpetraAlg.mult(matrixA, vectorX, vectorB);
    tpetraAlg.mult(matrixA, vectorX, vectorBB);
    Real normeb = tpetraAlg.norm2(vectorB);
    info() << "||b||=" << normeb;
    // tpetraAlg.dump(matrixA,"MatrixA.txt") ;
    // tpetraAlg.dump(vectorB,"vectorB.txt") ;
    // tpetraAlg.dump(vectorBB,"vectorBB.txt") ;
    // tpetraAlg.dump(vectorX,"vectorX.txt") ;
  }*/
#endif

  if (options()->unitRhs()) {
    Alien::LocalVectorWriter v(vectorBB);
    for (Integer i = 0; i < v.size(); ++i)
      v[i] = 1.;
  }

  if (options()->zeroRhs()) {
    Alien::LocalVectorWriter v(vectorB);
    for (Integer i = 0; i < v.size(); ++i)
      v[i] = 0.;
  }

  ///////////////////////////////////////////////////////////////////////////
  //
  // RESOLUTION
  //
  if(options()->alienCoreSolver.size()>0)
  {
#ifdef ALIEN_USE_SYCL
    if(m_use_accelerator)
    {
        Alien::SYCLInternalLinearAlgebra syclAlg;
        auto const& true_bb = vectorBB.impl()->get<Alien::SYCLInternalLinearAlgebra::BackEndType>() ;
        auto&       true_b = vectorB.impl()->get<Alien::SYCLInternalLinearAlgebra::BackEndType>(true) ;
        auto&       true_x = vectorX.impl()->get<Alien::SYCLInternalLinearAlgebra::BackEndType>(true) ;
        syclAlg.copy(true_bb,true_b) ;
        syclAlg.assign(true_x,0.) ;
    }
    else
#endif
    {
      Alien::LocalVectorReader reader(vectorBB);
      Alien::LocalVectorWriter vb(vectorB);
      Alien::LocalVectorWriter vx(vectorX);
      for (Integer i = 0; i < m_vdist.localSize(); ++i) {
        vx[i] = 0.;
        vb[i] = reader[i];
      }
    }
    typedef Alien::SimpleCSRInternalLinearAlgebra    AlgebraType ;
    typedef typename AlgebraType::BackEndType        BackEndType ;
    typedef Alien::Iteration<AlgebraType>            StopCriteriaType ;
    typedef Alien::CG<AlgebraType>                   SolverType ;

    AlgebraType alg ;

    auto const& opt   = options()->alienCoreSolver[0] ;
    int output_level  = opt->outputLevel() ;
    int max_iteration = opt->maxIter() ;
    double tol        = opt->tol() ;

    auto solver_opt   = opt->solver() ;
    auto precond_opt  = opt->preconditioner() ;

    auto backend      = opt->backend() ;
    auto asynch       = opt->asynch() ;

    // clang-format off
    auto run = [&](auto& alg)
              {
                typedef typename
                    std::remove_reference<decltype(alg)>::type AlgebraType ;
                typedef typename AlgebraType::BackEndType        BackEndType ;
                typedef Alien::Iteration<AlgebraType>            StopCriteriaType ;

                auto const& true_A = matrixA.impl()->get<BackEndType>() ;
                auto const& true_b = vectorB.impl()->get<BackEndType>() ;
                auto&       true_x = vectorX.impl()->get<BackEndType>(true) ;

                StopCriteriaType stop_criteria{alg,true_b,tol,max_iteration,output_level>0?traceMng():nullptr} ;

                switch(solver_opt)
                {
                  case AlienCoreSolverOptionTypes::CG:
                  {
                    typedef Alien::CG<AlgebraType> SolverType ;

                    SolverType solver{alg,traceMng()} ;
                    solver.setOutputLevel(output_level) ;
                    switch(precond_opt)
                    {
                      case AlienCoreSolverOptionTypes::ChebyshevPoly:
                      {
                          info()<<"CHEBYSHEV PRECONDITIONER";
                          double polynom_factor          = opt->polyFactor() ;
                          int    polynom_order           = opt->polyOrder() ;
                          int    polynom_factor_max_iter = opt->polyFactorMaxIter() ;

                          typedef Alien::ChebyshevPreconditioner<AlgebraType,true> PrecondType ;
                          PrecondType      precond{alg,true_A,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
                          precond.setOutputLevel(output_level) ;
                          precond.init() ;

                          Timer::Sentry ts(&psolve_timer);
                          if(asynch==0)
                            solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                          else
                            solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                      }
                      break;
                      case AlienCoreSolverOptionTypes::NeumannPoly:
                      {
                        info()<<"NEUMANN PRECONDITIONER";
                        double polynom_factor          = opt->polyFactor() ;
                        int    polynom_order           = opt->polyOrder() ;
                        int    polynom_factor_max_iter = opt->polyFactorMaxIter() ;

                        typedef Alien::NeumannPolyPreconditioner<AlgebraType> PrecondType ;
                        PrecondType precond{alg,true_A,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
                        precond.init() ;

                        Timer::Sentry ts(&psolve_timer);
                        if(asynch==0)
                          solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                        else
                          solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                      }
                      break ;
                      case AlienCoreSolverOptionTypes::Diag:
                      default:
                      {
                        info()<<"DIAG PRECONDITIONER";
                        typedef Alien::DiagPreconditioner<AlgebraType> PrecondType ;
                        PrecondType      precond{alg,true_A} ;
                        precond.init() ;

                        Timer::Sentry ts(&psolve_timer);
                        if(asynch==0)
                          solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                        else
                          solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                      }
                      break ;
                    }
                }
                break ;
                case AlienCoreSolverOptionTypes::BCGS:
                {
                  typedef Alien::BiCGStab<AlgebraType> SolverType ;
                  SolverType solver{alg,traceMng()} ;
                  solver.setOutputLevel(output_level) ;
                  switch(precond_opt)
                  {
                  case AlienCoreSolverOptionTypes::ChebyshevPoly:
                   {
                      info()<<"CHEBYSHEV PRECONDITIONER";
                      double polynom_factor          = opt->polyFactor() ;
                      int    polynom_order           = opt->polyOrder() ;
                      int    polynom_factor_max_iter = opt->polyFactorMaxIter() ;

                      typedef Alien::ChebyshevPreconditioner<AlgebraType,true> PrecondType ;
                      PrecondType      precond{alg,true_A,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
                      precond.setOutputLevel(output_level) ;
                      precond.init() ;

                      Timer::Sentry ts(&psolve_timer);
                      if(asynch==0)
                        solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                      else
                        solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                    }
                   break ;
                  case AlienCoreSolverOptionTypes::NeumannPoly:
                    {
                      info()<<"NEUMANN PRECONDITIONER";
                      double polynom_factor          = opt->polyFactor() ;
                      int    polynom_order           = opt->polyOrder() ;
                      int    polynom_factor_max_iter = opt->polyFactorMaxIter() ;

                      typedef Alien::NeumannPolyPreconditioner<AlgebraType> PrecondType ;
                      PrecondType precond{alg,true_A,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
                      precond.init() ;

                      Timer::Sentry ts(&psolve_timer);
                      if(asynch==0)
                        solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                      else
                        solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                    }
                    break ;
                  case AlienCoreSolverOptionTypes::ILU0:
                    {
                      info()<<"ILU0 PRECONDITIONER";
                      typedef Alien::ILU0Preconditioner<AlgebraType> PrecondType ;
                      PrecondType precond{alg,true_A,traceMng()} ;
                      precond.init() ;

                      Timer::Sentry ts(&psolve_timer);
                      if(asynch==0)
                        solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                      else
                        solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                    }
                    break ;
                  case AlienCoreSolverOptionTypes::FILU0:
                    {
                      info()<<"FILU0 PRECONDITIONER";
                      typedef Alien::FILU0Preconditioner<AlgebraType> PrecondType ;
                      PrecondType precond{alg,true_A,traceMng()} ;
                      precond.setParameter("nb-factor-iter",opt->filuFactorNiter()) ;
                      precond.setParameter("nb-solver-iter",opt->filuSolverNiter()) ;
                      precond.setParameter("tol",           opt->filuTol()) ;
                      precond.init() ;

                      Timer::Sentry ts(&psolve_timer);
                      if(asynch==0)
                        solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                      else
                        solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                    }
                    break ;
                  case AlienCoreSolverOptionTypes::Diag:
                  default:
                    {
                      info()<<"DIAG PRECONDITIONER";
                      typedef Alien::DiagPreconditioner<AlgebraType> PrecondType ;
                      PrecondType      precond{alg,true_A} ;
                      precond.init() ;

                      Timer::Sentry ts(&psolve_timer);
                      if(asynch==0)
                        solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                      else
                        solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                    }
                    break ;
                  }
                }
                break ;
                default :
                  fatal()<<"unknown solver";
                }

                if(stop_criteria.getStatus())
                {
                  info()<<"Solver has converged";
                  info()<<"Nb iterations  : "<<stop_criteria();
                  info()<<"Criteria value : "<<stop_criteria.getValue();
                }
                else
                {
                  info()<<"Solver convergence failed";
                }
              } ;

    auto run2 = [&](auto& alg)
              {
                typedef typename
                    std::remove_reference<decltype(alg)>::type AlgebraType ;
                typedef typename AlgebraType::BackEndType        BackEndType ;
                typedef Alien::Iteration<AlgebraType>            StopCriteriaType ;

                auto const& true_A = matrixA.impl()->get<BackEndType>() ;
                auto const& true_b = vectorB.impl()->get<BackEndType>() ;
                auto&       true_x = vectorX.impl()->get<BackEndType>(true) ;

                StopCriteriaType stop_criteria{alg,true_b,tol,max_iteration,output_level>0?traceMng():nullptr} ;

                switch(solver_opt)
                {
                  case AlienCoreSolverOptionTypes::CG:
                  {
                    typedef Alien::CG<AlgebraType> SolverType ;

                    SolverType solver{alg,traceMng()} ;
                    solver.setOutputLevel(output_level) ;
                    switch(precond_opt)
                    {
                      case AlienCoreSolverOptionTypes::ChebyshevPoly:
                      {
                          info()<<"CHEBYSHEV PRECONDITIONER";
                          double polynom_factor          = opt->polyFactor() ;
                          int    polynom_order           = opt->polyOrder() ;
                          int    polynom_factor_max_iter = opt->polyFactorMaxIter() ;

                          typedef Alien::ChebyshevPreconditioner<AlgebraType,true> PrecondType ;
                          PrecondType      precond{alg,true_A,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
                          precond.setOutputLevel(output_level) ;
                          precond.init() ;

                          Timer::Sentry ts(&psolve_timer);
                          if(asynch==0)
                            solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                          else
                            solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                      }
                      break;
                      case AlienCoreSolverOptionTypes::NeumannPoly:
                        {
                          info()<<"NEUMANN PRECONDITIONER";
                          double polynom_factor          = opt->polyFactor() ;
                          int    polynom_order           = opt->polyOrder() ;
                          int    polynom_factor_max_iter = opt->polyFactorMaxIter() ;

                          typedef Alien::NeumannPolyPreconditioner<AlgebraType> PrecondType ;
                          PrecondType precond{alg,true_A,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
                          precond.init() ;

                          Timer::Sentry ts(&psolve_timer);
                          if(asynch==0)
                            solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                          else
                            solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                        }
                        case AlienCoreSolverOptionTypes::Diag:
                        default:
                        {
                          info()<<"DIAG PRECONDITIONER";
                          typedef Alien::DiagPreconditioner<AlgebraType> PrecondType ;
                          PrecondType      precond{alg,true_A} ;
                          precond.init() ;

                          Timer::Sentry ts(&psolve_timer);
                          if(asynch==0)
                            solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                          else
                            solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                        }
                        break ;
                    }
                }
                break ;
                case AlienCoreSolverOptionTypes::BCGS:
                {
                  typedef Alien::BiCGStab<AlgebraType> SolverType ;
                  SolverType solver{alg,traceMng()} ;
                  solver.setOutputLevel(output_level) ;
                  switch(precond_opt)
                  {
                  case AlienCoreSolverOptionTypes::ChebyshevPoly:
                   {
                      info()<<"CHEBYSHEV PRECONDITIONER";
                      double polynom_factor          = opt->polyFactor() ;
                      int    polynom_order           = opt->polyOrder() ;
                      int    polynom_factor_max_iter = opt->polyFactorMaxIter() ;

                      typedef Alien::ChebyshevPreconditioner<AlgebraType,true> PrecondType ;
                      PrecondType      precond{alg,true_A,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
                      precond.setOutputLevel(output_level) ;
                      precond.init() ;

                      Timer::Sentry ts(&psolve_timer);
                      if(asynch==0)
                        solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                      else
                        solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                    }
                   break ;
                  case AlienCoreSolverOptionTypes::NeumannPoly:
                    {
                      info()<<"NEUMANN PRECONDITIONER";
                      double polynom_factor          = opt->polyFactor() ;
                      int    polynom_order           = opt->polyOrder() ;
                      int    polynom_factor_max_iter = opt->polyFactorMaxIter() ;

                      typedef Alien::NeumannPolyPreconditioner<AlgebraType> PrecondType ;
                      PrecondType precond{alg,true_A,polynom_factor,polynom_order,polynom_factor_max_iter,traceMng()} ;
                      precond.init() ;

                      Timer::Sentry ts(&psolve_timer);
                      if(asynch==0)
                        solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                      else
                        solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                    }
                    break ;
                  case AlienCoreSolverOptionTypes::Diag:
                  default:
                    {
                      info()<<"DIAG PRECONDITIONER";
                      typedef Alien::DiagPreconditioner<AlgebraType> PrecondType ;
                      PrecondType      precond{alg,true_A} ;
                      precond.init() ;

                      Timer::Sentry ts(&psolve_timer);
                      if(asynch==0)
                        solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
                      else
                        solver.solve2(precond,stop_criteria,true_A,true_b,true_x) ;
                    }
                    break ;
                  }
                }
                break ;
                default :
                  fatal()<<"unknown solver";
                }

                if(stop_criteria.getStatus())
                {
                  info()<<"Solver has converged";
                  info()<<"Nb iterations  : "<<stop_criteria();
                  info()<<"Criteria value : "<<stop_criteria.getValue();
                }
                else
                {
                  info()<<"Solver convergence failed";
                }
              } ;

    // clang-format on

    info()<<"START SOLVE A.X=B";
    switch(backend)
    {
      case AlienCoreSolverOptionTypes::SimpleCSR:
      {
        Alien::SimpleCSRInternalLinearAlgebra alg;
        run(alg);
      }
      break ;
      case AlienCoreSolverOptionTypes::SYCL:
      {
#ifdef ALIEN_USE_SYCL
        Alien::SYCLInternalLinearAlgebra alg;
        //alg.setDotAlgo(vm["dot-algo"].as<int>());
        run2(alg);
#else
        fatal() << "SYCL BackEnd not available";
#endif
      }
      break ;
      default:
        fatal() << "SYCL BackEnd not available";
        break ;
    }
  }
  else
  {
    Alien::ILinearSolver* solver = options()->linearSolver[0];
    solver->init();
    /*
    {
      auto const& true_A = matrixA.impl()->get<Alien::BackEnd::tag::hypre>() ;
      auto const& true_b = vectorB.impl()->get<Alien::BackEnd::tag::hypre>() ;
      auto const& true_x = vectorX.impl()->get<Alien::BackEnd::tag::hypre>() ;
      HypreLinearAlgebra alg;
      info()<<"HYPRE NORM B : "<<alg.norm2(vectorB);
      info()<<"HYPRE NORM X : "<<alg.norm2(vectorX);
    }*/
#ifdef ALIEN_USE_TRILINOS
    if(solver->getBackEndName().contains("tpetraserial"))
    {
      auto& mat = matrixA.impl()->get<Alien::BackEnd::tag::tpetraserial>(false) ;
      mat.setMatrixCoordinate(coordX,coordY,coordZ) ;
    }
#ifdef KOKKOS_ENABLE_OPENMP
    if(solver->getBackEndName().contains("tpetraomp"))
    {
      auto& mat = matrixA.impl()->get<Alien::BackEnd::tag::tpetraomp>(false) ;
      mat.setMatrixCoordinate(coordX,coordY,coordZ) ;
    }
#endif
#ifdef KOKKOS_ENABLE_THREADS
    if(solver->getBackEndName().contains("tpetrapth"))
    {
      auto& mat = matrixA.impl()->get<Alien::BackEnd::tag::tpetrapth>(false) ;
      mat.setMatrixCoordinate(coordX,coordY,coordZ) ;
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    if(solver->getBackEndName().contains("tpetracuda"))
    {
      auto& mat = matrixA.impl()->get<Alien::BackEnd::tag::tpetracuda>(true) ;
      mat->setCoordinate(coordX,coordY,coordZ) ;
    }
#endif
#endif
    

    if (not solver->hasParallelSupport() and m_parallel_mng->commSize() > 1) {
      info() << "Current solver has not a parallel support for solving linear system : "
                "skip it";
    } else {
      Integer nb_resolutions = options()->nbResolutions();
      for (Integer i = 0; i < nb_resolutions; ++i)
      {
        if (i > 0) // i=0, matrix allready filled
        {
#ifdef ALIEN_USE_SYCL
          if(m_use_accelerator)
          {
            if(m_with_usm)
              _fillSystemSYCLWithUSM(pbuild_timer,
                                     cell_cell_connection,
                                     all_cell_cell_connection,
                                     allUIndex,
                                     vectorB,
                                     vectorBB,
                                     vectorX,
                                     matrixA) ;
            else
              _fillSystemSYCL(pbuild_timer,
                             cell_cell_connection,
                             all_cell_cell_connection,
                             allUIndex,
                             vectorB,
                             vectorBB,
                             vectorX,
                             matrixA) ;
          }
          else
#endif
          {
            _fillSystemCPU(pbuild_timer,
                           cell_cell_connection,
                           all_cell_cell_connection,
                           allUIndex,
                           vectorB,
                           vectorBB,
                           vectorX,
                           matrixA) ;
          }
        }


        /*
        if(i==0)
        {
          Alien::MatrixMarketSystemWriter matrix_exporter("AlienBenchMatrixA.mtx",m_parallel_mng->messagePassingMng()) ;
          matrix_exporter.dump(matrixA,description.str()) ;
          Alien::MatrixMarketSystemWriter vector_exporter("AlienBenchVectorB.mtx",m_parallel_mng->messagePassingMng()) ;
          vector_exporter.dump(vectorB,description.str()) ;
        }*/

#ifdef ALIEN_USE_COMPOSYX
        if(solver->getBackEndName().contains("composyx"))
        {
          auto const& true_A = matrixA.impl()->get<Alien::BackEnd::tag::simplecsr>() ;
          auto& true_b = vectorB.impl()->get<Alien::BackEnd::tag::simplecsr>(false) ;
          auto& true_x = vectorX.impl()->get<Alien::BackEnd::tag::simplecsr>(true) ;
          SimpleCSRInternalLinearAlgebra alg;
          alg.synchronize(true_A,true_b) ;
        }
#endif
        Timer::Sentry ts(&psolve_timer);
        info()<<"START RESOLUTION";
        solver->solve(matrixA, vectorB, vectorX);
        info()<<"END RESOLUTION";
      }
      Alien::SolverStatus status = solver->getStatus();

      if (status.succeeded) {
        info()<<"RESOLUTION SUCCEED";
        
        Alien::VectorReader reader(vectorX);
        ENUMERATE_CELL (icell, areaU.own()) {
          const Integer iIndex = allUIndex[icell->localId()];
          m_x[icell] = reader[iIndex];
        }

#ifdef ALIEN_USE_SYCL
        if(m_use_accelerator)
        {
          SYCLLinearAlgebra alg;
          Alien::Vector vectorR(m_vdist);
          alg.mult(matrixA, vectorX, vectorR);
          alg.axpy(-1., vectorB, vectorR);
          Real res = alg.norm2(vectorR);
          info() << "RES : " << res;
        }
        else
#endif
        {
          SimpleCSRLinearAlgebra alg;
          Alien::Vector vectorR(m_vdist);
          alg.mult(matrixA, vectorX, vectorR);
          alg.axpy(-1., vectorB, vectorR);
          Real res = alg.norm2(vectorR);
          info() << "RES : " << res;
        }
      }
      else
        info()<<"SOLVER FAILED";
      solver->getSolverStat().print(Universe().traceMng(), status, "Linear Solver : ");
    }
    solver->end();
  }

  if (options()->redistribution() && m_parallel_mng->commSize() > 1) {
    info() << "Test REDISTRIBUTION";

    {
      Alien::LocalVectorWriter v(vectorX);
      for (Integer i = 0; i < v.size(); ++i)
        v[i] = 0;
    }
    Alien::Vector vectorR(m_vdist);

    bool keep_proc = false;
    if (m_parallel_mng->commRank() == 0)
      keep_proc = true;

    rbuild_timer.start();
    auto small_comm = Arccore::MessagePassing::mpSplit(m_parallel_mng->messagePassingMng(), keep_proc);
    Alien::Redistributor redist(matrixA.distribution().globalRowSize(),
                                m_parallel_mng->messagePassingMng(),
                                small_comm );

    Alien::RedistributedMatrix Aa(matrixA, redist);
    Alien::RedistributedVector bb(vectorB, redist);
    Alien::RedistributedVector xx(vectorX, redist);
    Alien::RedistributedVector rr(vectorR, redist);
    rbuild_timer.stop();
    if (keep_proc) {
      auto solver = options()->linearSolver[0];
      // solver->updateParallelMng(Aa.distribution().parallelMng());
      solver->init();
      {
        Timer::Sentry ts(&rsolve_timer);
        solver->solve(Aa, bb, xx);
      }

      Alien::SimpleCSRLinearAlgebra alg;

      alg.mult(Aa, xx, rr);
      alg.axpy(-1., bb, rr);
      Real res = alg.norm2(rr);
      info() << "REDISTRIBUTION RES : " << res;
    }
  }
  info() << "===================================================";
  info() << "BENCH INFO :";
  info() << " PBUILD    :" << pbuild_timer.totalTime();
  info() << " PSOLVE    :" << psolve_timer.totalTime();
  info() << " RBUILD    :" << rbuild_timer.totalTime();
  info() << " RSOLVE    :" << rsolve_timer.totalTime();
  info() << "===================================================";

  subDomain()->timeLoopMng()->stopComputeLoop(true);
}

void
AlienBenchModule::_test(Timer& pbuild_timer,
                        CellGroup& areaU,
                        CellCellGroup& cell_cell_connection,
                        CellCellGroup& all_cell_cell_connection,
                        Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                        Alien::Vector& vectorB,
                        Alien::Vector& vectorBB,
                        Alien::Vector& vectorX,
                        Alien::Vector& coordX,
                        Alien::Vector& coordY,
                        Alien::Vector& coordZ,
                        Alien::Matrix& matrixA)
{


  ///////////////////////////////////////////////////////////////////////////
  //
  // VECTOR BUILDING AND FILLING
  //
  info() << "Building & initializing vector b";
  info() << "Space size = " << m_vdist.globalSize()
         << ", local size= " << m_vdist.localSize();

  {
      ENUMERATE_CELL (icell, areaU)
      {
        Real3 x;
        for (Arcane::Node node : icell->nodes()) {
          x += m_node_coord[node];
        }
        x /= icell->nbNode();
        m_cell_center[icell] = x;
        m_u[icell] = funcn(x);
        m_k[icell] = funck(x);
      }
  }

  {
    // Builder du vecteur
    Alien::VectorWriter writer(vectorX);
    Alien::VectorWriter x(coordX);
    Alien::VectorWriter y(coordY);
    Alien::VectorWriter z(coordZ);

    ENUMERATE_CELL(icell,areaU.own())
    {
      const Integer iIndex = allUIndex[icell->localId()];
      writer[iIndex] = m_u[icell] ;
      x[iIndex] = m_cell_center[icell].x ;
      y[iIndex] = m_cell_center[icell].y ;
      z[iIndex] = m_cell_center[icell].z ;
    }
  }


  ///////////////////////////////////////////////////////////////////////////
  //
  // MATRIX BUILDING AND FILLING
  //

  {
    Timer::Sentry ts(&pbuild_timer);
    {
      Alien::MatrixProfiler profiler(matrixA);
      ///////////////////////////////////////////////////////////////////////////
      //
      // DEFINE PROFILE
      //
      ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
      {
        const Cell& cell = *icell;
        const Integer iIndex = allUIndex[cell.localId()];
        profiler.addMatrixEntry(iIndex, allUIndex[cell.localId()]);
        ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
        {
          const Cell& subcell = *isubcell;
          profiler.addMatrixEntry(iIndex, allUIndex[subcell.localId()]);
        }
      }
    }
    {
      Alien::ProfiledMatrixBuilder builder(
          matrixA, Alien::ProfiledMatrixOptions::eResetValues);
      ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
      {
        const Cell& cell = *icell;
        double diag = dii(cell);

        Integer i = allUIndex[cell.localId()];
        builder(i, i) += diag;
        ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
        {
          const Cell& subcell = *isubcell;
          double off_diag = fij(cell, subcell);
          builder(i, i) += off_diag;
          Integer j = allUIndex[subcell.localId()];
          builder(i, j) -= off_diag;
        }
      }
      if (options()->sigma() > 0.) {
        m_sigma = options()->sigma();
        auto xCmax = Real3{ 0.25, 0.25, 0.25 };
        auto xCmin = Real3{ 0.75, 0.75, 0.55 };
        ENUMERATE_ITEMPAIR(Cell, Cell, icell, all_cell_cell_connection)
        {
          const Cell& cell = *icell;
          Real3 xC = m_cell_center[icell];
          Real3 xDmax = xC - xCmax;
          Real3 xDmin = xC - xCmin;
          m_s[cell] = 0;
          if (xDmax.normL2() < options()->epsilon()) {
            m_s[cell] = 1.;
            Integer i = allUIndex[cell.localId()];
            info() << "MATRIX TRANSFO SIGMAMAX " << i;
            if (cell.isOwn())
              builder(i, i) = m_sigma;
            ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
            {
              const Cell& subcell = *isubcell;
              if (subcell.isOwn()) {
                Integer j = allUIndex[subcell.localId()];
                builder(j, i) = 0.;
              }
            }
          }
          if (xDmin.normL2() < options()->epsilon()) {
            m_s[cell] = -1.;
            Integer i = allUIndex[cell.localId()];
            info() << "MATRIX TRANSFO SIGMA MIN" << i;

            if (cell.isOwn())
              builder(i, i) = 1. / m_sigma;
            ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
            {
              const Cell& subcell = *isubcell;
              if (subcell.isOwn()) {
                Integer j = allUIndex[subcell.localId()];
                builder(j, i) = 0.;
              }
            }
          }
        }
      }
      builder.finalize();
    }
  }
  {
    Alien::SimpleCSRLinearAlgebra csrAlg;
    csrAlg.mult(matrixA, vectorX, vectorB);
    csrAlg.mult(matrixA, vectorX, vectorBB);
    Real normeb = csrAlg.norm2(vectorB);
    std::cout << "||b||=" << normeb<<std::endl;
  }
  {
    Alien::LocalVectorWriter vx(vectorX);
    vx = 0. ;
  }

}


void
AlienBenchModule::_fillSystemCPU( Timer& pbuild_timer,
                                  CellCellGroup& cell_cell_connection,
                                  CellCellGroup& all_cell_cell_connection,
                                  Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                                  Alien::Vector& vectorB,
                                  Alien::Vector& vectorBB,
                                  Alien::Vector& vectorX,
                                  Alien::Matrix& matrixA)
{
  {
    Timer::Sentry ts(&pbuild_timer);
    Alien::ProfiledMatrixBuilder builder(
        matrixA, Alien::ProfiledMatrixOptions::eResetValues);
    ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
    {
      const Cell& cell = *icell;
      double diag = dii(cell);

      Integer i = allUIndex[cell.localId()];
      builder(i, i) += diag;
      ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
      {
        const Cell& subcell = *isubcell;
        double off_diag = fij(cell, subcell);
        builder(i, i) += off_diag;
        Integer j = allUIndex[subcell.localId()];
        builder(i, j) -= off_diag;
      }
    }
    if (options()->sigma() > 0.)
    {
      m_sigma = options()->sigma();
      auto xCmax = Real3{ 0.25, 0.25, 0.25 };
      auto xCmin = Real3{ 0.75, 0.75, 0.55 };
      ENUMERATE_ITEMPAIR(Cell, Cell, icell, all_cell_cell_connection)
      {
        const Cell& cell = *icell;
        Real3 xC = m_cell_center[icell];
        Real3 xDmax = xC - xCmax;
        Real3 xDmin = xC - xCmin;
        m_s[cell] = 0;
        if (xDmax.normL2() < options()->epsilon()) {
          m_s[cell] = 1.;
          Integer i = allUIndex[cell.localId()];
          info() << "MATRIX TRANSFO SIGMAMAX " << i;
          if (cell.isOwn())
            builder(i, i) = m_sigma;
          ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
          {
            const Cell& subcell = *isubcell;
            if (subcell.isOwn()) {
              Integer j = allUIndex[subcell.localId()];
              builder(j, i) = 0.;
            }
          }
        }
        if (xDmin.normL2() < options()->epsilon()) {
          m_s[cell] = -1.;
          Integer i = allUIndex[cell.localId()];
          info() << "MATRIX TRANSFO SIGMA MIN" << i;

          if (cell.isOwn())
            builder(i, i) = 1. / m_sigma;
          ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
          {
            const Cell& subcell = *isubcell;
            if (subcell.isOwn()) {
              Integer j = allUIndex[subcell.localId()];
              builder(j, i) = 0.;
            }
          }
        }
      }
    }

    builder.finalize();
  }

  // Réinitialisation de vectorX
  {
    Alien::LocalVectorReader reader(vectorBB);
    Alien::LocalVectorWriter vb(vectorB);
    Alien::LocalVectorWriter vx(vectorX);
    for (Integer i = 0; i < m_vdist.localSize(); ++i) {
      vx[i] = 0.;
      vb[i] = reader[i];
    }
  }
}
/*---------------------------------------------------------------------------*/

ARCCORE_HOST_DEVICE Real
AlienBenchModule::funcn(Real3 p) const
{
  return p.x * p.x * p.y;
}


ARCCORE_HOST_DEVICE Real
AlienBenchModule::funck(Real3 p) const
{
#define PI 3.14159265358979323846264

  if (m_homogeneous)
    return m_off_diag_coeff;
  else
    return std::exp(-m_alpha * 0.5 * (1 + std::sin(2 * PI * p.x / m_lambdax))
        * (1 + std::sin(2 * PI * p.y / m_lambday)));
}


Real
AlienBenchModule::dii([[maybe_unused]] const Cell& ci) const
{
  return m_diag_coeff;
}

Real
AlienBenchModule::fij(const Cell& ci, const Cell& cj) const
{
  if (ci == cj)
    return dii(ci);
  else {
    Real3 xi = m_cell_center[ci];
    Real3 xj = m_cell_center[cj];
    Real3 xij = xi + xj;
    xij /= 2.;

    return funck(xij);
  }
}

ARCCORE_HOST_DEVICE Real
AlienBenchModule::fij(Integer vi, Integer vj, Arcane::Real3 xi, Arcane::Real3 xj) const
{
  if (vi == vj)
    return dii(vi);
  else {
    auto xij = xi + xj;
    xij /= 2.;

    return funck(xij);
  }
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ALIENBENCH(AlienBenchModule);
