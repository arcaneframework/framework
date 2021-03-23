// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephKappa.cc                                                     (C) 2012 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "arcane_packages.h"

#include "arcane/IParallelMng.h"
#include "arcane/IDirectExecution.h"
#include "arcane/BasicService.h"
#include "arcane/ArcaneVersion.h"
#include "arcane/FactoryService.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/MultiArray2.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/Timer.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/aleph/AlephTypesSolver.h"
#include "arcane/aleph/AlephParams.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IProcessorAffinityService.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/IApplication.h"

#include "arcane/aleph/Aleph.h"
#include "arcane/aleph/IAlephFactory.h"

#include "arcane/aleph/kappa/AlephKappa.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(AlephKappaService, IDirectExecution, AlephKappa);

AlephKappaService::
AlephKappaService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
, m_kernel(NULL)
, m_world_parallel(NULL)
, m_world_rank(-1)
, m_size(-1)
, m_world_size(-1)
, m_factory(NULL)
, m_underlying_solver(-1)
, m_solver_size(-1)
, m_reorder(false)
{
  debug() << "[AlephKappaService] NEW";
  traceMng()->flush();
  m_application = sbi.application();
}

AlephKappaService::
~AlephKappaService()
{
  if (m_kernel)
    delete m_kernel;
  if (m_factory)
    delete m_factory;
}

void AlephKappaService::
execute(void)
{
  debug() << "[AlephKappaService] Retrieving world size...";
  m_world_size = m_world_parallel->commSize();
  m_world_rank = m_world_parallel->commRank();

  debug() << "[AlephKappaService] I should be an additional site #"
          << m_world_rank << " among " << m_world_size;

  debug() << "[AlephKappaService] Retrieving configuration...";
  // cfg(0): m_underlying_solver
  // cfg(1): m_solver_size
  // cfg(2): m_reorder
  // cfg(3): m_size
  UniqueArray<Integer> cfg(4);
  m_world_parallel->broadcast(cfg, 0);
  for (Integer rnk = 0, max = cfg.size(); rnk < max; rnk += 1) {
    debug() << "[AlephKappaService] cfg[" << rnk << "]=" << cfg[rnk];
  }

  debug() << "[AlephKappaService] factory";
  m_factory = new AlephFactory(m_application, m_world_parallel->traceMng());
  debug() << "[AlephKappaService] kernel";
  m_kernel = new AlephKernel(m_world_parallel,
                             m_size = cfg.at(3),
                             m_factory,
                             m_underlying_solver = cfg.at(0),
                             m_solver_size = cfg.at(1),
                             m_reorder = (cfg.at(2) == 1) ? true : false);

  AlephParams* params =
  new AlephParams(traceMng(),
                  1.0e-10, // m_param_epsilon epsilon de convergence
                  2000, // m_param_max_iteration nb max iterations
                  TypesSolver::DIAGONAL, // m_param_preconditioner_method: DIAGONAL, AMG, IC
                  TypesSolver::PCG, // m_param_solver_method méthode de résolution
                  -1, // m_param_gamma
                  -1.0, // m_param_alpha
                  false, // m_param_xo_user par défaut Xo n'est pas égal à 0
                  false, // m_param_check_real_residue
                  false, // m_param_print_real_residue
                  false, // m_param_debug_info
                  1.e-40, // m_param_min_rhs_norm
                  false, // m_param_convergence_analyse
                  true, // m_param_stop_error_strategy
                  false, // m_param_write_matrix_to_file_error_strategy
                  "SolveErrorAlephMatrix.dbg", // m_param_write_matrix_name_error_strategy
                  false, // m_param_listing_output
                  0., // m_param_threshold
                  false, // m_param_print_cpu_time_resolution
                  0, // m_param_amg_coarsening_method
                  0, // m_param_output_level
                  1, // m_param_amg_cycle
                  1, // m_param_amg_solver_iterations
                  1, // m_param_amg_smoother_iterations
                  TypesSolver::SymHybGSJ_smoother, // m_param_amg_smootherOption
                  TypesSolver::ParallelRugeStuben, // m_param_amg_coarseningOption
                  TypesSolver::CG_coarse_solver, // m_param_amg_coarseSolverOption
                  false, // m_param_keep_solver_structure
                  false, // m_param_sequential_solver
                  TypesSolver::RB); // m_param_criteria_stop

  UniqueArray<AlephMatrix*> A_matrix_queue;
  Integer aleph_vector_idx = 0;
  UniqueArray<AlephVector*> b;
  UniqueArray<AlephVector*> x;
  // Ce flag permet d'éviter de prendre en compte le create() du vecteur temporaire des arguments du solveur
  bool firstVectorCreateForTmp = true;

  traceMng()->flush();

  while (true) {
    UniqueArray<unsigned long> token(1);
    debug() << "[AlephKappaService] listening for a token...";
    traceMng()->flush();
    m_world_parallel->broadcast(token.view(), 0);
    traceMng()->flush();
    debug() << "[AlephKappaService] found token " << token.at(0);
    traceMng()->flush();

    switch (token.at(0)) {
      /************************************************************************
       * AlephKernel::initialize
       ************************************************************************/
    case (0xd80dee82l): {
      debug() << "[AlephKappaService] AlephKernel::initialize!";
      UniqueArray<Integer> args(2);
      // Récupération des global_nb_row et local_nb_row
      m_world_parallel->broadcast(args.view(), 0);
      m_kernel->initialize((Integer)args.at(0), (Integer)args.at(1));
      break;
    }

      /************************************************************************
       * AlephKernel::createSolverMatrix
       ************************************************************************/
    case (0xef162166l):
      debug() << "[AlephKappaService] AlephKernel::createSolverMatrix (new A[" << A_matrix_queue.size() << "])!";
      firstVectorCreateForTmp = true; // On indique que le prochain create() est relatif au vecteur tmp
      A_matrix_queue.add(m_kernel->createSolverMatrix());
      break;

      /************************************************************************
       * AlephKernel::postSolver
       ************************************************************************/
    case (0xba9488bel): {
      debug() << "[AlephKappaService] AlephKernel::postSolver!";
      UniqueArray<Real> real_args(4);
      m_world_parallel->broadcast(real_args.view(), 0);
      params->setEpsilon(real_args.at(0));
      params->setAlpha(real_args.at(1));
      params->setMinRHSNorm(real_args.at(2));
      params->setDDMCParameterAmgDiagonalThreshold(real_args.at(3));

      UniqueArray<int> bool_args(11);
      m_world_parallel->broadcast(bool_args.view(), 0);
      params->setXoUser((bool)bool_args.at(0));
      params->setCheckRealResidue((bool)bool_args.at(1));
      params->setPrintRealResidue((bool)bool_args.at(2));
      params->setDebugInfo((bool)bool_args.at(3));
      params->setConvergenceAnalyse((bool)bool_args.at(4));
      params->setStopErrorStrategy((bool)bool_args.at(5));
      params->setWriteMatrixToFileErrorStrategy((bool)bool_args.at(6));
      params->setDDMCParameterListingOutput((bool)bool_args.at(7));
      params->setPrintCpuTimeResolution((bool)bool_args.at(8));
      params->setKeepSolverStructure((bool)bool_args.at(9));
      params->setSequentialSolver((bool)bool_args.at(10));

      UniqueArray<Integer> int_args(13);
      m_world_parallel->broadcast(int_args.view(), 0);
      params->setMaxIter(int_args.at(0));
      params->setGamma(int_args.at(1));
      params->setPrecond((TypesSolver::ePreconditionerMethod)int_args.at(2));
      params->setMethod((TypesSolver::eSolverMethod)int_args.at(3));
      params->setAmgCoarseningMethod((TypesSolver::eAmgCoarseningMethod)int_args.at(4));
      params->setOutputLevel(int_args.at(5));
      params->setAmgCycle(int_args.at(6));
      params->setAmgSolverIter(int_args.at(7));
      params->setAmgSmootherIter(int_args.at(8));
      params->setAmgSmootherOption((TypesSolver::eAmgSmootherOption)int_args.at(9));
      params->setAmgCoarseningOption((TypesSolver::eAmgCoarseningOption)int_args.at(10));
      params->setAmgCoarseSolverOption((TypesSolver::eAmgCoarseSolverOption)int_args.at(11));
      params->setCriteriaStop((TypesSolver::eCriteriaStop)int_args.at(12));

      m_kernel->postSolver(params, NULL, NULL, NULL);
      break;
    }

      /************************************************************************
       * AlephKernel::createSolverVector
       ************************************************************************/
    case (0xc4b28f2l): {
      if ((aleph_vector_idx % 2) == 0) {
        debug() << "[AlephKappaService] AlephKernel::createSolverVector (new b[" << b.size() << "])";
        b.add(m_kernel->createSolverVector());
      }
      else {
        debug() << "[AlephKappaService] AlephKernel::createSolverVector (new x[" << x.size() << "])";
        x.add(m_kernel->createSolverVector());
      }
      aleph_vector_idx += 1;
      break;
    }

      /************************************************************************
       * AlephMatrix::create(void)
       ************************************************************************/
    case (0xfff06e2l): {
      debug() << "[AlephKappaService] AlephMatrix::create(void)!";
      A_matrix_queue.at(m_kernel->index())->create();
      break;
    }

      /************************************************************************
       * AlephVector::create
       ************************************************************************/
    case (0x6bdba30al): {
      if (firstVectorCreateForTmp) { // Si c'est pour le vecteur tmp, on skip
        debug() << "[AlephKappaService] firstVectorCreateForTmp";
        firstVectorCreateForTmp = false; // Et on annonce que c'est fait pour le tmp
        break;
      }

      if ((aleph_vector_idx % 2) == 0) {
        debug() << "[AlephKappaService] AlephVector::create (b[" << m_kernel->index() << "])";
        b.at(m_kernel->index())->create();
      }
      else {
        debug() << "[AlephKappaService] AlephVector::create (x[" << m_kernel->index() << "])";
        x.at(m_kernel->index())->create();
      }
      aleph_vector_idx += 1;
      break;
    }

      /************************************************************************
       * AlephMatrix::assemble
       ************************************************************************/
    case (0x74f253cal): {
      debug() << "[AlephKappaService] AlephMatrix::assemble! (kernel->index=" << m_kernel->index() << ")";
      UniqueArray<Integer> setValue_idx(1);
      m_world_parallel->broadcast(setValue_idx.view(), 0);
      // On le fait avant pour seter le flag à true
      m_kernel->topology()->create(setValue_idx.at(0));
      A_matrix_queue.at(m_kernel->index())->assemble();
      break;
    }

      /************************************************************************
       * AlephVector::assemble
       ************************************************************************/
    case (0xec7a979fl): {
      if ((aleph_vector_idx % 2) == 0) {
        debug() << "[AlephKappaService] AlephVector::assemble! (b" << m_kernel->index() << ")";
        b.at(m_kernel->index())->assemble();
      }
      else {
        debug() << "[AlephKappaService] AlephVector::assemble! (x" << m_kernel->index() << ")";
        x.at(m_kernel->index())->assemble();
      }
      aleph_vector_idx += 1;
      break;
    }

      /************************************************************************
       * AlephKernel::syncSolver
       ************************************************************************/
    case (0xbf8d3adfl): {
      debug() << "[AlephKappaService] AlephKernel::syncSolver";
      traceMng()->flush();
      UniqueArray<Integer> gid(1);
      m_world_parallel->broadcast(gid.view(), 0);

      Integer nb_iteration;
      Real residual_norm[4];
      debug() << "[AlephKappaService] AlephKernel::syncSolver group id=" << gid.at(0);
      traceMng()->flush();
      m_kernel->syncSolver(gid.at(0), nb_iteration, &residual_norm[0]);
      break;
    }

      /************************************************************************
       * ArcaneMainBatch::SessionExec::executeRank
       ************************************************************************/
    case (0xdfeb699fl): {
      debug() << "[AlephKappaService] AlephKernel::finalize!";
      traceMng()->flush();
      delete params;
      return;
    }

      /************************************************************************
       * Should never happen
       ************************************************************************/
    default:
      debug() << "[AlephKappaService] default";
      traceMng()->flush();
      throw FatalErrorException("execute", "Unknown token for handshake");
    }
    traceMng()->flush();
  }
  throw FatalErrorException("execute", "Should never be there!");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
