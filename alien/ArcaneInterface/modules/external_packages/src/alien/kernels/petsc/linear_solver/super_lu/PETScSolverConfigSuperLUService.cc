// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PETScSolverConfigSuperLUService.cc                    (C) 2000-2024       */
/*                                                                           */
/* SuperLU Solver from PETSc                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include <alien/kernels/petsc/linear_solver/super_lu/PETScSolverConfigSuperLUService.h>
#include <ALIEN/axl/PETScSolverConfigSuperLU_StrongOptions.h>

#include <arccore/message_passing/IMessagePassingMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Alien {
/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
PETScSolverConfigSuperLUService::PETScSolverConfigSuperLUService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScSolverConfigSuperLUObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  ;
}
#endif
PETScSolverConfigSuperLUService::PETScSolverConfigSuperLUService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScSolverConfigSuperLU> options)
: ArcanePETScSolverConfigSuperLUObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
}

//! Initialisation
void
PETScSolverConfigSuperLUService::configure(
    KSP& ksp, const ISpace& space, const MatrixDistribution& distribution)
{
  alien_debug([&] { cout() << "configure PETSc superlu solver"; });

  checkError(
      "Set solver tolerances", KSPSetTolerances(ksp, 1e-9, 1e-15, PETSC_DEFAULT, 2));

  checkError("Solver set type", KSPSetType(ksp, KSPPREONLY));
  PC pc;
  checkError("Get preconditioner", KSPGetPC(ksp, &pc));
  checkError("Preconditioner set type", PCSetType(pc, PCLU));

  //if (isParallel())
    checkError("Set superlu_dist solver package",
        PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST));
    //else
    //checkError(
    //"Set superlu solver package", PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU));

  KSPSetUp(ksp);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCSOLVERCONFIGSUPERLU(
    SuperLU, PETScSolverConfigSuperLUService);

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCSOLVERCONFIGSUPERLU();
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
