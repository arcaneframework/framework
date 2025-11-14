// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/kernels/petsc/linear_solver/super_lu/PETScPrecConfigSuperLUService.h>
#include <ALIEN/axl/PETScPrecConfigSuperLU_StrongOptions.h>

#include <arccore/message_passing/IMessagePassingMng.h>

/* Pour debugger le ILU, utiliser l'option:
 * <cmd-line-param>-mat_superlu_printstat 1 </cmd-line-param>
 */

/*---------------------------------------------------------------------------*/
#if ((PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 3) || (PETSC_VERSION_MAJOR > 3))
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
PETScPrecConfigSuperLUService::PETScPrecConfigSuperLUService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScPrecConfigSuperLUObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  ;
}
#endif

PETScPrecConfigSuperLUService::PETScPrecConfigSuperLUService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScPrecConfigSuperLU> options)
: ArcanePETScPrecConfigSuperLUObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
  ;
}

//! Initialisation
void
PETScPrecConfigSuperLUService::configure(
    PC& pc, const ISpace& space, const MatrixDistribution& distribution)
{
  alien_debug([&] { cout() << "configure PETSc superlu preconditioner"; });
  checkError("Set preconditioner", PCSetType(pc, PCLU));

  //  if (m_is_parallel)
    checkError("Set superlu_dist solver package",
        PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST));
    //else
    //checkError(
    //    "Set superlu solver package", PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU));

  // checkError("Set fill factor",  PCFactorSetUpMatSolverPackage(pc));
  double fill_factor = options()->fillFactor();
  if (fill_factor < 1.0) {
    alien_fatal([&] { cout() << "Bad Fill Factor: cannot be less than 1.0"; });
  }
  checkError("Set fill factor", PCFactorSetFill(pc, fill_factor));
  // checkError("Set shift type",PCFactorSetShiftType(pc,PETSC_DECIDE));
  checkError("Set shift amount", PCFactorSetShiftAmount(pc, PETSC_DECIDE));

  checkError("Preconditioner setup", PCSetUp(pc));
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCPRECCONFIGSUPERLU(SuperLU, PETScPrecConfigSuperLUService);

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCPRECCONFIGSUPERLU();
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif /* PETSC_VERSION */
/*---------------------------------------------------------------------------*/
