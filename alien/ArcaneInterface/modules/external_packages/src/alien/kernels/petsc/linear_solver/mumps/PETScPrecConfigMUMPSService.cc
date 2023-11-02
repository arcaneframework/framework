// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#include <alien/kernels/petsc/linear_solver/mumps/PETScPrecConfigMUMPSService.h>
#include <ALIEN/axl/PETScPrecConfigMUMPS_StrongOptions.h>

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
PETScPrecConfigMUMPSService::PETScPrecConfigMUMPSService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScPrecConfigMUMPSObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  ;
}
#endif

PETScPrecConfigMUMPSService::PETScPrecConfigMUMPSService(
        Arccore::MessagePassing::IMessagePassingMng* parallel_mng, std::shared_ptr<IOptionsPETScPrecConfigMUMPS> options)
: ArcanePETScPrecConfigMUMPSObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
  ;
}

//! Initialisation
void
PETScPrecConfigMUMPSService::configure(
    PC& pc, const ISpace& space, const MatrixDistribution& distribution)
{
  alien_debug([&] { cout() << "configure PETSc mumps preconditioner"; });
  checkError("Set preconditioner", PCSetType(pc, PCLU));

  checkError("Set mumps solver package", PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS));

  // checkError("Set fill factor",  PCFactorSetUpMatSolverPackage(pc));
  double fill_factor = options()->fillFactor();
  if (fill_factor < 1.0)
    PETScConfig::traceMng()->fatal() << "Bad Fill Factor: cannot be less than 1.0";

  checkError("Set fill factor", PCFactorSetFill(pc, fill_factor));
  // checkError("Set shift type",PCFactorSetShiftType(pc,PETSC_DECIDE));
  checkError("Set shift amount", PCFactorSetShiftAmount(pc, PETSC_DECIDE));

  checkError("Preconditioner setup", PCSetUp(pc));
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCPRECCONFIGMUMPS(MUMPS, PETScPrecConfigMUMPSService);

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCPRECCONFIGMUMPS();
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif /* PETSC_VERSION */
