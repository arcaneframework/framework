// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/kernels/petsc/linear_solver/arcane/PETScSolverConfigBiCGStabService.h>
#include <alien/kernels/petsc/linear_solver/PETScInitType.h>
#include <ALIEN/axl/PETScSolverConfigBiCGStab_StrongOptions.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// using namespace Arcane;
namespace Alien {

/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
PETScSolverConfigBiCGStabService::PETScSolverConfigBiCGStabService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScSolverConfigBiCGStabObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  ;
}
#endif

PETScSolverConfigBiCGStabService::PETScSolverConfigBiCGStabService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScSolverConfigBiCGStab> options)
: ArcanePETScSolverConfigBiCGStabObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
  ;
}

//! Initialisation
void
PETScSolverConfigBiCGStabService::configure(
    KSP& ksp, const ISpace& space, const MatrixDistribution& distribution)
{
  alien_debug([&] { cout() << "configure PETSc bicgs solver"; });

  PETScInitType::apply(this, ksp, options()->initType());

  checkError(
      "Set solver tolerances", KSPSetTolerances(ksp, options()->stopCriteriaValue(),
                                   1e-15, PETSC_DEFAULT, options()->numIterationsMax()));
  checkError("Solver set type", KSPSetType(ksp, KSPBCGS));
  if (options()->right()) {
  //#ifndef PETSC_KSPSETPCSIDE_NEW
  //  checkError(
  //      " Set solver preconditioner side ", KSPSetPreconditionerSide(ksp, PC_RIGHT));
  //#else /* PETSC_KSPSETPCSIDE_NEW */
    checkError(" Set solver preconditioner side ", KSPSetPCSide(ksp, PC_RIGHT));
  //#endif /* PETSC_KSPSETPCSIDE_NEW */
    checkError(" Set solver unpreconditioned norm type ",
        KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  }

  IPETScPC* preconditioner = options()->preconditioner();
  {
    PC pc;
    checkError("Get preconditioner", KSPGetPC(ksp, &pc));
    bool needSetUp = preconditioner->needPrematureKSPSetUp();
    if (needSetUp) {
      checkError("Solver setup", KSPSetUp(ksp));
      preconditioner->configure(pc, space, distribution);
    } else {
      preconditioner->configure(pc, space, distribution);
      checkError("Solver setup", KSPSetUp(ksp));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCSOLVERCONFIGBICGSTAB(
    BiCGStab, PETScSolverConfigBiCGStabService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCSOLVERCONFIGBICGSTAB();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
