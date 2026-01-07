// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/kernels/petsc/linear_solver/arcane/PETScSolverConfigRichardsonService.h>
#include <ALIEN/axl/PETScSolverConfigRichardson_StrongOptions.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
PETScSolverConfigRichardsonService::PETScSolverConfigRichardsonService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScSolverConfigRichardsonObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  ;
}
#endif
PETScSolverConfigRichardsonService::PETScSolverConfigRichardsonService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScSolverConfigRichardson> options)
: ArcanePETScSolverConfigRichardsonObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
  ;
}

//! Initialisation
void
PETScSolverConfigRichardsonService::configure(
    KSP& ksp, const ISpace& space, const MatrixDistribution& distribution)
{
  alien_debug([&] { cout() << "configure PETSc richardson solver"; });

  PETScInitType::apply(this, ksp, options()->initType());

  checkError(
      "Set solver tolerances", KSPSetTolerances(ksp, options()->stopCriteriaValue(),
                                   1e-15, PETSC_DEFAULT, options()->numIterationsMax()));
  checkError("Solver set type", KSPSetType(ksp, KSPRICHARDSON));

  if (options()->right()) {
    //#ifndef PETSC_KSPSETPCSIDE_NEW
    //checkError(
    //    " Set solver preconditioner side ", KSPSetPreconditionerSide(ksp, PC_RIGHT));
    //#else /* PETSC_KSPSETPCSIDE_NEW */
    checkError(" Set solver preconditioner side ", KSPSetPCSide(ksp, PC_RIGHT));
    //#endif /* PETSC_KSPSETPCSIDE_NEW */
    checkError(" Set solver unpreconditioned norm type ",
        KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  }

  double scale = options()->scale();
#ifdef PETSC_HAVE_KSPRICHARDSONSETSELFSCALE
  if (scale != 0.0) {
#endif // PETSC_HAVE_KSPRICHARDSONSETSELFSCALE
    checkError("Set Richardson Scale", KSPRichardsonSetScale(ksp, scale));
#ifdef PETSC_HAVE_KSPRICHARDSONSETSELFSCALE
  } else {
    checkError("Set Richardson Self Scale", KSPRichardsonSetSelfScale(ksp, PETSC_TRUE));
  }
#endif // PETSC_HAVE_KSPRICHARDSONSETSELFSCALE

  PC pc;
  checkError("Get preconditioner", KSPGetPC(ksp, &pc));
  IPETScPC* preconditioner = options()->preconditioner();
  bool needSetUp = preconditioner->needPrematureKSPSetUp();
  if (needSetUp) {
    checkError("Solver setup", KSPSetUp(ksp));
    preconditioner->configure(pc, space, distribution);
  } else {
    preconditioner->configure(pc, space, distribution);
    checkError("Solver setup", KSPSetUp(ksp));
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCSOLVERCONFIGRICHARDSON(
    Richardson, PETScSolverConfigRichardsonService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCSOLVERCONFIGRICHARDSON();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
