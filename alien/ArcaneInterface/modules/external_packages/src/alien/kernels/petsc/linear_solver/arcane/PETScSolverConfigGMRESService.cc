// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#include <alien/kernels/petsc/linear_solver/arcane/PETScSolverConfigGMRESService.h>
#include <ALIEN/axl/PETScSolverConfigGMRES_StrongOptions.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
PETScSolverConfigGMRESService::PETScSolverConfigGMRESService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScSolverConfigGMRESObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  ;
}
#endif
PETScSolverConfigGMRESService::PETScSolverConfigGMRESService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScSolverConfigGMRES> options)
: ArcanePETScSolverConfigGMRESObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
  ;
}

//! Initialisation
void
PETScSolverConfigGMRESService::configure(
    KSP& ksp, const ISpace& space, const MatrixDistribution& distribution)
{
  alien_debug([&] { cout() << "configure PETSc gmres solver"; });

  PETScInitType::apply(this, ksp, options()->initType());

  checkError(
      "Set solver tolerances", KSPSetTolerances(ksp, options()->stopCriteriaValue(),
                                   1e-15, PETSC_DEFAULT, options()->numIterationsMax()));
  checkError("Solver set type", KSPSetType(ksp, KSPGMRES));
  checkError("Solver set restart", KSPGMRESSetRestart(ksp, options()->restart()));
#ifdef PETSC_HAVE_KSPGMRESSETBREAKDOWNTOLERANCE
  if(options()->breakdownTol()>0.)
  {
	  std::cout<<"CHANGE DEFAULT BREAKDOWNTOLERANCE : "<<options()->breakdownTol()<<std::endl ;
    checkError("Solver set breakdown tol",KSPGMRESSetBreakdownTolerance(ksp,options()->breakdownTol())) ;
  }
#endif
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

  if (options()->useModifiedGramSchmidt()) {
    checkError("Solver set Modified Gram Schmidt Orthogonalization ",
        KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization));
  } else {
    checkError("Solver set Classical Gram Schmidt Orthogonalization ",
        KSPGMRESSetOrthogonalization(ksp, KSPGMRESClassicalGramSchmidtOrthogonalization));
  }

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

ARCANE_REGISTER_SERVICE_PETSCSOLVERCONFIGGMRES(GMRES, PETScSolverConfigGMRESService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCSOLVERCONFIGGMRES();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
