﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

/* Author : havep at Fri Jun 27 21:34:36 2008
 * Generated by createNew
 */

#include <alien/kernels/petsc/linear_solver/spai/PETScPrecConfigSPAIService.h>
#include <ALIEN/axl/PETScPrecConfigSPAI_StrongOptions.h>

#include <arccore/message_passing/IMessagePassingMng.h>

namespace Alien {

/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
PETScPrecConfigSPAIService::PETScPrecConfigSPAIService(
    const Arcane::ServiceBuildInfo& sbi)
: ArcanePETScPrecConfigSPAIObject(sbi)
, PETScConfig(sbi.subDomain()->parallelMng()->isParallel())
{
  ;
}
#endif
PETScPrecConfigSPAIService::PETScPrecConfigSPAIService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsPETScPrecConfigSPAI> options)
: ArcanePETScPrecConfigSPAIObject(options)
, PETScConfig(parallel_mng->commSize() > 1)
{
}

void
PETScPrecConfigSPAIService::configure([[maybe_unused]] PC& pc,
                                      [[maybe_unused]] const ISpace& space,
                                      [[maybe_unused]] const MatrixDistribution& distribution)
{
#ifndef PETSC_HAVE_SPAI
  alien_fatal([&] { cout() << "SPAI not available in PETSc"; });
#else
  checkError("Set preconditioner",PCSetType(pc,PCSPAI));

  double epsilon = options()->epsilon();
  if (epsilon > 0 and epsilon < 1)
    checkError("Set SPAI epsilon",PCSPAISetEpsilon(pc,epsilon));
  else
    alien_fatal([&] { cout() << "SPAI epsilon must be in [0:1]"; });

  int nonzero_max = options()->nonzeroMax();
  if (nonzero_max >= 0)
    checkError("Set SPAI epsilon",PCSPAISetMaxNew(pc,nonzero_max));
  else
     alien_fatal([&] { cout() << "SPAI nonzero-max must be positive"; });

  int nb_steps = options()->nbSteps();
  if(nb_steps > 0)
  {
    checkError("Set SPAI nb steps", PCSPAISetNBSteps(pc,nb_steps));
  }

  bool verbose = options()->verbose();
  checkError("Set SPAI verbose", PCSPAISetVerbose(pc,verbose));

#endif
}


//  //! Check need of KSPSetUp before calling this PC configure

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCPRECCONFIGSPAI(SPAI, PETScPrecConfigSPAIService);

} // namespace Alien

REGISTER_STRONG_OPTIONS_PETSCPRECCONFIGSPAI();
