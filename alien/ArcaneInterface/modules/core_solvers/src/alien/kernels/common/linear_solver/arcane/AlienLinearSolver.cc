// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

/**
 * Interface du service de résolution de système linéaire
 */
#include "alien/AlienLegacyConfig.h"
#include <alien/kernels/common/linear_solver/arcane/AlienLinearSolver.h>
#include <ALIEN/axl/AlienCoreSolver_StrongOptions.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
AlienLinearSolver::AlienLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneAlienCoreSolverObject(sbi)
, Alien::AlienCoreLinearSolver(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
{
}
#endif

AlienLinearSolver::AlienLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsAlienCoreSolver> _options)
: ArcaneAlienCoreSolverObject(_options)
, Alien::AlienCoreLinearSolver(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ALIENCORESOLVER(AlienCoreSolver, AlienLinearSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_ALIENCORESOLVER();
