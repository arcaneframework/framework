// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

/**
 * Interface du service de résolution de système linéaire
 */
#include "alien/AlienLegacyConfig.h"
#ifdef ALIEN_USE_HARTS
#include "HARTS/HARTS.h"
#endif
#ifdef ALIEN_USE_HTSSOLVER
#include "HARTSSolver/HTS.h"
#endif
#include <alien/kernels/hts/linear_solver/arcane/HTSLinearSolver.h>
#include <ALIEN/axl/HTSSolver_StrongOptions.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
HTSLinearSolver::HTSLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneHTSSolverObject(sbi)
, Alien::HTSInternalLinearSolver(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
//, LinearSolver<BackEnd::tag::htssolver>(sbi.subDomain()->parallelMng(), options())
{
}
#endif

HTSLinearSolver::HTSLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsHTSSolver> _options)
: ArcaneHTSSolverObject(_options)
, Alien::HTSInternalLinearSolver(parallel_mng, options())
//, LinearSolver<BackEnd::tag::htssolver>(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_HTSSOLVER(HTSSolver, HTSLinearSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_HTSSOLVER();
