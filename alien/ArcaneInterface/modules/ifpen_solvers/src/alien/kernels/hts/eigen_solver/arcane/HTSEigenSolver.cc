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
#include <alien/kernels/hts/eigen_solver/arcane/HTSEigenSolver.h>
#include <ALIEN/axl/HTSEigenSolver_StrongOptions.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
HTSEigenSolver::HTSEigenSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneHTSEigenSolverObject(sbi)
, Alien::HTSInternalEigenSolver(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
//, EigenSolver<BackEnd::tag::htssolver>(sbi.subDomain()->parallelMng(), options())
{
}
#endif

HTSEigenSolver::HTSEigenSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsHTSEigenSolver> _options)
: ArcaneHTSEigenSolverObject(_options)
, Alien::HTSInternalEigenSolver(parallel_mng, options())
//, EigenSolver<BackEnd::tag::htssolver>(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_HTSEIGENSOLVER(HTSEigenSolver, HTSEigenSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_HTSEIGENSOLVER();
