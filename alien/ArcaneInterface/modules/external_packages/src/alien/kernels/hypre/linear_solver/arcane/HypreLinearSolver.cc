// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"

#include <alien/kernels/hypre/linear_solver/arcane/HypreLinearSolver.h>
#include <ALIEN/axl/HypreSolver_StrongOptions.h>

namespace Alien {
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ALIEN_USE_ARCANE
HypreLinearSolver::HypreLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneHypreSolverObject(sbi)
, LinearSolver<BackEnd::tag::hypre>(sbi.subDomain()->parallelMng()->messagePassingMng(),
                                    sbi.subDomain()->acceleratorMng()->defaultRunner(),
                                    options())
{

}
#endif

/*---------------------------------------------------------------------------*/

HypreLinearSolver::HypreLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
                                     Arcane::Accelerator::Runner* runner,
                                     std::shared_ptr<IOptionsHypreSolver> _options)
: ArcaneHypreSolverObject(_options)
, LinearSolver<BackEnd::tag::hypre>(parallel_mng, runner, options())
{
  ;
}

/*---------------------------------------------------------------------------*/

HypreLinearSolver::~HypreLinearSolver()
{
  ;
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_HYPRESOLVER(HypreSolver, HypreLinearSolver);

} // namespace Alien

REGISTER_STRONG_OPTIONS_HYPRESOLVER();
