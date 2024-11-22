// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/utils/Precomp.h>
#include <alien/AlienComposyxPrecomp.h>

#include <alien/kernels/composyx/ComposyxPrecomp.h>
#include <alien/kernels/composyx/ComposyxBackEnd.h>
#include <alien/core/backend/LinearSolver.h>
#include <alien/kernels/composyx/linear_solver/ComposyxOptionTypes.h>
#include <ALIEN/axl/ComposyxSolver_axl.h>

#include <alien/kernels/composyx/linear_solver/ComposyxInternalSolver.h>
#include <ALIEN/axl/ComposyxSolver_StrongOptions.h>

namespace Alien {

class ALIEN_COMPOSYX_EXPORT ComposyxLinearSolver 
: public ArcaneComposyxSolverObject
, public ComposyxInternalSolver
//, public LinearSolver<BackEnd::tag::composyx>
{
 public:
/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  ComposyxLinearSolver(const Arcane::ServiceBuildInfo& sbi);
#endif

  ComposyxLinearSolver(
      IMessagePassingMng* parallel_mng, std::shared_ptr<IOptionsComposyxSolver> _options);

  /** Destructeur de la classe */
  virtual ~ComposyxLinearSolver(){};
};

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
ComposyxLinearSolver::ComposyxLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneComposyxSolverObject(sbi)
, Alien::ComposyxInternalSolver(sbi.subDomain()->parallelMng()->messagePassingMng(), options())
//, LinearSolver<BackEnd::tag::composyx>(sbi.subDomain()->parallelMng()->messagePassingMng(), options())
{
}
#endif

ComposyxLinearSolver::ComposyxLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng, std::shared_ptr<IOptionsComposyxSolver> _options)
: ArcaneComposyxSolverObject(_options)
, Alien::ComposyxInternalSolver(parallel_mng, options())
//, LinearSolver<BackEnd::tag::composyx>(parallel_mng, options())
{
}


/*---------------------------------------------------------------------------*/
ARCANE_REGISTER_SERVICE_COMPOSYXSOLVER(ComposyxSolver, ComposyxLinearSolver);
}

REGISTER_STRONG_OPTIONS_COMPOSYXSOLVER();
