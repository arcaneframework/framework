// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ALIEN_MCGIMPL_MCGLINEARSOLVER_H
#define ALIEN_MCGIMPL_MCGLINEARSOLVER_H

#include <alien/utils/Precomp.h>
#include <alien/core/backend/LinearSolver.h>

#include "alien/AlienIFPENSolversPrecomp.h"
#include "alien/kernels/mcg/MCGPrecomp.h"
#include "alien/kernels/mcg/MCGBackEnd.h"
#include "alien/kernels/mcg/linear_solver/MCGInternalLinearSolver.h"
#include "alien/kernels/mcg/linear_solver/MCGOptionTypes.h"
#include "ALIEN/axl/MCGSolver_axl.h"

/**
 * Interface du service de résolution de système linéaire
 */

namespace Alien {

class ALIEN_IFPEN_SOLVERS_EXPORT MCGLinearSolver : public ArcaneMCGSolverObject,
                                                   public MCGInternalLinearSolver
#ifdef ARCGEOSIM_COMP
                                                  ,public IInfoModel
#endif
{
 public:
  /** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  MCGLinearSolver(const Arcane::ServiceBuildInfo& sbi);
#endif

  MCGLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsMCGSolver> _options);

  /** Destructeur de la classe */
  ~MCGLinearSolver() final = default;
};

}

#endif /* ALIEN_MCGIMPL_MCGLINEARSOLVER_H */
