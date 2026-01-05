// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#ifdef ARCGEOSIM_COMP
#include "Appli/IInfoModel.h"
#endif

#include <alien/utils/Precomp.h>
#include <alien/AlienCoreSolversPrecomp.h>
#include <alien/core/backend/LinearSolver.h>
#include <alien/kernels/sycl/linear_solver/AlienCoreSYCLLinearSolver.h>
#include <alien/kernels/common/AlienCoreSolverOptionTypes.h>
#include <ALIEN/axl/AlienCoreSolver_axl.h>


namespace Alien {

class ALIEN_CORE_SOLVERS_EXPORT AlienSYCLLinearSolver
: public ArcaneAlienCoreSolverObject
, public AlienCoreSYCLLinearSolver
{
 public:
/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
  AlienSYCLLinearSolver(const Arcane::ServiceBuildInfo& sbi);
#endif

  AlienSYCLLinearSolver(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsAlienCoreSolver> _options);

  /** Destructeur de la classe */
  virtual ~AlienSYCLLinearSolver(){};
};

} // namespace Alien
