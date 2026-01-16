// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/kernels/petsc/PETScPrecomp.h>
#include <alien/kernels/petsc/PETScBackEnd.h>
#include <alien/core/backend/LinearSolver.h>
#include <alien/kernels/petsc/linear_solver/PETScOptionTypes.h>
#include <alien/kernels/petsc/linear_solver/PETScInternalLinearSolver.h>
#include <alien/kernels/petsc/linear_solver/IPETScKSP.h>
#include <ALIEN/axl/PETScLinearSolver_axl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! Classe/service de résolution linéaire PETSc
 *  Une couche intermédiaire IPETScLinearSolver pourrait etre
 *  ajoutée afin d'utiliser PETScLinearSolverService sans sa partie 'service'
 */
class ALIEN_EXTERNAL_PACKAGES_EXPORT PETScLinearSolverService
    : public ArcanePETScLinearSolverObject,
      public LinearSolver<BackEnd::tag::petsc>
{
 private:
 public:
/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
  PETScLinearSolverService(const Arcane::ServiceBuildInfo& sbi);
#endif

  PETScLinearSolverService(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsPETScLinearSolver> options);

  /** Destructeur de la classe */
  virtual ~PETScLinearSolverService();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

