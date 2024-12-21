// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef PETSCSOLVERCONFIGSUPERLUSERVICE_H
#define PETSCSOLVERCONFIGSUPERLUSERVICE_H

#include <alien/kernels/petsc/PETScPrecomp.h>
#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/kernels/petsc/linear_solver/IPETScKSP.h>
#include <alien/kernels/petsc/linear_solver/IPETScPC.h>
#include <alien/kernels/petsc/linear_solver/PETScConfig.h>
#include <ALIEN/axl/PETScSolverConfigMUMPS_axl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNAL_PACKAGES_EXPORT PETScSolverConfigMUMPSService :
        public ArcanePETScSolverConfigMUMPSObject,
        public PETScConfig
{
 public:
  /** Constructeur de la classe */
  PETScSolverConfigMUMPSService(const Arcane::ServiceBuildInfo& sbi);

  PETScSolverConfigMUMPSService(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsPETScSolverConfigMUMPS> options);

  /** Destructeur de la classe */
  virtual ~PETScSolverConfigMUMPSService() {}

 public:
  //! Initialisation
  void configure(KSP& ksp, const ISpace& space, const MatrixDistribution& distribution);
};

} // namespace Alien

#endif
