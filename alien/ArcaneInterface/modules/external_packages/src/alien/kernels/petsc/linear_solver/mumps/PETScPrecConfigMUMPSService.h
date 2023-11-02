﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef PETSCPRECCONFIGMUMPSSERVICE_H
#define PETSCPRECCONFIGMUMPSSERVICE_H

#include <alien/kernels/petsc/PETScPrecomp.h>
#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/utils/parameter_manager/BaseParameterManager.h>

#include <alien/kernels/petsc/linear_solver/IPETScPC.h>
#include <alien/kernels/petsc/linear_solver/IPETScKSP.h>
#include <alien/kernels/petsc/linear_solver/PETScConfig.h>

/* Pour debugger le ILU, utiliser l'option:
 * <cmd-line-param>-mat_superlu_printstat 1 </cmd-line-param>
 */

#include "petscversion.h"

/*---------------------------------------------------------------------------*/
#if ((PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 3) || (PETSC_VERSION_MAJOR > 3))
/*---------------------------------------------------------------------------*/

#include <ALIEN/axl/PETScPrecConfigMUMPS_axl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Alien {

class ALIEN_EXTERNAL_PACKAGES_EXPORT PETScPrecConfigMUMPSService
    : public ArcanePETScPrecConfigMUMPSObject,
      public PETScConfig
{
 public:
/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
  PETScPrecConfigMUMPSService(const Arcane::ServiceBuildInfo& sbi);
#endif

  PETScPrecConfigMUMPSService(
          Arccore::MessagePassing::IMessagePassingMng* parallel_mng, std::shared_ptr<IOptionsPETScPrecConfigMUMPS> options);

  /** Destructeur de la classe */
  virtual ~PETScPrecConfigMUMPSService() {}

 public:
  //! Initialisation
  void configure(PC& pc, const ISpace& space, const MatrixDistribution& distribution);

  //! Check need of KSPSetUp before calling this PC configure
  virtual bool needPrematureKSPSetUp() const { return true; }
};

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_PETSCPRECCONFIGMUMPS(MUMPS, PETScPrecConfigMUMPSService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace Alien
/*---------------------------------------------------------------------------*/
#endif /* PETSC_VERSION */
/*---------------------------------------------------------------------------*/
#endif // PETSCPRECCONFIGSUPERLUSERVICE_H
