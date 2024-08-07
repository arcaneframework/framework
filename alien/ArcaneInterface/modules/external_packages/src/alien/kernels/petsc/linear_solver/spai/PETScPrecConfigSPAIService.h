﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

/* Author : havep at Fri Jun 27 21:34:36 2008
 * Generated by createNew
 */

#ifndef PETSCPRECCONFIGSPAISERVICE_H
#define PETSCPRECCONFIGSPAISERVICE_H

#include <alien/kernels/petsc/PETScPrecomp.h>
#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/kernels/petsc/linear_solver/IPETScPC.h>
#include <alien/kernels/petsc/linear_solver/PETScConfig.h>
#include <ALIEN/axl/PETScPrecConfigSPAI_axl.h>

namespace Alien {

class ALIEN_EXTERNAL_PACKAGES_EXPORT PETScPrecConfigSPAIService
    : public ArcanePETScPrecConfigSPAIObject,
      public PETScConfig
{
 public:
/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
  PETScPrecConfigSPAIService(const Arcane::ServiceBuildInfo& sbi);
#endif

  PETScPrecConfigSPAIService(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsPETScPrecConfigSPAI> options);

  /** Destructeur de la classe */
  virtual ~PETScPrecConfigSPAIService() {}

 public:
  void configure(PC& pc, const ISpace& space, const MatrixDistribution& distribution);

  //! Check need of KSPSetUp before calling this PC configure
  virtual bool needPrematureKSPSetUp() const { return false; }
};
} // namespace Alien

#endif // PETSCPRECCONFIGSPAISERVICE_H
