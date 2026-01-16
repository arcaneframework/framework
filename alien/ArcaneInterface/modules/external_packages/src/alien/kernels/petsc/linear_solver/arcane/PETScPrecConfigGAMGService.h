// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <alien/kernels/petsc/PETScPrecomp.h>
#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/kernels/petsc/linear_solver/IPETScPC.h>
#include <alien/kernels/petsc/linear_solver/IPETScKSP.h>
#include <alien/kernels/petsc/linear_solver/PETScConfig.h>

#include <ALIEN/axl/PETScPrecConfigGAMG_axl.h>


/*
 * PCGAMG
   Geometric algebraic multigrid (AMG) preconditioner


 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNAL_PACKAGES_EXPORT PETScPrecConfigGAMGService
    : public ArcanePETScPrecConfigGAMGObject,
      public PETScConfig
{
 public:
/** Constructeur de la classe */
#ifdef ALIEN_USE_ARCANE
  PETScPrecConfigGAMGService(const Arcane::ServiceBuildInfo& sbi);
#endif

  PETScPrecConfigGAMGService(
      Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::shared_ptr<IOptionsPETScPrecConfigGAMG> options);

  /** Destructeur de la classe */
  virtual ~PETScPrecConfigGAMGService() {}

 public:
  //! Initialisation
  void configure(PC& pc, const ISpace& space, const MatrixDistribution& distribution);

  //! Check need of KSPSetUp before calling this PC configure
  virtual bool needPrematureKSPSetUp() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
