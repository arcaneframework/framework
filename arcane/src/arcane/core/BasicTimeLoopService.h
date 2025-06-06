// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicTimeLoopService.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un service opérant lors de la boucle en temps.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_BASICTIMELOOPSERVICE_H
#define ARCANE_CORE_BASICTIMELOOPSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeLoopService.h"
#include "arcane/core/BasicService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un service opérant lors de la boucle en temps.
 *
 * \ingroup StandardService
 */
class BasicTimeLoopService
: public BasicService
, public ITimeLoopService
{
 public:

  explicit BasicTimeLoopService(const ServiceBuildInfo& sbi)
  : BasicService(sbi)
  {}

 public:

  void onTimeLoopBeginLoop() override {}
  void onTimeLoopEndLoop() override {}
  void onTimeLoopStartInit() override {}
  void onTimeLoopContinueInit() override {}
  void onTimeLoopExit() override {}
  void onTimeLoopMeshChanged() override {}
  void onTimeLoopRestore() override {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
