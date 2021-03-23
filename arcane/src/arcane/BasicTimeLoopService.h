// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicTimeLoopService.cc                                     (C) 2000-2006 */
/*                                                                           */
/* Classe de base d'un service opérant lors de la boucle en temps.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_BASICTIMELOOPSERVICE_H
#define ARCANE_BASICTIMELOOPSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ITimeLoopService.h"
#include "arcane/BasicService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

  BasicTimeLoopService(const ServiceBuildInfo& sbi)
  : BasicService(sbi) {}
  virtual ~BasicTimeLoopService() {}

 public:

  virtual void onTimeLoopBeginLoop() {}
  virtual void onTimeLoopEndLoop() {}
  virtual void onTimeLoopStartInit() {}
  virtual void onTimeLoopContinueInit() {}
  virtual void onTimeLoopExit() {}
  virtual void onTimeLoopMeshChanged() {}
  virtual void onTimeLoopRestore() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
