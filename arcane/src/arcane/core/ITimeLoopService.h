// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeLoopService.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service opérant lors de la boucle en temps.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMELOOPSERVICE_H
#define ARCANE_CORE_ITIMELOOPSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Interface d'un service opérant lors de la boucle en temps.
 *
 * Un service implémentant cette interface permet de spécifier une
 * action qui sera effectuée à un point précis de la boucle en temps.
 */
class ITimeLoopService
{
 public:

  virtual ~ITimeLoopService() = default;

 public:

  virtual void onTimeLoopBeginLoop() = 0;
  virtual void onTimeLoopEndLoop() = 0;
  virtual void onTimeLoopStartInit() = 0;
  virtual void onTimeLoopContinueInit() = 0;
  virtual void onTimeLoopExit() = 0;
  virtual void onTimeLoopMeshChanged() = 0;
  virtual void onTimeLoopRestore() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
