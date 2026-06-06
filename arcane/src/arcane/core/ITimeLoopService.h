// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimeLoopService.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Interface of a service operating during the time loop.                    */
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
 * \brief Interface of a service operating during the time loop.
 *
 * A service implementing this interface allows specifying an
 * action that will be performed at a specific point in the time loop.
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
