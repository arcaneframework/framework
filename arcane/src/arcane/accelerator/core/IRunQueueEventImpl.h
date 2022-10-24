// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRunQueueEventImpl.h                                        (C) 2000-2022 */
/*                                                                           */
/* Interface de l'implémentation d'un évènement.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_IRUNQUEUEEVENTIMPL_H
#define ARCANE_ACCELERATOR_IRUNQUEUEEVENTIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface de l'implémentation d'un évènement.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IRunQueueEventImpl
{
 public:

  virtual ~IRunQueueEventImpl() = default;

 public:

  virtual void recordQueue(IRunQueueStream* stream) = 0;
  virtual void wait() = 0;
  virtual void waitForEvent(IRunQueueStream* stream) = 0;

  //! Temps écoulé (en nanoseconde) entre l'évènement \a from_event et cet évènement.
  virtual Int64 elapsedTime(IRunQueueEventImpl* from_event) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
