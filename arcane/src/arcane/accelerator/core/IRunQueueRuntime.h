// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRunQueueRuntime.h                                          (C) 2000-2022 */
/*                                                                           */
/* Interface du runtime associé à une RunQueue.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_IRUNQUEUERUNTIME_H
#define ARCANE_ACCELERATOR_IRUNQUEUERUNTIME_H
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
 * \brief Interface du runtime associé à une RunQueue.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IRunQueueRuntime
{
 public:

  virtual ~IRunQueueRuntime() = default;

 public:

  virtual void notifyBeginLaunchKernel() = 0;
  virtual void notifyEndLaunchKernel() = 0;
  virtual void barrier() = 0;
  virtual eExecutionPolicy executionPolicy() const = 0;
  virtual IRunQueueStream* createStream(const RunQueueBuildInfo& bi) = 0;
  virtual impl::IRunQueueEventImpl* createEventImpl() = 0;
  virtual void setMemoryAdvice(MemoryView buffer, eMemoryAdvice advice, DeviceId device_id) = 0;
  virtual void unsetMemoryAdvice(MemoryView buffer, eMemoryAdvice advice, DeviceId device_id) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
