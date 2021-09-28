// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRunQueueStream.h                                           (C) 2000-2021 */
/*                                                                           */
/* Interface d'un flux d'exécution pour une RunQueue.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_IRUNQUEUESTREAM_H
#define ARCANE_ACCELERATOR_IRUNQUEUESTREAM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un flux d'exécution pour une RunQueue.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT IRunQueueStream
{
 public:
 virtual ~IRunQueueStream() noexcept(false) {}
 public:
  virtual void notifyBeginKernel(RunCommand& command) =0;
  virtual void notifyEndKernel(RunCommand& command) =0;
  virtual void barrier() =0;
  virtual void* _internalImpl() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
