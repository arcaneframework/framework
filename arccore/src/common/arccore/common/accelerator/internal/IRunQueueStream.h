// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRunQueueStream.h                                           (C) 2000-2026 */
/*                                                                           */
/* Interface of an execution stream for a RunQueue.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_IRUNQUEUESTREAM_H
#define ARCCORE_COMMON_ACCELERATOR_IRUNQUEUESTREAM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of an execution stream for a RunQueue.
 */
class ARCCORE_COMMON_EXPORT IRunQueueStream
{
 public:

  virtual ~IRunQueueStream() = default;

 public:

  //! Notification before command launch
  virtual void notifyBeginLaunchKernel(RunCommandImpl& command) = 0;

  /*!
   * \brief Notification of command launch completion.
   *
   * In asynchronous mode, the command can continue to run in the background.
   */
  virtual void notifyEndLaunchKernel(RunCommandImpl& command) = 0;

  /*!
   * \brief Blocks until all actions associated with this queue
   * are finished.
   *
   * This includes commands (RunCommandImpl) and other actions such as
   * asynchronous memory copies.
   */
  virtual void barrier() = 0;

  //! Performs a copy between two memory regions
  virtual void copyMemory(const MemoryCopyArgs& args) = 0;

  //! Performs a prefetch of a memory region
  virtual void prefetchMemory(const MemoryPrefetchArgs& args) = 0;

 public:

  //! Pointer to the internal structure dependent on the implementation
  virtual Impl::NativeStream nativeStream() = 0;

  //! Barrier without exception. Returns true in case of error
  virtual bool _barrierNoException() = 0;

  //! For SYCL, positions the event associated with the last executed command.
  virtual void _setSyclLastCommandEvent([[maybe_unused]] void* sycl_event_ptr) {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
