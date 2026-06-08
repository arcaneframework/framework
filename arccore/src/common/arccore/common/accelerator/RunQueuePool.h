// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueuePool.h                                              (C) 2000-2025 */
/*                                                                           */
/* Collection of RunQueues.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_RUNQUEUEPOOL_H
#define ARCCORE_COMMON_ACCELERATOR_RUNQUEUEPOOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Array.h"

#include "arccore/common/accelerator/Runner.h"
#include "arccore/common/accelerator/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Collection of RunQueues.
 *
 * initialize() must be called before using the instance.
 *
 * The element access operator returns the
 * (i % poolSize())-th RunQueue in the collection.
 */
class ARCCORE_COMMON_EXPORT RunQueuePool
{
 public:

  //! Creates an empty instance
  RunQueuePool();

 public:

  RunQueuePool(const RunQueuePool&) = delete;
  RunQueuePool(RunQueuePool&&) = delete;
  RunQueuePool& operator=(const RunQueuePool&) = delete;
  RunQueuePool& operator=(RunQueuePool&&) = delete;

 public:

  //! Initializes the instance with \a pool_size RunQueues
  void initialize(Runner& runner, Int32 pool_size);
  //! Initializes the instance with \a pool_size RunQueues
  void initialize(Runner& runner, Int32 pool_size, const RunQueueBuildInfo& bi);

 public:

  //! To retrieve the i % poolSize()-th execution queue
  const RunQueue& operator[](Int32 i) const
  {
    return m_pool[i % m_pool_size];
  }

  //! To retrieve the i % poolSize()-th execution queue
  RunQueue& operator[](Int32 i)
  {
    return m_pool[i % m_pool_size];
  }

  //! Forces waiting for all RunQueues
  void barrier() const;

  //! Size of the collection
  Int32 size() const { return m_pool_size; }

  //! Modifies the asynchronous state of the queues.
  void setAsync(bool v) const;

 private:

  UniqueArray<RunQueue> m_pool;
  Runner m_runner;
  Int32 m_pool_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
