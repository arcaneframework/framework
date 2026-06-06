// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AsyncRunQueuePool.h                                         (C) 2000-2022 */
/*                                                                           */
/* Collection of asynchronous execution queues with priority on accelerator. */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_ASYNC_RUNQUEUE_POOL_H
#define ARCANE_ACCELERATOR_ASYNC_RUNQUEUE_POOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Collection of asynchronous execution queues with priority on accelerator.
 *
 * The size of the collection is only configurable upon creation and there
 * is a maximum size of POOL_MAX_SIZE.
 * If the requested size is greater than this, the actual size of the
 * collection will be POOL_MAX_SIZE.
 * The element access operator returns the (i % poolSize())-th
 *
 * \warning API is currently under definition.
 * \note Courtesy of D.Dureau from Pattern4GPU
 */
class AsyncRunQueuePool
{
 public:

  //! Up to 32 queues (32 = max number of kernels executable simultaneously)
  // TODO: Constant taken from David Dureau's code in Pattern4GPU, is this limitation necessary?
  static constexpr Int32 POOL_MAX_SIZE = 32;

 public:

  AsyncRunQueuePool() = delete;
  AsyncRunQueuePool(const AsyncRunQueuePool&) = delete;
  AsyncRunQueuePool(AsyncRunQueuePool&&) = delete;
  AsyncRunQueuePool& operator=(const AsyncRunQueuePool&) = delete;
  AsyncRunQueuePool& operator=(AsyncRunQueuePool&&) = delete;

  explicit AsyncRunQueuePool(Runner& runner, Int32 pool_size = POOL_MAX_SIZE,
                             eRunQueuePriority queues_priority = eRunQueuePriority::Default)
  : m_pool_size(std::min(pool_size, POOL_MAX_SIZE))
  {
    m_pool.reserve(m_pool_size);
    for (Int32 i(0); i < m_pool_size; ++i) {
      RunQueueBuildInfo bi;
      // TODO: can be changed by std::to_underlying in c++23 (GCC11	CLANG13	MSVC19.30)
      bi.setPriority(static_cast<std::underlying_type_t<eRunQueuePriority>>(queues_priority));
      auto queue_ref = makeQueueRef(runner, bi);
      queue_ref->setAsync(true);
      m_pool.add(queue_ref);
    }
  }

  // TODO: Should the destructor be virtual for potential inheritance?
  ~AsyncRunQueuePool()
  {
    m_pool_size = 0;
    m_pool.clear();
  }

  //! To retrieve the i % poolSize() i-th execution queue
  inline const RunQueue& operator[](Int32 i) const
  {
    return *(m_pool[i % m_pool_size].get());
  }

  //! To retrieve the i % poolSize() i-th execution queue
  inline RunQueue* operator[](Int32 i)
  {
    return m_pool[i % m_pool_size].get();
  }

  //! Forces waiting for all RunQueues
  void waitAll() const
  {
    for (auto q : m_pool)
      q->barrier();
  }

  //! Size of the collection
  inline Int32 poolSize() const
  {
    return m_pool_size;
  }

  // TODO: Should it be changed to protected for potential inheritance?
 private:

  UniqueArray<Ref<RunQueue>> m_pool;
  Int32 m_pool_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creates a temporary queue pool associated with \a runner.
 *
 * The pool size is AsyncRunQueuePool::POOL_MAX_SIZE and the queues have
 * a default priority.
 *
 * This call is thread-safe if runner.isConcurrentQueueCreation()==true.
 */
inline AsyncRunQueuePool
makeAsyncQueuePool(Runner& runner)
{
  return AsyncRunQueuePool(runner);
}

/*!
 * \brief Creates a temporary queue pool associated with \a runner.
 *
 * This call is thread-safe if runner.isConcurrentQueueCreation()==true.
 */
inline AsyncRunQueuePool
makeAsyncQueuePool(Runner& runner, Int32 size, eRunQueuePriority priority = eRunQueuePriority::Default)
{
  return AsyncRunQueuePool(runner, size, priority);
}

/*!
 * \brief Creates a temporary queue pool associated with \a runner.
 *
 * The pool size is AsyncRunQueuePool::POOL_MAX_SIZE and the queues have
 * a default priority.
 *
 * This call is thread-safe if runner.isConcurrentQueueCreation()==true.
 */
inline AsyncRunQueuePool
makeAsyncQueuePool(Runner* runner)
{
  ARCANE_CHECK_POINTER(runner);
  return AsyncRunQueuePool(*runner);
}

/*!
 * \brief Creates a temporary queue pool associated with \a runner.
 *
 * This call is thread-safe if runner.isConcurrentQueueCreation()==true.
 */
inline AsyncRunQueuePool
makeAsyncQueuePool(Runner* runner, Int32 size, eRunQueuePriority priority = eRunQueuePriority::Default)
{
  ARCANE_CHECK_POINTER(runner);
  return AsyncRunQueuePool(*runner, size, priority);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
