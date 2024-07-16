// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AsyncQueue.cc                                               (C) 2000-2024 */
/*                                                                           */
/* Implémentation d'une file de messages en mémoire partagée.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/IAsyncQueue.h"

#include "arcane/parallel/thread/ArcaneThreadMisc.h"

#include <queue>
#include <mutex>
#include <condition_variable>

#include "arcane_packages.h"

#ifdef ARCANE_HAS_PACKAGE_TBB
#include <tbb/concurrent_queue.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation basique d'une file multi-thread.
 *
 * Utilise un mutex pour protéger les appels.
 */
class SharedMemoryBasicAsyncQueue
: public IAsyncQueue
{
 public:
  void push(void* v) override
  {
    {
      std::unique_lock<std::mutex> lg(m_mutex);
      m_shared_queue.push(v);
    }
    m_conditional_variable.notify_one();
  }
  void* pop() override
  {
    std::unique_lock<std::mutex> lg(m_mutex);
    while (m_shared_queue.empty())
    {
      m_conditional_variable.wait(lg);
    }
    void* v = m_shared_queue.front();
    m_shared_queue.pop();
    return v;
  }
  void* tryPop() override
  {
    std::unique_lock<std::mutex> lg(m_mutex);
    if (m_shared_queue.empty())
      return nullptr;
    void* p = m_shared_queue.front();
    m_shared_queue.pop();
    return p;
  }

 private:

  std::queue<void*> m_shared_queue;
  std::mutex m_mutex;
  std::condition_variable m_conditional_variable;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_HAS_PACKAGE_TBB
class TBBAsyncQueue
: public IAsyncQueue
{
 public:
  void push(void* v)
  {
    m_shared_queue.push(v);
  }
  void* pop()
  {
    void* v = 0;
    int count = 1;
    while (!m_shared_queue.try_pop(v)){
      arcaneDoCPUPause(count);
      if (count<100)
        count *= 2;
    }
    return v;
  }
  void* tryPop()
  {
    void* v = 0;
    m_shared_queue.try_pop(v);
    return v;
  }

 private:

  tbb::concurrent_queue<void*> m_shared_queue;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
#ifdef ARCANE_HAS_PACKAGE_TBB
bool global_use_tbb_queue = true;
#else
bool global_use_tbb_queue = false;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAsyncQueue* IAsyncQueue::
createQueue()
{
  global_use_tbb_queue = false;
#ifdef ARCANE_HAS_PACKAGE_TBB
  if (global_use_tbb_queue)
    return new TBBAsyncQueue();
#endif
  auto* v = new SharedMemoryBasicAsyncQueue();
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
