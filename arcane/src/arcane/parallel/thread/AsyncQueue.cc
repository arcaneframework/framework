// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AsyncQueue.cc                                               (C) 2000-2021 */
/*                                                                           */
/* Implémentation d'une file de messages en mémoire partagée.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/IAsyncQueue.h"

#include "arcane/parallel/thread/ArcaneThreadMisc.h"

#include "arcane_packages.h"

#ifdef ARCANE_HAS_PACKAGE_TBB
#include <tbb/tbb.h>
#endif

#include <glib.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GlibAsyncQueue
: public IAsyncQueue
{
 public:
  GlibAsyncQueue()
  {
    m_shared_queue = g_async_queue_new();
  }
  ~GlibAsyncQueue() override
  {
    g_async_queue_unref(m_shared_queue);
  }
  void push(void* v) override
  {
    g_async_queue_push(m_shared_queue,v);
  }
  void* pop() override
  {
    void* p = g_async_queue_pop(m_shared_queue);
    return p;
  }
  void* tryPop() override
  {
    void* p = g_async_queue_try_pop(m_shared_queue);
    return p;
  }

 private:

  GAsyncQueue* m_shared_queue;
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
    int count = 1;
    m_shared_queue.try_pop(v);
    return v;
  }

 private:

  tbb::concurrent_queue<void*> m_shared_queue;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAsyncQueue* IAsyncQueue::
createQueue()
{
#ifdef ARCANE_HAS_PACKAGE_TBB
  typedef TBBAsyncQueue AsyncQueueType;
#else
  typedef GlibAsyncQueue AsyncQueueType;
#endif
  auto x = new AsyncQueueType();
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
