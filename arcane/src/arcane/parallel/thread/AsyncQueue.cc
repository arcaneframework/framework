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

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"

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
    std::unique_lock<std::mutex> lg(m_mutex);
    m_shared_queue.push(v);
    // NOTE: normalement il n'y a pas besoin d'avoir le verrou actif
    // lors de l'appel à 'notify_one()' mais cela génère des avertissements
    // avec helgrind (valgrind). Du coup on laisse le verrou pour éviter cela.
    // Il faudrait vérifier si cela à des effets sur les performances (dans
    // les tests Arcane du CI ce n'est pas le cas).
    m_conditional_variable.notify_one();
  }
  void* pop() override
  {
    std::unique_lock<std::mutex> lg(m_mutex);
    while (m_shared_queue.empty()) {
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
    while (!m_shared_queue.try_pop(v)) {
      arcaneDoCPUPause(count);
      if (count < 100)
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

IAsyncQueue* IAsyncQueue::
createQueue()
{
  // Par défaut n'utilise pas l'attente active car il n'y a pas de différence
  // notable de performance  et cela évite des contentions lorsque le nombre
  // de coeurs disponibles est inférieure au nombre de threads.
  [[maybe_unused]] bool use_active_queue = false;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_ACTIVE_SHM_QUEUE", true))
    use_active_queue = (v.value() != 0);
  IAsyncQueue* queue = nullptr;
#ifdef ARCANE_HAS_PACKAGE_TBB
  if (use_active_queue)
    queue = new TBBAsyncQueue();
#endif
  if (!queue)
    queue = new SharedMemoryBasicAsyncQueue();
  return queue;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
