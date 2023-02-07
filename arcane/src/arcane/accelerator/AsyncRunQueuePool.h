// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AsyncRunQueuePool.h                                         (C) 2000-2022 */
/*                                                                           */
/* Collection de file d'exécution asynchrone avec priorité sur accélérateur. */
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
 * \brief Collection de file d'exécution asynchrone avec priorité sur accélérateur.
 *
 * La taille de la collection est uniquement paramétrable à sa création et il
 * existe une taille maximale de POOL_MAX_SIZE.
 * Si la taille demandée est supérieure à celle-ci, la taille réelle de la 
 * collection sera de POOL_MAX_SIZE.
 * L'opérateur d'accès aux éléments renvoit le (i % poolSize()) ème
 *
 * \warning API en cours de définition.
 * \note Courtesy of D.Dureau from Pattern4GPU
 */
class ARCANE_ACCELERATOR_CORE_EXPORT AsyncRunQueuePool
{
 public:
  //! au plus 32 queues (32 = nb de kernels max exécutables simultanément)
  // TODO: Constante tirée du code de David Dureau dans Pattern4GPU, cette limitation est-elle nécessaire ?
  // TODO: Doit on autoriser a demander plus ? puis restreindre ? En ce cas, qu'advient il du random accessor ?
  static constexpr Int32 POOL_MAX_SIZE = 32;

 public:
  AsyncRunQueuePool() = delete;
  AsyncRunQueuePool(const AsyncRunQueuePool&) = delete;
  AsyncRunQueuePool(AsyncRunQueuePool&&) = delete;
  AsyncRunQueuePool& operator=(const AsyncRunQueuePool&) = delete;
  AsyncRunQueuePool& operator=(AsyncRunQueuePool&&) = delete;

  AsyncRunQueuePool(Runner& runner, Int32 pool_size = POOL_MAX_SIZE,
                    eRunQueuePriority queues_priority = eRunQueuePriority::Default)
  : m_pool_size(std::min(pool_size, POOL_MAX_SIZE))
  {
    m_pool.reserve(m_pool_size);
    for (Int32 i(0); i < m_pool_size; ++i) {
      RunQueueBuildInfo bi;
      // TODO: pourra etre changé par std::to_underlying en c++23 (GCC11	CLANG13	MSVC19.30)
      bi.setPriority(static_cast<std::underlying_type_t<eRunQueuePriority>>(queues_priority));
      auto queue_ref = makeQueueRef(runner, bi);
      queue_ref->setAsync(true);
      m_pool.add(queue_ref);
    }
  }

  // TODO: Doit on mettre le destructeur virutal pour un éventuel héritage ?
  ~AsyncRunQueuePool()
  {
    m_pool_size = 0;
    m_pool.clear();
  }

  //! Pour récupérer la i % poolSize() ième queue d'exécution
  inline const RunQueue& operator[](Int32 i) const
  {
    return *(m_pool[i % m_pool_size].get());
  }

  //! Pour récupérer la i % poolSize() ième queue d'exécution
  inline RunQueue* operator[](Int32 i)
  {
    return m_pool[i % m_pool_size].get();
  }

  //! Force l'attente de toutes les RunQueue
  void waitAll() const {
    for (auto q : m_pool)
      q->barrier();
  }

  //! Taille de la collection
  inline Int32 poolSize() const {
    return m_pool_size;
  }

 // TODO: Doit on changer pour protected pour un éventuel héritage ?
 private:
  UniqueArray<Ref<RunQueue>> m_pool;
  Int32 m_pool_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé un pool de file temporaire associée à \a runner.
 *
 * La taille du pool est de AsyncRunQueuePool::POOL_MAX_SIZE et les queues ont
 * une priorité par défault.
 * 
 * Cet appel est thread-safe si runner.isConcurrentQueueCreation()==true.
 */
inline AsyncRunQueuePool
makeAsyncQueuePool(Runner& runner)
{
  return AsyncRunQueuePool(runner);
}

/*!
 * \brief Créé un pool de file temporaire associée à \a runner.
 *
 * Cet appel est thread-safe si runner.isConcurrentQueueCreation()==true.
 */
inline AsyncRunQueuePool
makeAsyncQueuePool(Runner& runner, Int32 size, eRunQueuePriority priority = eRunQueuePriority::Default)
{
  return AsyncRunQueuePool(runner, size, priority);
}

/*!
 * \brief Créé un pool de file temporaire associée à \a runner.
 *
 * La taille du pool est de AsyncRunQueuePool::POOL_MAX_SIZE et les queues ont
 * une priorité par défault.
 * 
 * Cet appel est thread-safe si runner.isConcurrentQueueCreation()==true.
 */
inline AsyncRunQueuePool
makeAsyncQueuePool(Runner* runner)
{
  ARCANE_CHECK_POINTER(runner);
  return AsyncRunQueuePool(*runner);
}

/*!
 * \brief Créé un pool de file temporaire associée à \a runner.
 *
 * Cet appel est thread-safe si runner.isConcurrentQueueCreation()==true.
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
