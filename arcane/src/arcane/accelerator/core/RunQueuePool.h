// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueuePool.h                                              (C) 2000-2025 */
/*                                                                           */
/* Collection de RunQueue.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_RUNQUEUE_POOL_H
#define ARCANE_ACCELERATOR_CORE_RUNQUEUE_POOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Array.h"

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Collection de RunQueue.
 *
 * Il faut appeler initialize() avant d'utiliser l'instance.
 *
 * L'opérateur d'accès aux éléments renvoit la
 * (i % poolSize()) ème RunQueue de la collection.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT RunQueuePool
{
 public:

  //! Créé une instance vide
  RunQueuePool();

 public:

  RunQueuePool(const RunQueuePool&) = delete;
  RunQueuePool(RunQueuePool&&) = delete;
  RunQueuePool& operator=(const RunQueuePool&) = delete;
  RunQueuePool& operator=(RunQueuePool&&) = delete;

 public:

  //! Initialise l'instance avec \a pool_size RunQueue
  void initialize(Runner& runner, Int32 pool_size);
  //! Initialise l'instance avec \a pool_size RunQueue
  void initialize(Runner& runner, Int32 pool_size, const RunQueueBuildInfo& bi);

 public:

  //! Pour récupérer la i % poolSize() ième queue d'exécution
  const RunQueue& operator[](Int32 i) const
  {
    return m_pool[i % m_pool_size];
  }

  //! Pour récupérer la i % poolSize() ième queue d'exécution
  RunQueue& operator[](Int32 i)
  {
    return m_pool[i % m_pool_size];
  }

  //! Force l'attente de toutes les RunQueue
  void barrier() const;

  //! Taille de la collection
  Int32 size() const { return m_pool_size; }

  //! Modifie l'état d'asynchronisme des files.
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
