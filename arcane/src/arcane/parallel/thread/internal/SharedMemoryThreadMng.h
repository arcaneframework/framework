// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryThreadMng.h                                     (C) 2000-2024 */
/*                                                                           */
/* Implémentation de IThreadMng pour la mémoire partagée.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYTHREADMNG_H
#define ARCANE_PARALLEL_THREAD_INTERNAL_SHAREDMEMORYTHREADMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/ArcaneThread.h"

#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SharedMemoryThreadMng
: public IThreadMng
{
 public:

  void beginCriticalSection() override
  {
    m_mutex.lock();
  }
  void endCriticalSection() override
  {
    m_mutex.unlock();
  }

 private:

  std::mutex m_mutex;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
