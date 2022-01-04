// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GlibThreadMng.h                                             (C) 2000-2020 */
/*                                                                           */
/* Implémentation de IThreadMng avec la Glib.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_GLIBTHREADMNG_H
#define ARCANE_PARALLEL_THREAD_GLIBTHREADMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/ArcaneThread.h"
#include "arccore/concurrency/GlibAdapter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GlibThreadMng
: public IThreadMng
{
 public:
  GlibThreadMng() = default;
 public:
  virtual void beginCriticalSection()
  {
    m_critical_section_mutex.lock();
  }
  virtual void endCriticalSection()
  {
    m_critical_section_mutex.unlock();
  }
 private:
  Arccore::GlibMutex m_critical_section_mutex;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
