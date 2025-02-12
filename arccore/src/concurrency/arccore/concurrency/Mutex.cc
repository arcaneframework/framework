// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Mutex.cc                                                    (C) 2000-2025 */
/*                                                                           */
/* Mutex pour le multi-threading.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/Mutex.h"
#include "arccore/concurrency/IThreadImplementation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Mutex::
Mutex()
{
  m_thread_impl = Concurrency::getThreadImplementation();
  m_p = m_thread_impl->createMutex();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Mutex::
~Mutex()
{
  m_thread_impl->destroyMutex(m_p);
}

void Mutex::
lock()
{
  m_thread_impl->lockMutex(m_p);
}

void Mutex::
unlock()
{
  m_thread_impl->unlockMutex(m_p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MutexImpl* GlobalMutex::m_p = 0;

void GlobalMutex::
init(MutexImpl* p)
{
  m_p = p;
}

void GlobalMutex::
destroy()
{
  m_p = 0;
}

void GlobalMutex::
lock()
{
  if (m_p)
    Concurrency::getThreadImplementation()->lockMutex(m_p);
}

void GlobalMutex::
unlock()
{
  if (m_p)
    Concurrency::getThreadImplementation()->unlockMutex(m_p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
