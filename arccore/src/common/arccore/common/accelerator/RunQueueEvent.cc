// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueEvent.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Evènement sur une file d'exécution.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/RunQueueEvent.h"

#include "arccore/base/FatalErrorException.h"

#include "arccore/common/accelerator/internal/IRunQueueEventImpl.h"
#include "arccore/common/accelerator/internal/RunnerImpl.h"
#include "arccore/common/accelerator/Runner.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class RunQueueEvent::InternalImpl
{
 public:

  explicit InternalImpl(Impl::IRunQueueEventImpl* p)
  : m_impl(p)
  {}
  ~InternalImpl()
  {
    delete m_impl;
  }

 public:

  void addRef()
  {
    ++m_nb_ref;
  }
  void removeRef()
  {
    Int32 v = std::atomic_fetch_add(&m_nb_ref, -1);
    if (v == 1)
      delete this;
  }

 public:

  Impl::IRunQueueEventImpl* m_impl = nullptr;

 private:

  //! Nombre de références sur l'instance.
  std::atomic<Int32> m_nb_ref = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueEvent::
RunQueueEvent()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueEvent::
RunQueueEvent(const Runner& runner)
: m_p(new InternalImpl(runner._impl()->_createEvent()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueEvent::
RunQueueEvent(const RunQueueEvent& x)
: m_p(x.m_p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueEvent::
RunQueueEvent(RunQueueEvent&& x) noexcept
: m_p(x.m_p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueEvent& RunQueueEvent::
operator=(const RunQueueEvent& x)
{
  if (&x != this)
    m_p = x.m_p;
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueEvent& RunQueueEvent::
operator=(RunQueueEvent&& x) noexcept
{
  m_p = x.m_p;
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueEvent::
~RunQueueEvent()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueEvent::
wait()
{
  if (m_p)
    m_p->m_impl->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RunQueueEvent::
hasPendingWork() const
{
  if (m_p)
    return m_p->m_impl->hasPendingWork();
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Impl::IRunQueueEventImpl* RunQueueEvent::
_internalEventImpl() const
{
  ARCCORE_FATAL_IF((!m_p),"Invalid usage of null RunQueueEvent");
  return m_p->m_impl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
