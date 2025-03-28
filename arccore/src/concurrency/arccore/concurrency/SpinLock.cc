// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpinLock.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* SpinLock pour le multi-threading.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/SpinLock.h"

#include "arccore/concurrency/NullThreadImplementation.h"
#include "arccore/concurrency/SpinLock.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include <iostream>

// La fonction atomic_flag::wait() n'est disponible que pour le C++20
#if __cpp_lib_atomic_wait >= 201907L
#define ARCCORE_HAS_ATOMIC_WAIT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SpinLock::
SpinLock()
{
  bool is_need_sync = Concurrency::getThreadImplementation()->isMultiThread();
  m_mode = (is_need_sync) ? eMode::SpinAndMutex : eMode::None;
  _doUnlock();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SpinLock::
SpinLock(eMode mode)
: m_mode(mode)
{
  _doUnlock();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SpinLock::
~SpinLock()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SpinLock::
_doLockReal()
{
  while (m_spin_lock.test_and_set(std::memory_order_acquire)) {
    // L'appel au wait() permet d'eviter la contention
    // si beaucoup de threads essaient d'accéder en même temps
    // au verrou.
    if (m_mode == eMode::SpinAndMutex) {
#ifdef ARCCORE_HAS_ATOMIC_WAIT
      m_spin_lock.wait(true, std::memory_order_relaxed);
#endif
    }
  }
}

void SpinLock::
_doUnlockReal()
{
  m_spin_lock.clear(std::memory_order_release);
#ifdef ARCCORE_HAS_ATOMIC_WAIT
  if (m_mode == eMode::SpinAndMutex) {
    // A noter que notify_one() augmente de 50% le temps passé dans cette
    // fonction. Il n'est utile que si on utilise wait()/
    // Comme wait() n'est intéressant que s'il y a contention, on pourrait
    // rendre cet appel configurable.
    m_spin_lock.notify_one();
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::IThreadImplementation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
