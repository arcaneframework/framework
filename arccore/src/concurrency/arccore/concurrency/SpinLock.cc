// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpinLock.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* SpinLock for multi-threading.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/SpinLock.h"

#include "arccore/concurrency/NullThreadImplementation.h"
#include "arccore/concurrency/SpinLock.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include <iostream>

// The atomic_flag::wait() function is only available for C++20
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
    // Calling wait() prevents contention
    // if many threads try to access
    // the lock at the same time.
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
    // Note that notify_one() increases the time spent in this
    // function by 50%. It is only useful if wait() is used.
    // Since wait() is only useful if there is contention, this
    // call could be made configurable.
    m_spin_lock.notify_one();
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::IThreadImplementation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
