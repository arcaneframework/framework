// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpinLock.h                                                  (C) 2000-2023 */
/*                                                                           */
/* SpinLock pour le multi-threading.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_SPINLOCK_H
#define ARCCORE_CONCURRENCY_SPINLOCK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/IThreadImplementation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SpinLockImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief SpinLock.
 */
class ARCCORE_CONCURRENCY_EXPORT SpinLock
{
 public:

  class ScopedLock
  {
   public:

    ARCCORE_DEPRECATED_REASON("Y2023: SpinLock are deprecated. Use std::atomic_flag instead")
    ScopedLock(SpinLock& sl)
    {
      spin_lock_addr = &sl.spin_lock;
      Concurrency::getThreadImplementation()->_deprecatedLockSpinLock(spin_lock_addr, scoped_spin_lock);
    }
    ~ScopedLock()
    {
      Concurrency::getThreadImplementation()->_deprecatedUnlockSpinLock(spin_lock_addr, scoped_spin_lock);
    }

   private:

    Int64* spin_lock_addr;
    Int64 scoped_spin_lock[2];
  };

  class ManualLock
  {
   public:

    ARCCORE_DEPRECATED_REASON("Y2023: SpinLock are deprecated. Use std::atomic_flag instead")
    void lock(SpinLock& sl)
    {
      Concurrency::getThreadImplementation()->_deprecatedLockSpinLock(&sl.spin_lock, scoped_spin_lock);
    }
    ARCCORE_DEPRECATED_REASON("Y2023: SpinLock are deprecated. Use std::atomic_flag instead")
    void unlock(SpinLock& sl)
    {
      Concurrency::getThreadImplementation()->_deprecatedUnlockSpinLock(&sl.spin_lock, scoped_spin_lock);
    }

   private:

    Int64 scoped_spin_lock[2];
  };

  friend class ScopedLock;
  friend class ManualLock;

 public:

  ARCCORE_DEPRECATED_REASON("Y2023: SpinLock are deprecated. Use std::atomic_flag instead")
  SpinLock()
  {
    Concurrency::getThreadImplementation()->_deprecatedCreateSpinLock(&spin_lock);
  }

 private:

  Int64 spin_lock = 0;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
