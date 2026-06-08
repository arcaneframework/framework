// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpinLock.h                                                  (C) 2000-2025 */
/*                                                                           */
/* SpinLock for multi-threading.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_SPINLOCK_H
#define ARCCORE_CONCURRENCY_SPINLOCK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/IThreadImplementation.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Add enumeration for spin_lock management (None, FullSpin, Spin+Wait)

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

    ScopedLock(SpinLock& sl)
    : m_spin_lock_ref(sl)
    {
      m_spin_lock_ref._doLock();
    }
    ~ScopedLock()
    {
      m_spin_lock_ref._doUnlock();
    }

   private:

    SpinLock& m_spin_lock_ref;
  };

  class ManualLock
  {
   public:

    void lock(SpinLock& sl)
    {
      sl._doLock();
    }
    void unlock(SpinLock& sl)
    {
      sl._doUnlock();
    }
  };

  friend class ScopedLock;
  friend class ManualLock;

 public:

  //! Spinlock mode. The default is 'Auto'
  enum class eMode : uint8_t
  {
    // No synchronization
    None,
    /*!
     * \brief Automatic choice.
     *
     * If `Concurrency::getThreadImplementation()->isMultiThread()` is true,
     * then the mode is SpinAndMutex. Otherwise, the mode is None.
     */
    Auto,
    /*!
     * \brief Always uses a spinlock.
     *
     * This type is faster if there is very little contention, but performance
     * is very poor otherwise.
     */
    FullSpin,
    /*!
     * \brief SpinLock then mutex.
     *
     * Performs a spinlock then yields (std::this_thread::yield())
     * if it takes too long. This mode is only available if C++20 is used.
     * Otherwise, it is equivalent to FullSpin.
     */
    SpinAndMutex
  };

 public:

  //! Default SpinLock
  SpinLock();
  //! SpinLock with the \a mode
  SpinLock(eMode mode);
  ~SpinLock();

 private:

  std::atomic_flag m_spin_lock = ATOMIC_FLAG_INIT;
  eMode m_mode = eMode::SpinAndMutex;

 private:

  void _doLock()
  {
    if (m_mode != eMode::None)
      _doLockReal();
  }
  void _doUnlock()
  {
    if (m_mode != eMode::None)
      _doUnlockReal();
  }

 private:

  // TODO: make these functions inline when C++20 is available everywhere.
  // For now, we cannot do this because of the ODR.
  void _doLockReal();
  void _doUnlockReal();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
