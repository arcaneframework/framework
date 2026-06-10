// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IThreadImplementation.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface of a service implementing thread support.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_ITHREADIMPLEMENTATION_H
#define ARCCORE_CONCURRENCY_ITHREADIMPLEMENTATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"
#include "arccore/base/BaseTypes.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a service implementing thread support.
 *
 * This service uses a reference counter and must be destroyed
 * if there are no more references to it. It should therefore generally not
 * be destroyed explicitly.
 */
class ARCCORE_CONCURRENCY_EXPORT IThreadImplementation
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 private:

  friend class SpinLock;
  friend class ScopedLock;
  friend class ManualLock;

 protected:

  virtual ~IThreadImplementation() = default;

 public:

  ARCCORE_DEPRECATED_REASON("Y2023: Use Ref<IThreadImplementation> to manage references")
  virtual void addReference() = 0;
  ARCCORE_DEPRECATED_REASON("Y2023: Use Ref<IThreadImplementation> to manage references")
  virtual void removeReference() = 0;

 public:

  virtual void initialize() = 0;

 public:

  virtual ThreadImpl* createThread(IFunctor* f) = 0;
  virtual void joinThread(ThreadImpl* t) = 0;
  virtual void destroyThread(ThreadImpl* t) = 0;

  ARCCORE_DEPRECATED_REASON("Y2023: SpinLock are deprecated. Use std::atomic_flag instead")
  virtual void createSpinLock(Int64* spin_lock_addr) = 0;
  ARCCORE_DEPRECATED_REASON("Y2023: SpinLock are deprecated. Use std::atomic_flag instead")
  virtual void lockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr) = 0;
  ARCCORE_DEPRECATED_REASON("Y2023: SpinLock are deprecated. Use std::atomic_flag instead")
  virtual void unlockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr) = 0;

  virtual MutexImpl* createMutex() = 0;
  virtual void destroyMutex(MutexImpl*) = 0;
  virtual void lockMutex(MutexImpl* mutex) = 0;
  virtual void unlockMutex(MutexImpl* mutex) = 0;

  virtual Int64 currentThread() = 0;

  virtual IThreadBarrier* createBarrier() = 0;

  /*!
   * \brief True if the implementation supports multiple threads.
   *
   * In single-thread mode, only one thread executes. Therefore, there is no
   * need to create synchronization management classes such as Mutexes or SpinLocks.
   */
  virtual bool isMultiThread() const { return true; }

 private:

  // Definitions to avoid displaying warnings
  // due to deprecated methods.

  void _deprecatedCreateSpinLock(Int64* spin_lock_addr);
  void _deprecatedLockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr);
  void _deprecatedUnlockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
