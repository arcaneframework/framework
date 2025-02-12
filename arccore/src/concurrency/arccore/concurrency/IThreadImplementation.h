// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IThreadImplementation.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service implémentant le support des threads.               */
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
 * \brief Interface d'un service implémentant le support des threads.
 *
 * Ce service utilise un compteur de référence et doit être détruit
 * s'il n'y a plus de références dessus. Il ne doit donc en général pas
 * être détruit explicitement.
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
   * \brief Vrai si l'implémentation supporte plusieurs threads.
   *
   * En mono-thread, seul un thread s'exécute. Il n'y a donc pas
   * besoin de créer les classes gérant la synchronisation telles que les
   * Mutex ou les SpinLock.
   */
  virtual bool isMultiThread() const { return true; }

 private:

  // Définitions pour éviter d'affichier les messages d'avertissement
  // à cause des méthodes obsolètes.

  void _deprecatedCreateSpinLock(Int64* spin_lock_addr);
  void _deprecatedLockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr);
  void _deprecatedUnlockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
