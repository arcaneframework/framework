// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NullThreadImplementation.h                                  (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de thread en mode mono-thread.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_NULLTHREADIMPLEMENTATION_H
#define ARCCORE_CONCURRENCY_NULLTHREADIMPLEMENTATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/IThreadBarrier.h"
#include "arccore/concurrency/IThreadImplementation.h"

#include "arccore/base/ReferenceCounterImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation d'une barrière en mono-thread.
 */
class ARCCORE_CONCURRENCY_EXPORT NullThreadBarrier
: public IThreadBarrier
{
  void init(Integer nb_thread) override { ARCCORE_UNUSED(nb_thread); }
  void destroy() override { delete this; }
  bool wait() override { return true; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation des threads en mode mono-thread.
 */
class ARCCORE_CONCURRENCY_EXPORT NullThreadImplementation
: public IThreadImplementation
, public ReferenceCounterImpl
{
  // Pour créer dynamiquement les instances
  friend class NullThreadImplementationFactory;

 private:

  // TODO Utiliser ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS()
  // quand il n'y a aura plus d'instances statiques de cette classe.
  ReferenceCounterImpl* _internalReferenceCounter() override { return this; }
  void _internalAddReference() override
  {
    if (m_do_destroy)
      Arccore::ReferenceCounterImpl::_internalAddReference();
  }
  bool _internalRemoveReference() override
  {
    if (m_do_destroy)
      return Arccore::ReferenceCounterImpl::_internalRemoveReference();
    return false;
  }

 public:

  void addReference() override { _internalAddReference(); }
  void removeReference() override { _internalRemoveReference(); }

 public:

  ARCCORE_DEPRECATED_REASON("Y2023: This constructor is internal to Arcane. Use Concurrency::createNullThreadImplementation() instead")
  NullThreadImplementation()
  : m_do_destroy(false)
  {}

 public:

  void initialize() override {}
  ThreadImpl* createThread(IFunctor*) override { return nullptr; }
  void joinThread(ThreadImpl*) override {}
  void destroyThread(ThreadImpl*) override {}

  void createSpinLock(Int64* spin_lock_addr) override
  {
    ARCCORE_UNUSED(spin_lock_addr);
  }
  void lockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr) override
  {
    ARCCORE_UNUSED(spin_lock_addr);
    ARCCORE_UNUSED(scoped_spin_lock_addr);
  }
  void unlockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr) override
  {
    ARCCORE_UNUSED(spin_lock_addr);
    ARCCORE_UNUSED(scoped_spin_lock_addr);
  }

  MutexImpl* createMutex() override { return nullptr; }
  void destroyMutex(MutexImpl*) override {}
  void lockMutex(MutexImpl*) override {}
  void unlockMutex(MutexImpl*) override {}

  Int64 currentThread() override { return 0; }

  IThreadBarrier* createBarrier() override { return new NullThreadBarrier(); }

  bool isMultiThread() const override { return false; }

 private:

  // Constructeur utilisé par NullThreadImplementationFactory qui oblige à créer via 'new'
  NullThreadImplementation(bool)
  : m_do_destroy(true)
  {}

 private:

  bool m_do_destroy = true;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
