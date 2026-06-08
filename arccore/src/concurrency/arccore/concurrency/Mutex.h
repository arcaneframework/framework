// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Mutex.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Mutex for multi-threading.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_MUTEX_H
#define ARCCORE_CONCURRENCY_MUTEX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"
#include "arccore/concurrency/IThreadImplementation.h"
#include "arccore/base/ReferenceCounter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mutex.
 */
class ARCCORE_CONCURRENCY_EXPORT Mutex
{
 public:
  class ScopedLock
  {
   public:
    ScopedLock(Mutex& m)
    : m_p(&m)
    {
      m_p->lock();
    }
    ScopedLock(Mutex* m)
    : m_p(m)
    {
      m_p->lock();
    }
    ~ScopedLock()
    {
      m_p->unlock();
    }
   private:
    Mutex* m_p;
  };

  class ManualLock
  {
   public:
    void lock(Mutex& m)
    {
      m.lock();
    }
    void unlock(Mutex& m)
    {
      m.unlock();
    }
   private:
  };

  friend class ScopedLock;
  friend class ManualLock;

 public:
  Mutex();
  ~Mutex();
 private:
  void lock();
  void unlock();
 private:
  MutexImpl* m_p;
  //! Implementation used for this mutex.
  ReferenceCounter<IThreadImplementation> m_thread_impl;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Global Mutex.
 */
class ARCCORE_CONCURRENCY_EXPORT GlobalMutex
{
 public:
  class ScopedLock
  {
   public:
    ScopedLock()
    {
      GlobalMutex::lock();
    }
    ~ScopedLock()
    {
      GlobalMutex::unlock();
    }
  };
  friend class ScopedLock;
 public:
  GlobalMutex(){}
  ~GlobalMutex() {}
 public:
  //! Initializes the global mutex. Internal to Arccore. Must be allocated by new
  static void init(MutexImpl* p);
  static void lock();
  static void unlock();
  static void destroy();
 private:
  static MutexImpl* m_p;
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
