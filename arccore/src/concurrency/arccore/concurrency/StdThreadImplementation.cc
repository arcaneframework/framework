// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StdThreadImplementation.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Implémentation des threads utilisant la bibliothèque standard C++.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/NotImplementedException.h"
#include "arccore/base/IFunctor.h"
#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/Ref.h"

#include "arccore/concurrency/IThreadBarrier.h"
#include "arccore/concurrency/Mutex.h"

#include <new>
#include <thread>
#include <condition_variable>
#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Concurrency
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de ITreadImplementation avec la bibliothèque standard C++.
 */
class ARCCORE_CONCURRENCY_EXPORT StdThreadImplementation
: public IThreadImplementation
, public ReferenceCounterImpl
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  StdThreadImplementation();
  ~StdThreadImplementation() override;

 public:

  void initialize() override;

 public:

  ThreadImpl* createThread(IFunctor* f) override;
  void joinThread(ThreadImpl* t) override;
  void destroyThread(ThreadImpl* t) override;

  void createSpinLock(Int64* spin_lock_addr) override;
  void lockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr) override;
  void unlockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr) override;

  MutexImpl* createMutex() override;
  void destroyMutex(MutexImpl*) override;
  void lockMutex(MutexImpl* mutex) override;
  void unlockMutex(MutexImpl* mutex) override;

  Int64 currentThread() override;

  IThreadBarrier* createBarrier() override;

 public:

  void addReference() override { ReferenceCounterImpl::addReference(); }
  void removeReference() override { ReferenceCounterImpl::removeReference(); }

 private:

  MutexImpl* m_global_mutex_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  void* _StdStartFunc(void* f)
  {
    IFunctor* ff = reinterpret_cast<IFunctor*>(f);
    ff->executeFunctor();
    return 0;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StdThreadBarrier
: public IThreadBarrier
{
 public:

  void init(Integer nb_thread) override
  {
    m_nb_thread = nb_thread;
    m_current_reached = 0;
  }

  void destroy() override
  {
    m_nb_thread = 0;
    m_current_reached = 0;
    delete this;
  }

  bool wait() override
  {
    bool is_last = false;
    {
      std::unique_lock<std::mutex> lk(m_wait_mutex);
      ++m_current_reached;
      //cout << "ADD BARRIER N=" << m_current_reached << '\n';
      if (m_current_reached == m_nb_thread) {
        m_current_reached = 0;
        is_last = true;
        //cout << "BROADCAST BARRIER N=" << m_current_reached << '\n';
        m_wait.notify_all();
      }
      else
        m_wait.wait(lk);
    }
    return is_last;
  }

 private:

  std::mutex m_wait_mutex;
  std::condition_variable m_wait;
  Integer m_nb_thread = 0;
  Integer m_current_reached = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StdThreadImplementation::
StdThreadImplementation()
: m_global_mutex_impl(nullptr)
{
}

StdThreadImplementation::
~StdThreadImplementation()
{
  GlobalMutex::destroy();
  if (m_global_mutex_impl)
    destroyMutex(m_global_mutex_impl);
}

void StdThreadImplementation::
initialize()
{
  m_global_mutex_impl = createMutex();
  GlobalMutex::init(m_global_mutex_impl);
}

ThreadImpl* StdThreadImplementation::
createThread(IFunctor* f)
{
  return reinterpret_cast<ThreadImpl*>(new std::thread(&_StdStartFunc, f));
}

void StdThreadImplementation::
joinThread(ThreadImpl* t)
{
  std::thread* tt = reinterpret_cast<std::thread*>(t);
  tt->join();
}

void StdThreadImplementation::
destroyThread(ThreadImpl* t)
{
  std::thread* tt = reinterpret_cast<std::thread*>(t);
  delete tt;
}

void StdThreadImplementation::
createSpinLock(Int64* spin_lock_addr)
{
  ARCCORE_THROW(NotSupportedException, "Spin lock. Use std::atomic_flag instead()");
}

void StdThreadImplementation::
lockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr)
{
  ARCCORE_THROW(NotSupportedException, "Spin lock. Use std::atomic_flag instead()");
}

void StdThreadImplementation::
unlockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr)
{
  ARCCORE_THROW(NotSupportedException, "Spin lock. Use std::atomic_flag instead()");
}

MutexImpl* StdThreadImplementation::
createMutex()
{
  std::mutex* m = new std::mutex();
  return reinterpret_cast<MutexImpl*>(m);
}

void StdThreadImplementation::
destroyMutex(MutexImpl* mutex)
{
  std::mutex* m = reinterpret_cast<std::mutex*>(mutex);
  delete m;
}

void StdThreadImplementation::
lockMutex(MutexImpl* mutex)
{
  std::mutex* m = reinterpret_cast<std::mutex*>(mutex);
  m->lock();
}

void StdThreadImplementation::
unlockMutex(MutexImpl* mutex)
{
  std::mutex* m = reinterpret_cast<std::mutex*>(mutex);
  m->unlock();
}

Int64 StdThreadImplementation::
currentThread()
{
  Int64 v = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return v;
}

IThreadBarrier* StdThreadImplementation::
createBarrier()
{
  return new StdThreadBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IThreadImplementation>
createStdThreadImplementation()
{
  return makeRef<IThreadImplementation>(new Concurrency::StdThreadImplementation());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::Concurrency

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
