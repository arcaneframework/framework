// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TBBThreadImplementation.cc                                  (C) 2000-2026 */
/*                                                                           */
/* Implementation of threads using TBB (Intel Threads Building Blocks).      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IThreadImplementationService.h"
#include "arcane/utils/IThreadBarrier.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/utils/Mutex.h"
#include "arcane/utils/PlatformUtils.h"
#include "arccore/base/internal/DependencyInjection.h"

#include "arcane/parallel/thread/ArcaneThreadMisc.h"

#include <tbb/tbb.h>
#define ARCANE_TBB_USE_STDTHREAD
#include <thread>
#include <mutex>
#include <new>

// NOTE:
// This implementation has not been the default implementation since the end of 2025.
// The default implementation now uses the STL.
// If everything is okay, we can delete this implementation at the end of 2026, for example.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef std::thread::id ThreadId;
typedef std::thread ThreadType;
// Attempts to convert a std::thread::id into an 'Int64'.
// There is no portable way to do this, so we are doing something
// that is not necessarily clean. In the long run, it would be preferable to remove
// the IThreadImplementation::currentThread() method.
inline Int64 arcaneGetThisThreadId()
{
  Int64 v = static_cast<Int64>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TBBMutexImpl
{
 public:

  void lock()
  {
    m_mutex.lock();
  }
  void unlock()
  {
    m_mutex.unlock();
  }

 private:

  std::mutex m_mutex;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TBBBarrier
: public IThreadBarrier
{
 public:

  TBBBarrier() = default;

  void destroy() override { delete this; }

  void init(Integer nb_thread) override
  {
    m_nb_thread = nb_thread;
    m_nb_thread_finished = 0;
    m_timestamp = 0;
  };

  void wait() override
  {
    Int32 ts = m_timestamp;
    int remaining_thread = m_nb_thread - m_nb_thread_finished.fetch_add(1) - 1;
    if (remaining_thread > 0) {

      int count = 1;
      while (m_timestamp == ts) {
        arcaneDoCPUPause(count);
        if (count < 200)
          count *= 2;
        else {
          //count = 0;
          //__TBB_Yield();
          //TODO: maybe make the main (__TBB_Yield()) if too
          // iterations.
        }
      }
    }
    m_nb_thread_finished = 0;
    ++m_timestamp;
  }

 private:

  Int32 m_nb_thread = 0;
  std::atomic<Int32> m_nb_thread_finished = 0;
  std::atomic<Int32> m_timestamp = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" IThreadBarrier*
createGlibThreadBarrier();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation of threads using TBB (Intel Threads Building Blocks).
 */
class TBBThreadImplementation
: public IThreadImplementation
, public ReferenceCounterImpl
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

  void addReference() override { ReferenceCounterImpl::addReference(); }
  void removeReference() override { ReferenceCounterImpl::removeReference(); }

 public:

  class StartFunc
  {
   public:

    explicit StartFunc(IFunctor* f)
    : m_f(f)
    {}
    void operator()() const { m_f->executeFunctor(); }
    IFunctor* m_f = nullptr;
  };

 public:

  TBBThreadImplementation()
  {
    if (!platform::getEnvironmentVariable("ARCANE_SPINLOCK_BARRIER").null())
      m_use_tbb_barrier = true;
    m_std_thread_implementation = Arccore::Concurrency::createStdThreadImplementation();
  }

  ~TBBThreadImplementation() override
  {
    //std::cout << "DESTROYING TBB IMPLEMENTATION\n";
    GlobalMutex::destroy();
    if (m_global_mutex_impl)
      this->destroyMutex(m_global_mutex_impl);
  }

 public:

  void build()
  {
  }

  void initialize() override
  {
    m_global_mutex_impl = createMutex();
    GlobalMutex::init(m_global_mutex_impl);
  }

 public:

  ThreadImpl* createThread(IFunctor* f) override
  {
    return reinterpret_cast<ThreadImpl*>(new ThreadType(StartFunc(f)));
  }
  void joinThread(ThreadImpl* t) override
  {
    auto* tt = reinterpret_cast<ThreadType*>(t);
    tt->join();
  }
  void destroyThread(ThreadImpl* t) override
  {
    auto* tt = reinterpret_cast<ThreadType*>(t);
    delete tt;
  }

  void createSpinLock(Int64* spin_lock_addr) override
  {
    void* addr = spin_lock_addr;
    new (addr) tbb::spin_mutex();
  }
  void lockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr) override
  {
    auto* s = reinterpret_cast<tbb::spin_mutex*>(spin_lock_addr);
    auto* sl = new (scoped_spin_lock_addr) tbb::spin_mutex::scoped_lock();
    sl->acquire(*s);
  }
  void unlockSpinLock(Int64* spin_lock_addr, Int64* scoped_spin_lock_addr) override
  {
    ARCANE_UNUSED(spin_lock_addr);
    auto* s = reinterpret_cast<tbb::spin_mutex::scoped_lock*>(scoped_spin_lock_addr);
    s->release();
    //TODO: destroy the scoped_lock.
  }

  MutexImpl* createMutex() override
  {
    auto* m = new TBBMutexImpl();
    return reinterpret_cast<MutexImpl*>(m);
  }
  void destroyMutex(MutexImpl* mutex) override
  {
    auto* tm = reinterpret_cast<TBBMutexImpl*>(mutex);
    delete tm;
  }
  void lockMutex(MutexImpl* mutex) override
  {
    auto* tm = reinterpret_cast<TBBMutexImpl*>(mutex);
    tm->lock();
  }
  void unlockMutex(MutexImpl* mutex) override
  {
    auto* tm = reinterpret_cast<TBBMutexImpl*>(mutex);
    tm->unlock();
  }

  Int64 currentThread() override
  {
    Int64 v = arcaneGetThisThreadId();
    return v;
  }

  IThreadBarrier* createBarrier() override
  {
    // We must use TBB only if requested, because it uses
    // active waiting which can quickly bring the machine to its knees
    // if the total number of threads exceeds the number of cores
    // of the machine.
    if (m_use_tbb_barrier)
      return new TBBBarrier();
    return m_std_thread_implementation->createBarrier();
  }

 private:

  bool m_use_tbb_barrier = false;
  MutexImpl* m_global_mutex_impl = nullptr;
  Ref<IThreadImplementation> m_std_thread_implementation;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TBBThreadImplementationService
: public IThreadImplementationService
{
 public:

  TBBThreadImplementationService() = default;

 public:

  void build() {}

 public:

  Ref<IThreadImplementation> createImplementation() override
  {
    return makeRef<IThreadImplementation>(new TBBThreadImplementation());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DI_REGISTER_PROVIDER(TBBThreadImplementationService,
                            DependencyInjection::ProviderProperty("TBBThreadImplementationService"),
                            ARCANE_DI_INTERFACES(IThreadImplementationService),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
