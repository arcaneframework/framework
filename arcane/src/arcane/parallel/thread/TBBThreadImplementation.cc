// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TBBThreadImplementation.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Implémentation des threads utilisant TBB (Intel Threads Building Blocks). */
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
#if TBB_VERSION_MAJOR >= 2020
#define ARCANE_TBB_USE_STDTHREAD
#include <thread>
#else
#include <tbb/tbb_thread.h>
#include <tbb/atomic.h>
#include <tbb/mutex.h>
#endif

#include <mutex>
#include <new>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_TBB_USE_STDTHREAD
typedef std::thread::id ThreadId;
typedef std::thread ThreadType;
// Essaie de convertir un std::thread::id en un 'Int64'.
// Il n'existe pas de moyen portable de le faire donc on fait quelque
// chose de pas forcément propre. A terme il serait préférable de supprimer
// la méthode IThreadImplementation::currentThread().
inline Int64 arcaneGetThisThreadId()
{
  Int64 v = std::hash<std::thread::id>{}(std::this_thread::get_id());
  return v;
}
#else
struct ThreadId
{
 public:
#if defined(_WIN32) || defined(_WIN64)
  DWORD my_id;
#else
  pthread_t my_id;
#endif // _WIN32||_WIN64
};

typedef tbb::tbb_thread ThreadType;
inline Int64 arcaneGetThisThreadId()
{
  ThreadType::id i = tbb::this_tbb_thread::get_id();
  ThreadId* t = (ThreadId*)(&i);
  Int64 v = Int64(t->my_id);
  return v;
}
#endif

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
  TBBBarrier()
  : m_nb_thread(0) {}

  virtual void destroy(){ delete this; }

  virtual void init(Integer nb_thread)
  {
    m_nb_thread = nb_thread;
    m_nb_thread_finished = 0;
    m_timestamp = 0;
  };

  virtual bool wait()
  {
    Int32 ts = m_timestamp;
    int remaining_thread = m_nb_thread - m_nb_thread_finished.fetch_add(1) - 1;
    if (remaining_thread > 0) {

      int count = 1;
      while (m_timestamp==ts){
        arcaneDoCPUPause(count);
        if (count<200)
          count *= 2;
        else{
          //count = 0;
          //__TBB_Yield();
          //TODO: peut-être rendre la main (__TBB_Yield()) si trop
        // d'itérations.
        }
      }

      return false;
    }
    m_nb_thread_finished = 0;
    ++m_timestamp;
    return true;
  }
 private:
  Int32 m_nb_thread;
  std::atomic<Int32> m_nb_thread_finished;
  std::atomic<Int32> m_timestamp;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" IThreadBarrier*
createGlibThreadBarrier();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation des threads utilisant TBB (Intel Threads Building Blocks).
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
    StartFunc(IFunctor* f) : m_f(f){}
    void operator()() { m_f->executeFunctor(); }
    IFunctor* m_f;
  };

 public:

  TBBThreadImplementation()
  : m_use_tbb_barrier(false)
  , m_global_mutex_impl(nullptr)
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
    ThreadType* tt = reinterpret_cast<ThreadType*>(t);
    tt->join();
  }
  void destroyThread(ThreadImpl* t) override
  {
    ThreadType* tt = reinterpret_cast<ThreadType*>(t);
    delete tt;
  }

  void createSpinLock(Int64* spin_lock_addr) override
  {
    void* addr = spin_lock_addr;
    new (addr) tbb::spin_mutex();
  }
  void lockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr) override
  {
    tbb::spin_mutex* s = reinterpret_cast<tbb::spin_mutex*>(spin_lock_addr);
    tbb::spin_mutex::scoped_lock* sl = new (scoped_spin_lock_addr) tbb::spin_mutex::scoped_lock();
    sl->acquire(*s);
  }
  void unlockSpinLock(Int64* spin_lock_addr,Int64* scoped_spin_lock_addr) override
  {
    ARCANE_UNUSED(spin_lock_addr);
    tbb::spin_mutex::scoped_lock* s = reinterpret_cast<tbb::spin_mutex::scoped_lock*>(scoped_spin_lock_addr);
    s->release();
    //TODO: detruire le scoped_lock.
  }
  
  MutexImpl* createMutex() override
  {
    TBBMutexImpl* m = new TBBMutexImpl();
    return reinterpret_cast<MutexImpl*>(m);
  }
  void destroyMutex(MutexImpl* mutex) override
  {
    TBBMutexImpl* tm = reinterpret_cast<TBBMutexImpl*>(mutex);
    delete tm;
  }
  void lockMutex(MutexImpl* mutex) override
  {
    TBBMutexImpl* tm = reinterpret_cast<TBBMutexImpl*>(mutex);
    tm->lock();
  }
  void unlockMutex(MutexImpl* mutex) override
  {
    TBBMutexImpl* tm = reinterpret_cast<TBBMutexImpl*>(mutex);
    tm->unlock();
  }

  Int64 currentThread() override
  {
    Int64 v = arcaneGetThisThreadId();
    return v;
  }

  IThreadBarrier* createBarrier() override
  {
    // Il faut utiliser les TBB uniquement si demandé car il utilise
    // l'attente active ce qui peut vite mettre la machine à genoux
    // si le nombre de thread total est supérieur au nombre de coeurs
    // de la machine.
    if (m_use_tbb_barrier)
      return new TBBBarrier();
    return m_std_thread_implementation->createBarrier();
  }

 private:

  bool m_use_tbb_barrier;
  MutexImpl* m_global_mutex_impl;
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
