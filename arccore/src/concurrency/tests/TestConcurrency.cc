// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/PlatformUtils.h"
#include "arccore/base/Functor.h"
#include "arccore/base/Ref.h"

#include "arccore/concurrency/SpinLock.h"
#include "arccore/concurrency/Mutex.h"
#include "arccore/concurrency/IThreadBarrier.h"

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MyThread
{
 public:

  MyThread() = default;
  explicit MyThread(IFunctor* f)
  {
    create(f);
  }
  ~MyThread()
  {
    if (m_thread)
      Concurrency::getThreadImplementation()->destroyThread(m_thread);
  }
  void create(IFunctor* f)
  {
    m_functor = f;
    m_thread = Concurrency::getThreadImplementation()->createThread(f);
  }
  void join()
  {
    Concurrency::getThreadImplementation()->joinThread(m_thread);
  }

 public:

  IFunctor* m_functor = nullptr;
  ThreadImpl* m_thread = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TestSpinLock1
{
 public:

  TestSpinLock1()
  {
  }

  explicit TestSpinLock1(SpinLock::eMode mode)
  : m_lock(mode)
  {
  }

  void exec()
  {
    Real v1 = Platform::getRealTime();
    const Int32 nb_iter = 70;
    const Int32 nb_thread = 10;
    for (Integer i = 0; i < nb_iter; ++i) {
      FunctorT<TestSpinLock1> f1(this, &TestSpinLock1::_F1);
      std::vector<MyThread> threads(nb_thread);
      for (Integer j = 0; j < nb_thread; ++j)
        threads[j].create(&f1);
      for (Integer j = 0; j < nb_thread; ++j)
        threads[j].join();
    }
    Real v2 = Platform::getRealTime();
    std::cout << "Test1 spin_time=" << (v2 - v1) << " count2=" << m_count2 << " count3=" << m_count3 << "\n";
    Int64 expected_count3 = nb_iter * m_nb_sub_iter * nb_thread;
    Int64 expected_count2 = 10 * expected_count3 + (expected_count3 * (expected_count3 + 1)) / 2;
    std::cout << " expected_count2=" << expected_count2 << " expected_count3=" << expected_count3 << "\n";
    ASSERT_EQ(m_count2, expected_count2);
    ASSERT_EQ(m_count3, expected_count3);
  }
  void _F1()
  {
    for (Int32 i = 0; i < m_nb_sub_iter; ++i) {
      ++m_count;
      {
        SpinLock::ScopedLock sl(m_lock);
        ++m_count3;
        m_count2 += 10;
        m_count2 += m_count3;
      }
    }
  }
  SpinLock m_lock;
  std::atomic<Int64> m_count = 0;
  Int64 m_count2 = 0;
  Int64 m_count3 = 0;
  Int32 m_nb_sub_iter = 1000;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TestMutexLock1
{
 public:

  TestMutexLock1()
  : m_thread_implementation(Concurrency::getThreadImplementation())
  {
  }

  void exec()
  {
    Real v1 = Platform::getRealTime();
    const Int32 nb_iter = 70;
    const Int32 nb_thread = 10;
    m_thread_barrier = m_thread_implementation->createBarrier();
    m_thread_barrier->init(nb_thread);
    for (Integer i = 0; i < nb_iter; ++i) {
      FunctorT<TestMutexLock1> f1(this, &TestMutexLock1::_F1);
      std::vector<MyThread> threads(nb_thread);
      for (Integer j = 0; j < nb_thread; ++j)
        threads[j].create(&f1);
      for (Integer j = 0; j < nb_thread; ++j)
        threads[j].join();
      m_do_print_id = false;
    }
    Real v2 = Platform::getRealTime();
    Int64 nb_sub_iter = m_nb_sub_iter;
    std::cout << "Test1 spin_time=" << (v2 - v1) << " count2=" << m_count2 << " count3=" << m_count3 << "\n";
    Int64 i64_nb_thread = nb_thread;
    Int64 expected_count3 = nb_iter * nb_sub_iter * i64_nb_thread;
    Int64 expected_count2 = 10 * expected_count3 + (expected_count3 * (expected_count3 + 1)) / 2;
    std::cout << " expected_count2=" << expected_count2 << " expected_count3=" << expected_count3 << "\n";
    ASSERT_EQ(m_count2, expected_count2);
    ASSERT_EQ(m_count3, expected_count3);
    m_thread_barrier->destroy();
  }
  void _F1()
  {
    for (Int32 i = 0; i < m_nb_sub_iter; ++i) {
      ++m_count;
      {
        Mutex::ScopedLock sl(m_mutex);
        ++m_count3;
        m_count2 += 10;
        m_count2 += m_count3;
      }
    }
    m_thread_barrier->wait();
    if (m_do_print_id) {
      std::ostringstream ostr;
      ostr << "THREAD_ID=" << m_thread_implementation->currentThread() << "\n";
      std::cout << ostr.str();
    }
  }
  IThreadImplementation* m_thread_implementation = nullptr;
  Mutex m_mutex;
  IThreadBarrier* m_thread_barrier = nullptr;
  std::atomic<Int64> m_count = 0;
  Int64 m_count2 = 0;
  Int64 m_count3 = 0;
  Int32 m_nb_sub_iter = 1000;
  bool m_do_print_id = true;
};

namespace
{
void _doSpinLock()
{

  {
    TestSpinLock1 test1;
    test1.exec();
  }
  {
    TestSpinLock1 test2(SpinLock::eMode::FullSpin);
    test2.exec();
  }
  {
    TestSpinLock1 test3(SpinLock::eMode::SpinAndMutex);
    test3.exec();
  }
}
void _doMutexLock()
{

  {
    TestMutexLock1 test1;
    test1.exec();
  }
}
} // namespace

TEST(Concurrency, GlibSpinLock)
{
  Ref<IThreadImplementation> timpl(Concurrency::createGlibThreadImplementation());
  Concurrency::setThreadImplementation(timpl.get());
  _doSpinLock();
  Concurrency::setThreadImplementation(nullptr);
}

TEST(Concurrency, StdSpinLock)
{
  Ref<IThreadImplementation> timpl(Concurrency::createStdThreadImplementation());
  Concurrency::setThreadImplementation(timpl.get());
  _doSpinLock();
  Concurrency::setThreadImplementation(nullptr);
}

TEST(Concurrency, GlibMutexLock)
{
  Ref<IThreadImplementation> timpl(Concurrency::createGlibThreadImplementation());
  Concurrency::setThreadImplementation(timpl.get());
  _doMutexLock();
  Concurrency::setThreadImplementation(nullptr);
}

TEST(Concurrency, StdMutexLock)
{
  Ref<IThreadImplementation> timpl(Concurrency::createStdThreadImplementation());
  Concurrency::setThreadImplementation(timpl.get());
  _doMutexLock();
  Concurrency::setThreadImplementation(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
