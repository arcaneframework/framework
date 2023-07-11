// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueUnitTest.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Service de test unitaire des 'RunQueue'.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueueEvent.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLoop.h"

#include <thread>
#include <chrono>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;
namespace ax = Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la classe 'NumArray'.
 */
class RunQueueUnitTest
: public BasicUnitTest
{
 public:

  explicit RunQueueUnitTest(const ServiceBuildInfo& cb);
  ~RunQueueUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner* m_runner = nullptr;

 public:

  void _executeTest1(bool use_priority);
  void _executeTest2();
  void _executeTest3();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(RunQueueUnitTest, IUnitTest, RunQueueUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueUnitTest::
RunQueueUnitTest(const ServiceBuildInfo& sb)
: BasicUnitTest(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunQueueUnitTest::
~RunQueueUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueUnitTest::
initializeTest()
{
  m_runner = subDomain()->acceleratorMng()->defaultRunner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueUnitTest::
executeTest()
{
  _executeTest2();
  bool old_v = m_runner->isConcurrentQueueCreation();
  m_runner->setConcurrentQueueCreation(true);
  _executeTest1(false);
  _executeTest1(true);
  _executeTest3();
  m_runner->setConcurrentQueueCreation(old_v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueUnitTest::
_executeTest1(bool use_priority)
{
  info() << "Test RunQueue with multiple threads use_priority=" << use_priority;

  Integer nb_thread = 8;
  Integer N = 1000000;

  UniqueArray<NumArray<Int32, MDDim1>> values(8);
  for (Integer i = 0; i < nb_thread; ++i)
    values[i].resize(N);

  auto task_func = [&](Ref<RunQueue> q, int id) {
    info() << "EXECUTE_THREAD_ID=" << id;
    info(4) << "Queue pointer=" << q->platformStream();
    auto command1 = makeCommand(q.get());
    auto v = viewOut(command1, values[id]);
    command1 << RUNCOMMAND_LOOP1(iter, N)
    {
      auto [i] = iter();
      v(iter) = (int)math::sqrt((double)i);
    };
    q->barrier();
  };

  UniqueArray<std::thread*> allthreads;

  for (Integer i = 0; i < nb_thread; ++i) {
    ax::RunQueueBuildInfo bi;
    if (use_priority && (i > 3))
      bi.setPriority(-8);
    auto queue_ref = makeQueueRef(*m_runner, bi);
    queue_ref->setAsync(true);
    allthreads.add(new std::thread(task_func, queue_ref, i));
  }
  for (auto thr : allthreads) {
    thr->join();
    delete thr;
  }

  Int64 true_total = 0;
  Int64 expected_true_total = 0;
  for (Integer i = 0; i < nb_thread; ++i) {
    for (Integer j = 0; j < N; ++j) {
      true_total += values[i](j);
      expected_true_total += (int)math::sqrt((double)j);
    }
  }
  info() << "End TestCudaThread TOTAL=" << true_total
         << " expected=" << expected_true_total;
  if (true_total != expected_true_total)
    ARCANE_FATAL("Bad value v={0} expected={1}", true_total, expected_true_total);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Test la synchronisation entre deux RunQueue par un évènement.
void RunQueueUnitTest::
_executeTest2()
{
  info() << "Test2: use events";
  ValueChecker vc(A_FUNCINFO);

  auto event{ makeEvent(*m_runner) };
  auto queue1{ makeQueue(*m_runner) };
  queue1.setAsync(true);
  auto queue2{ makeQueue(*m_runner) };
  queue2.setAsync(true);

  Integer nb_value = 100000;
  NumArray<Int32, MDDim1> values(nb_value);
  {
    auto command1 = makeCommand(queue1);
    auto v = viewOut(command1, values);
    command1 << RUNCOMMAND_LOOP1 (iter, nb_value)
    {
      auto [i] = iter();
      v(iter) = i + 3;
    };
    queue1.recordEvent(event);
  }
  {
    queue2.waitEvent(event);
    auto command2 = makeCommand(queue2);
    auto v = viewInOut(command2, values);
    command2 << RUNCOMMAND_LOOP1 (iter, nb_value)
    {
      v(iter) = v(iter) * 2;
    };
  }
  queue1.barrier();
  queue2.barrier();

  // Vérifie les valeurs
  for (Integer i = 0; i < nb_value; ++i) {
    Int32 v = values(i);
    Int32 expected_v = (i + 3) * 2;
    vc.areEqual(v, expected_v, "Bad value");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Test la synchronisation de avec un évènement.
void RunQueueUnitTest::
_executeTest3()
{
  info() << "Test3: use events with wait()";
  ValueChecker vc(A_FUNCINFO);

  UniqueArray<Ref<ax::RunQueueEvent>> event_array;
  event_array.add(makeEventRef(*m_runner));

  auto queue1{ makeQueue(*m_runner) };
  queue1.setAsync(true);
  auto queue2{ makeQueue(*m_runner) };
  queue2.setAsync(true);

  Integer nb_value = 100000;
  NumArray<Int32, MDDim1> values(nb_value);
  {
    auto command1 = makeCommand(queue1);
    auto v = viewOut(command1, values);
    command1 << RUNCOMMAND_LOOP1 (iter, nb_value)
    {
      auto [i] = iter();
      v(iter) = i + 3;
    };
    queue1.recordEvent(event_array[0]);
  }
  event_array[0]->wait();
  {
    auto command2 = makeCommand(queue2);
    auto v = viewInOut(command2, values);
    command2 << RUNCOMMAND_LOOP1 (iter, nb_value)
    {
      v(iter) = v(iter) * 2;
    };
    queue2.recordEvent(event_array[0]);
  }
  event_array[0]->wait();

  // Vérifie les valeurs
  for (Integer i = 0; i < nb_value; ++i) {
    Int32 v = values(i);
    Int32 expected_v = (i + 3) * 2;
    vc.areEqual(v, expected_v, "Bad value");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
