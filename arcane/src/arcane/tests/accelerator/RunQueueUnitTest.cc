// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueUnitTest.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Service de test unitaire des 'RunQueue'.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueueEvent.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/internal/RunQueueImpl.h"

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/SpanViews.h"
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

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::Runner m_runner;

 public:

  void _executeTestNullQueue();
  void _executeTest1(bool use_priority);
  void _executeTest2();
  void _executeTest3(bool use_pooling);
  void _executeTest4();
  void _executeTest5();
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueUnitTest::
initializeTest()
{
  m_runner = subDomain()->acceleratorMng()->runner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueUnitTest::
executeTest()
{
  _executeTestNullQueue();
  _executeTest2();
  _executeTest1(false);
  _executeTest1(true);
  _executeTest3(false);
  if (m_runner.executionPolicy() != ax::eExecutionPolicy::SYCL)
    _executeTest3(true);
  _executeTest4();
  _executeTest5();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueUnitTest::
_executeTestNullQueue()
{
  using namespace Arcane::Accelerator;
  ValueChecker vc(A_FUNCINFO);
  MemoryAllocationOptions default_mem_opt;
  RunQueue queue;
  vc.areEqual(queue.isNull(), true, "isNull()");
  queue.barrier();
  vc.areEqual(queue.executionPolicy(), eExecutionPolicy::None, "executionPolicy()");
  vc.areEqual(queue.isAcceleratorPolicy(), false, "isAcceleratorPolicy()");
  vc.areEqual(queue.memoryRessource(), eMemoryRessource::Unknown, "memoryRessource()");
  if (queue.allocationOptions() != default_mem_opt)
    ARCANE_FATAL("Bad null allocationOptions()");

  queue = makeQueue(m_runner);
  vc.areEqual(queue.isNull(), false, "not null");

  queue = RunQueue();
  vc.areEqual(queue.isNull(), true, "is null (2)");

  queue = makeQueue(m_runner);
  if (queue.executionPolicy() == eExecutionPolicy::None)
    ARCANE_FATAL("Bad execution policy");
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
    auto queue_ref = makeQueueRef(m_runner, bi);
    queue_ref->setAsync(true);
    allthreads.add(new std::thread(task_func, queue_ref, i));
  }
  for (auto thr : allthreads) {
    thr->join();
    delete thr;
  }
  info() << "End of wait";

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

  auto event{ makeEvent(m_runner) };
  auto queue1{ makeQueue(m_runner) };
  queue1.setAsync(true);
  auto queue2{ makeQueue(m_runner) };
  queue2.setAsync(true);

  Integer nb_value = 100000;
  NumArray<Int32, MDDim1> values(nb_value);
  {
    auto command1 = makeCommand(queue1);
    auto v = viewOut(command1, values);
    command1 << RUNCOMMAND_LOOP1(iter, nb_value)
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
    command2 << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      v(iter) = v(iter) * 2;
    };
  }

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
// Teste la synchronisation avec un évènement.
void RunQueueUnitTest::
_executeTest3(bool use_pooling)
{
  info() << "Test3: use events with wait() or pooling is_pooling?=" << use_pooling;
  ValueChecker vc(A_FUNCINFO);

  UniqueArray<Ref<ax::RunQueueEvent>> event_array;
  event_array.add(makeEventRef(m_runner));

  auto queue1{ makeQueue(m_runner) };
  queue1.setAsync(true);
  auto queue2{ makeQueue(m_runner) };
  queue2.setAsync(true);

  Integer nb_value = 100000;
  NumArray<Int32, MDDim1> values(nb_value);
  {
    auto command1 = makeCommand(queue1);
    auto v = viewOut(command1, values);
    command1 << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      v(iter) = i + 3;
    };
    queue1.recordEvent(event_array[0]);
  }
  if (use_pooling)
    while (event_array[0]->hasPendingWork()) {
      // Do something ...
    }
  else
    event_array[0]->wait();
  {
    auto command2 = makeCommand(queue2);
    auto v = viewInOut(command2, values);
    command2 << RUNCOMMAND_LOOP1(iter, nb_value)
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
// Test la synchronisation de avec un évènement et sémantique par référence.
void RunQueueUnitTest::
_executeTest4()
{
  info() << "Test4: use events with wait()";

  {
    Arcane::Accelerator::RunQueueEvent event0;
    if (!event0.isNull())
      ARCANE_FATAL("Event is not null");
    event0 = makeEvent(m_runner);
    if (event0.isNull())
      ARCANE_FATAL("Event is null");
    Arcane::Accelerator::RunQueueEvent event1(event0);
    if (event1.isNull())
      ARCANE_FATAL("Event is null");
  }

  ValueChecker vc(A_FUNCINFO);
  //![SampleRunQueueEventSample1]
  Arcane::Accelerator::Runner runner = m_runner;

  Arcane::Accelerator::RunQueueEvent event(makeEvent(runner));

  Arcane::Accelerator::RunQueue queue1{ makeQueue(runner) };
  queue1.setAsync(true);
  Arcane::Accelerator::RunQueue queue2{ makeQueue(runner) };
  queue2.setAsync(true);

  Integer nb_value = 100000;
  Arcane::NumArray<Int32, MDDim1> values(nb_value);
  {
    auto command1 = makeCommand(queue1);
    auto v = viewOut(command1, values);
    command1 << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      v(iter) = i + 3;
    };
    queue1.recordEvent(event);
  }
  event.wait();
  {
    auto command2 = makeCommand(queue2);
    auto v = viewInOut(command2, values);
    command2 << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      v(iter) = v(iter) * 2;
    };
    queue2.recordEvent(event);
  }
  event.wait();
  //![SampleRunQueueEventSample1]

  // Vérifie les valeurs
  for (Integer i = 0; i < nb_value; ++i) {
    Int32 v = values(i);
    Int32 expected_v = (i + 3) * 2;
    vc.areEqual(v, expected_v, "Bad value");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueUnitTest::
_executeTest5()
{
  info() << "Test RunQueue allocation";
  ValueChecker vc(A_FUNCINFO);

  auto queue = makeQueue(m_runner);
  if (queue.isAcceleratorPolicy())
    queue.setMemoryRessource(eMemoryRessource::Device);

  const Int32 nb_value = 100000;

  UniqueArray<Int32> ref_array1(nb_value);
  UniqueArray<Int32> ref_array2(nb_value);

  UniqueArray<Int32> array1(queue.allocationOptions());
  NumArray<Int32, MDDim1> array2(queue.memoryRessource());
  array1.resize(nb_value);
  array2.resize(nb_value);

  vc.areEqual(array2.memoryRessource(), queue.memoryRessource(), "NumArray MemoryRessource");

  {
    auto command = makeCommand(queue);
    auto v1 = viewOut(command, array1);
    auto v2 = viewOut(command, array2);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      v1[i] = i + 3;
      v2[i] = i * 2;
    };
    for (Int32 i = 0; i < nb_value; ++i) {
      ref_array1[i] = i + 3;
      ref_array2[i] = i * 2;
    }
  }

  NumArray<Int32, MDDim1> host_array(eMemoryRessource::Host);
  host_array.resize(nb_value);
  MemoryUtils::copy(host_array.to1DSmallSpan(), array1.constSmallSpan(), &queue);
  vc.areEqual(host_array.to1DSpan(), ref_array1.span(), "Array1");

  host_array.copy(array2);
  vc.areEqual(host_array.to1DSpan(), ref_array2.span(), "Array2");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
