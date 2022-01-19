// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunQueueUnitTest.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Service de test unitaire des 'RunQueue'.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/Runner.h"
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

  ax::Runner m_runner;

 public:

  void _executeTest1(bool use_priority);
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
  IApplication* app = subDomain()->application();
  const auto& acc_info = app->acceleratorRuntimeInitialisationInfo();
  initializeRunner(m_runner, traceMng(), acc_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueUnitTest::
executeTest()
{
  _executeTest1(false);
  _executeTest1(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunQueueUnitTest::
_executeTest1(bool use_priority)
{
  info() << "Test RunQueue with multiple threads use_priority=" << use_priority;

  Integer nb_thread = 8;
  Integer N = 1000000;

  UniqueArray<NumArray<Int32, 1>> values(8);
  for (Integer i = 0; i < nb_thread; ++i)
    values[i].resize(N);

  auto task_func = [&](Ref<RunQueue> q, int id) {
    info() << "EXECUTE_THREAD_ID=" << id;
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

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
