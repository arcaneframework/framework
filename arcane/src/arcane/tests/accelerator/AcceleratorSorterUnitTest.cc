// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorSorterUnitTest.cc                                (C) 2000-2024 */
/*                                                                           */
/* Service de test des algorithmes accélérateurs de tri.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/IUnitTest.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/Memory.h"

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLoop.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/accelerator/Sort.h"

#include <random>

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
class AcceleratorSorterUnitTest
: public BasicService
, public IUnitTest
{
 public:

  explicit AcceleratorSorterUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override;
  void executeTest() override;
  void finalizeTest() override {}

 private:

  ax::RunQueue m_queue;

 public:

  void _executeTest1();
  template <typename DataType> void _executeTestDataType(Int32 size, Int32 test_id);

 private:

  void executeTest2(Int32 size, Int32 test_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(AcceleratorSorterUnitTest,
                        ServiceProperty("AcceleratorSorterUnitTest", ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IUnitTest));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorSorterUnitTest::
AcceleratorSorterUnitTest(const ServiceBuildInfo& sb)
: BasicService(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorSorterUnitTest::
initializeTest()
{
  m_queue = *subDomain()->acceleratorMng()->defaultQueue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorSorterUnitTest::
executeTest()
{
  for (Int32 i = 0; i < 1; ++i) {
    executeTest2(400, i);
    executeTest2(1000000, i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorSorterUnitTest::
executeTest2(Int32 size, Int32 test_id)
{
  _executeTestDataType<Int64>(size, test_id);
  _executeTestDataType<Int32>(size, test_id);
  _executeTestDataType<double>(size, test_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorSorterUnitTest::
_executeTestDataType(Int32 size, Int32 test_id)
{
  ValueChecker vc(A_FUNCINFO);

  RunQueue queue(makeQueue(subDomain()->acceleratorMng()->defaultRunner()));
  queue.setAsync(true);

  info() << "Execute Sorter Test1 size=" << size << " test_id=" << test_id;

  const Int32 n1 = size;

  NumArray<DataType, MDDim1> t1(n1);
  NumArray<DataType, MDDim1> t2(n1);

  std::seed_seq rng_seed{ 37, 49, 23 };
  std::mt19937 randomizer(rng_seed);
  std::uniform_int_distribution<> rng_distrib(0, 32);
  for (Int32 i = 0; i < n1; ++i) {
    int to_add = 2 + (rng_distrib(randomizer));
    DataType v = static_cast<DataType>(to_add + ((i * 2) % 2348));
    t1[i] = v;
    t2[i] = 0;
  }
  switch (test_id) {
  case 0: {
    Arcane::Accelerator::GenericSorter generic_partitioner(m_queue);
    generic_partitioner.apply(n1, t1.to1DSpan().begin(), t2.to1DSpan().begin());
  } break;
  }
  for (Int32 i = 1; i < size; ++i) {
    if (t2[i - 1] > t2[i])
      ARCANE_FATAL("Bad sort i={0} t[i-1]={1} t[i]={2}", i, t2[i - 1], t2[i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
