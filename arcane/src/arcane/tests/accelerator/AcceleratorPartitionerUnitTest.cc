﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorPartitionerUnitTest.cc                           (C) 2000-2024 */
/*                                                                           */
/* Service de test des algorithmes de partitionnement de liste.              */
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

#include "arcane/accelerator/Partitioner.h"

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
class AcceleratorPartitionerUnitTest
: public BasicService
, public IUnitTest
{
 public:

  explicit AcceleratorPartitionerUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override;
  void executeTest() override;
  void finalizeTest() override {}

 private:

  ax::RunQueue* m_queue = nullptr;

 public:

  void _executeTest1();
  template <typename DataType> void _executeTestDataType(Int32 size, Int32 test_id);

 private:

  void executeTest2(Int32 size, Int32 test_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(AcceleratorPartitionerUnitTest,
                        ServiceProperty("AcceleratorPartitionerUnitTest",ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IUnitTest));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorPartitionerUnitTest::
AcceleratorPartitionerUnitTest(const ServiceBuildInfo& sb)
: BasicService(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorPartitionerUnitTest::
initializeTest()
{
  m_queue = subDomain()->acceleratorMng()->defaultQueue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorPartitionerUnitTest::
executeTest()
{
  for (Int32 i = 0; i < 1; ++i) {
    executeTest2(400, i);
    executeTest2(1000000, i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorPartitionerUnitTest::
executeTest2(Int32 size, Int32 test_id)
{
  _executeTestDataType<Int64>(size, test_id);
  _executeTestDataType<Int32>(size, test_id);
  _executeTestDataType<double>(size, test_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorPartitionerUnitTest::
_executeTestDataType(Int32 size, Int32 test_id)
{
  ValueChecker vc(A_FUNCINFO);

  RunQueue queue(makeQueue(subDomain()->acceleratorMng()->defaultRunner()));
  queue.setAsync(true);

  info() << "Execute Partitioner Test1 size=" << size << " test_id=" << test_id;

  constexpr Int32 min_size_display = 100;
  const Int32 n1 = size;

  NumArray<DataType, MDDim1> t1(n1);
  NumArray<DataType, MDDim1> t2(n1);
  NumArray<DataType, MDDim1> expected_t2(n1);
  NumArray<Int16, MDDim1> filter_flags(n1);

  UniqueArray<DataType> result_part1;
  UniqueArray<DataType> result_part2;

  std::seed_seq rng_seed{ 37, 49, 23 };
  std::mt19937 randomizer(rng_seed);
  std::uniform_int_distribution<> rng_distrib(0, 32);
  Int32 list2_index = n1;
  Int32 list1_index = 0;
  for (Int32 i = 0; i < n1; ++i) {
    int to_add = 2 + (rng_distrib(randomizer));
    DataType v = static_cast<DataType>(to_add + ((i * 2) % 2348));
    t1[i] = v;
    t2[i] = 0;

    bool is_filter = (v > static_cast<DataType>(569));
    if (is_filter) {
      expected_t2[list1_index] = v;
      ++list1_index;
    }
    else{
      --list2_index;
      expected_t2[list2_index] = v;
    }
  }
  Int32 expected_nb_list1 = list1_index;
  if (n1 < min_size_display) {
    info() << "T1=" << t1.to1DSpan();
    info() << "Expected NbList1=" << expected_nb_list1;
    info() << "Expected T2=" << expected_t2.to1DSpan();
  }
  switch (test_id) {
  case 0: // Mode avec lambda de filtrage
  {
    auto filter_lambda = [] ARCCORE_HOST_DEVICE(const DataType& x) -> bool {
      return (x > static_cast<DataType>(569));
    };
    //PartitionerLambda filter_lambda;
    Arcane::Accelerator::GenericPartitioner generic_partitioner(m_queue);
    Int32 nb_list1 = 0;
    generic_partitioner.applyIf(n1, t1.to1DSpan().begin(), t2.to1DSpan().begin(), filter_lambda);
    nb_list1 = generic_partitioner.nbFirstPart();
    info() << "NB_List1_accelerator1=" << nb_list1;
    vc.areEqual(nb_list1, expected_nb_list1, "NbList1");
    if (n1 < min_size_display)
      info() << "Out T2=" << t2.to1DSpan();
    vc.areEqualArray(t2.to1DSpan(), expected_t2.to1DSpan(), "OutputList");
  } break;
  }
}

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
