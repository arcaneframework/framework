// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorReduceUnitTest.cc                                (C) 2000-2023 */
/*                                                                           */
/* Service de test des réductions sur accélérateur.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"

#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/Memory.h"

#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLoop.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/tests/accelerator/AcceleratorScanUnitTest_axl.h"
#include "arcane/accelerator/Scan.h"

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
class AcceleratorScanUnitTest
: public ArcaneAcceleratorScanUnitTestObject
{
 public:

  explicit AcceleratorScanUnitTest(const ServiceBuildInfo& cb);
  ~AcceleratorScanUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  ax::RunQueue* m_queue = nullptr;

 public:

  void _executeTest1();
  template <typename DataType> void _executeTestDataType(Int32 size, Int32 nb_iteration);

 private:

  void executeTest2(Int32 size, Int32 nb_iteration);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(AcceleratorScanUnitTest, IUnitTest, AcceleratorScanUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorScanUnitTest::
AcceleratorScanUnitTest(const ServiceBuildInfo& sb)
: ArcaneAcceleratorScanUnitTestObject(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorScanUnitTest::
~AcceleratorScanUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorScanUnitTest::
initializeTest()
{
  m_queue = subDomain()->acceleratorMng()->defaultQueue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorScanUnitTest::
executeTest()
{
  executeTest2(15, 10);
  executeTest2(1000000, 1);
}

void AcceleratorScanUnitTest::
executeTest2(Int32 size, Int32 nb_iteration)
{
  _executeTestDataType<Int64>(size, nb_iteration);
  _executeTestDataType<Int32>(size, nb_iteration);
  _executeTestDataType<double>(size, nb_iteration);
}

template <typename DataType> void AcceleratorScanUnitTest::
_executeTestDataType(Int32 size, Int32 nb_iteration)
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Scan Test1";

  constexpr Int32 min_size_display = 100;
  const Int32 n1 = size;

  NumArray<DataType, MDDim1> t1(n1);
  NumArray<DataType, MDDim1> t2(n1);
  ConstMemoryView t1_mem_view(makeMemoryView(t1.to1DSpan()));

  for (Int32 i = 0; i < n1; ++i) {
    int to_add = 2 + (rand() % 32);
    DataType v = static_cast<DataType>(to_add + ((i * 2) % 2348));
    t1[i] = v;
    t2[i] = 0;
  }
  if (n1 < min_size_display) {
    info() << "T1=" << t1.to1DSpan();
  }
  NumArray<DataType, MDDim1> expected_sum(n1);
  NumArray<DataType, MDDim1> expected_min(n1);
  // Effectue la version séquentielle pour test
  {
    DataType sum_value = 0;
    DataType min_value = std::numeric_limits<DataType>::max();
    for (Int32 i = 0; i < n1; ++i) {
      expected_sum[i] = sum_value;
      expected_min[i] = min_value;

      sum_value = sum_value + t1[i];
      min_value = math::min(min_value, t1[i]);
    }
  }

  if (n1 < min_size_display) {
    info() << "Expected_Sum=" << expected_sum.to1DSpan();
    info() << "Expected_Min=" << expected_min.to1DSpan();
  }

  // Teste la somme
  {
    info() << "Check exclusive sum";
    for (int z = 0; z < nb_iteration; ++z) {
      ax::Scanner<DataType> scanner;
      scanner.exclusiveSum(m_queue, t1, t2);
    }
    if (n1 < min_size_display) {
      info() << "T2=" << t2.to1DSpan();
    }
    vc.areEqualArray(t2.to1DSpan(), expected_sum.to1DSpan(), "ExclusiveScan Sum");
  }

  // Teste le minimum
  {
    info() << "Check exclusive min";
    for (int z = 0; z < nb_iteration; ++z) {
      ax::Scanner<DataType> scanner;
      scanner.exclusiveMin(m_queue, t1, t2);
    }
    if (n1 < min_size_display) {
      info() << "T2=" << t2.to1DSpan();
    }
    vc.areEqualArray(t2.to1DSpan(), expected_min.to1DSpan(), "ExclusiveScan Min");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
