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
  template<typename DataType> void _executeTestDataType(Int32 nb_iteration);

 private:

  void executeTest2(Int32 nb_iteration);
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
  executeTest2(2);
}

void AcceleratorScanUnitTest::
executeTest2(Int32 nb_iteration)
{
  _executeTestDataType<Int64>(nb_iteration);
  _executeTestDataType<Int32>(nb_iteration);
  _executeTestDataType<double>(nb_iteration);
}

template <typename DataType> void AcceleratorScanUnitTest::
_executeTestDataType(Int32 nb_iteration)
{
  ValueChecker vc(A_FUNCINFO);

  info() << "Execute Scan Test1";

  constexpr Int32 n1 = 15; //3000000;

  NumArray<DataType, MDDim1> t1(n1);
  NumArray<DataType, MDDim1> t2(n1);
  ConstMemoryView t1_mem_view(makeMemoryView(t1.to1DSpan()));

  for (Int32 i = 0; i < n1; ++i) {
    int to_add = 2 + (rand() % 32);
    DataType v = static_cast<DataType>(to_add + ((i * 2) % 257));
    t1[i] = v;
    t2[i] = 0;
  }
  info() << "T1=" << t1.to1DSpan();
  NumArray<DataType, MDDim1> expected_t2(n1);
  // Effectue la version séquentielle pour test
  {
    DataType sum = 0;
    for (Int32 i = 0; i < n1; ++i) {
      expected_t2[i] = sum;
      sum += t1[i];
    }
  }

  for (int z = 0; z < nb_iteration; ++z) {
    ax::ScannerSum<DataType> scanner_sum(m_queue);
    scanner_sum.exclusiveSum(t1, t2);
  }
  info() << "T2=" << t2.to1DSpan();
  info() << "Expected_T2=" << expected_t2.to1DSpan();
  vc.areEqualArray(t2.to1DSpan(), expected_t2.to1DSpan(), "ExclusiveScan Sum");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
