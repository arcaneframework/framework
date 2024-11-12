// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorScanUnitTest.cc                                  (C) 2000-2024 */
/*                                                                           */
/* Service de test des algorithmes de 'Scan' sur accélérateur.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"

#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/ServiceFactory.h"

#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/GenericScanner.h"

#include "arcane/tests/accelerator/AcceleratorScanUnitTest_axl.h"

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

ARCANE_REGISTER_SERVICE_ACCELERATORSCANUNITTEST(AcceleratorScanUnitTest, AcceleratorScanUnitTest);

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

  std::seed_seq rng_seed{ 13, 49, 23 };
  std::mt19937 randomizer(rng_seed);
  std::uniform_int_distribution<> rng_distrib(0, 32);
  for (Int32 i = 0; i < n1; ++i) {
    int to_add = 2 + (rng_distrib(randomizer));
    DataType v = static_cast<DataType>(to_add + ((i * 2) % 2348));
    if ((i % 3) == 0)
      // Pour avoir des nombres négatifs
      v = -v;
    t1[i] = v;
    t2[i] = 0;
  }
  if (n1 < min_size_display) {
    info() << "T1=" << t1.to1DSpan();
  }
  NumArray<DataType, MDDim1> expected_exclusive_sum(n1);
  NumArray<DataType, MDDim1> expected_exclusive_min(n1);
  NumArray<DataType, MDDim1> expected_exclusive_max(n1);
  NumArray<DataType, MDDim1> expected_inclusive_sum(n1);
  NumArray<DataType, MDDim1> expected_inclusive_min(n1);
  NumArray<DataType, MDDim1> expected_inclusive_max(n1);
  // Effectue la version séquentielle pour test
  {
    DataType sum_value = 0;
    DataType min_value = std::numeric_limits<DataType>::max();
    DataType max_value = std::numeric_limits<DataType>::lowest();
    for (Int32 i = 0; i < n1; ++i) {
      expected_exclusive_sum[i] = sum_value;
      expected_exclusive_min[i] = min_value;
      expected_exclusive_max[i] = max_value;

      sum_value = sum_value + t1[i];
      min_value = math::min(min_value, t1[i]);
      max_value = math::max(max_value, t1[i]);

      expected_inclusive_sum[i] = sum_value;
      expected_inclusive_min[i] = min_value;
      expected_inclusive_max[i] = max_value;
    }
  }

  if (n1 < min_size_display) {
    info() << "Expected_ExclusiveSum=" << expected_exclusive_sum.to1DSpan();
    info() << "Expected_ExclusiveMin=" << expected_exclusive_min.to1DSpan();
  }

  ax::GenericScanner generic_scanner(*m_queue);

  // Teste la somme exclusive
  {
    info() << "Check exclusive sum";
    for (int z = 0; z < nb_iteration; ++z) {
      ax::Scanner<DataType> scanner;
      scanner.exclusiveSum(m_queue, t1, t2);
    }
    if (n1 < min_size_display) {
      info() << "T2=" << t2.to1DSpan();
    }
    vc.areEqualArray(t2.to1DSpan(), expected_exclusive_sum.to1DSpan(), "ExclusiveScan Sum");
  }

  // Teste la somme exclusive (V2)
  {
    info() << "Check exclusive sum (V2)";
    for (int z = 0; z < nb_iteration; ++z) {
      ax::ScannerSumOperator<DataType> op;
      DataType init_value = op.defaultValue();
      t2.fill(init_value, m_queue);
      SmallSpan<const DataType> t1_view(t1);
      SmallSpan<DataType> t2_view(t2);
      if ((z % 2) == 0) {
        generic_scanner.applyExclusive(init_value, t1_view, t2_view, op, A_FUNCINFO);
        vc.areEqualArray(t2.to1DSpan(), expected_exclusive_sum.to1DSpan(), "ExclusiveScan Sum V2");
      }
      else {
        generic_scanner.applyInclusive(init_value, t1_view, t2_view, op, A_FUNCINFO);
        vc.areEqualArray(t2.to1DSpan(), expected_inclusive_sum.to1DSpan(), "InclusiveScan Sum V2");
      }
    }
  }

  // Teste le minimum exclusif
  {
    info() << "Check exclusive min";
    for (int z = 0; z < nb_iteration; ++z) {
      ax::Scanner<DataType> scanner;
      scanner.exclusiveMin(m_queue, t1, t2);
    }
    if (n1 < min_size_display) {
      info() << "T2=" << t2.to1DSpan();
    }
    vc.areEqualArray(t2.to1DSpan(), expected_exclusive_min.to1DSpan(), "ExclusiveScan Min");
  }

  // Teste le maximum exclusif
  {
    info() << "Check exclusive max";
    for (int z = 0; z < nb_iteration; ++z) {
      ax::Scanner<DataType> scanner;
      scanner.exclusiveMax(m_queue, t1, t2);
    }
    if (n1 < min_size_display) {
      info() << "T2=" << t2.to1DSpan();
    }
    vc.areEqualArray(t2.to1DSpan(), expected_exclusive_max.to1DSpan(), "ExclusiveScan Max");
  }

  // Teste la somme inclusive
  {
    info() << "Check inclusive sum";
    for (int z = 0; z < nb_iteration; ++z) {
      ax::Scanner<DataType> scanner;
      scanner.inclusiveSum(m_queue, t1, t2);
    }
    if (n1 < min_size_display) {
      info() << "T2=" << t2.to1DSpan();
    }
    vc.areEqualArray(t2.to1DSpan(), expected_inclusive_sum.to1DSpan(), "InclusiveScan Sum");
  }

  // Teste le minimum inclusif
  {
    info() << "Check inclusive min";
    for (int z = 0; z < nb_iteration; ++z) {
      ax::Scanner<DataType> scanner;
      scanner.inclusiveMin(m_queue, t1, t2);
    }
    if (n1 < min_size_display) {
      info() << "T2=" << t2.to1DSpan();
    }
    vc.areEqualArray(t2.to1DSpan(), expected_inclusive_min.to1DSpan(), "InclusiveScan Min");
  }

  // Teste le maximum inclusif
  {
    info() << "Check inclusive max";
    for (int z = 0; z < nb_iteration; ++z) {
      ax::Scanner<DataType> scanner;
      scanner.inclusiveMax(m_queue, t1, t2);
    }
    if (n1 < min_size_display) {
      info() << "T2=" << t2.to1DSpan();
    }
    vc.areEqualArray(t2.to1DSpan(), expected_inclusive_max.to1DSpan(), "InclusiveScan Max");
  }

  {
    info() << "Check inclusive sum with index";
    SmallSpan<const DataType> t1_view(t1);
    SmallSpan<DataType> t2_view(t2);
    auto getter = [=] ARCCORE_HOST_DEVICE(Int32 index) -> DataType {
      return t1_view[index];
    };
    auto setter = [=] ARCCORE_HOST_DEVICE(Int32 index, const DataType& value) {
      t2_view[index] = value;
    };
    ax::GenericScanner scanner(*m_queue);
    ax::ScannerSumOperator<DataType> op;
    DataType init_value = op.defaultValue();

    // Test Exclusive
    t2.fill(init_value, m_queue);
    scanner.applyWithIndexExclusive(n1, init_value, getter, setter, op);
    vc.areEqualArray(t2.to1DSpan(), expected_exclusive_sum.to1DSpan(), "ExclusiveScan Sum WithIndex");

    // Test Inclusive
    t2.fill(init_value, m_queue);
    scanner.applyWithIndexInclusive(n1, init_value, getter, setter, op);
    vc.areEqualArray(t2.to1DSpan(), expected_inclusive_sum.to1DSpan(), "InclusiveScan Sum WithIndex");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
