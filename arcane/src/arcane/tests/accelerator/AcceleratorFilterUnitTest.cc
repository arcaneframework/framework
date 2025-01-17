// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorFilterUnitTest.cc                                (C) 2000-2025 */
/*                                                                           */
/* Service de test des algorithmes de 'Filtrage' sur accélérateur.           */
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
#include "arcane/accelerator/GenericFilterer.h"

#include "arcane/tests/accelerator/AcceleratorFilterUnitTest_axl.h"

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
class AcceleratorFilterUnitTest
: public ArcaneAcceleratorFilterUnitTestObject
{
 public:

  explicit AcceleratorFilterUnitTest(const ServiceBuildInfo& cb);
  ~AcceleratorFilterUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 public:

  void _executeTest1();
  template <typename DataType> void _executeTestDataType(Int32 size, Int32 test_id);

 private:

  void executeTest2(Int32 size, Int32 test_id);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ACCELERATORFILTERUNITTEST(AcceleratorFilterUnitTest, AcceleratorFilterUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorFilterUnitTest::
AcceleratorFilterUnitTest(const ServiceBuildInfo& sb)
: ArcaneAcceleratorFilterUnitTestObject(sb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorFilterUnitTest::
~AcceleratorFilterUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorFilterUnitTest::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorFilterUnitTest::
executeTest()
{
  for (Int32 i = 3; i < 7; ++i) {
    executeTest2(400, i);
    executeTest2(1000000, i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorFilterUnitTest::
executeTest2(Int32 size, Int32 test_id)
{
  _executeTestDataType<Int64>(size, test_id);
  _executeTestDataType<Int32>(size, test_id);
  _executeTestDataType<double>(size, test_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void AcceleratorFilterUnitTest::
_executeTestDataType(Int32 size, Int32 test_id)
{
  ValueChecker vc(A_FUNCINFO);

  RunQueue queue(makeQueue(subDomain()->acceleratorMng()->runner()).addAsync(true));
  if (!queue.isAsync())
    ARCANE_FATAL("Queue is not asynchronous");

  info() << "Execute Filter Test1 size=" << size << " test_id=" << test_id;

  constexpr Int32 min_size_display = 100;
  const Int32 n1 = size;

  NumArray<DataType, MDDim1> t1(n1);
  NumArray<DataType, MDDim1> t2(n1);
  NumArray<DataType, MDDim1> expected_t2(n1);
  NumArray<Int16, MDDim1> filter_flags(n1);

  std::seed_seq rng_seed{ 37, 49, 23 };
  std::mt19937 randomizer(rng_seed);
  std::uniform_int_distribution<> rng_distrib(0, 32);
  Int32 nb_filter = 0;
  for (Int32 i = 0; i < n1; ++i) {
    int to_add = 2 + (rng_distrib(randomizer));
    DataType v = static_cast<DataType>(to_add + ((i * 2) % 2348));
    t1[i] = v;
    t2[i] = 0;

    bool is_filter = (v > static_cast<DataType>(569));
    if (is_filter) {
      expected_t2[nb_filter] = t1[i];
      ++nb_filter;
    }
    filter_flags[i] = (is_filter) ? 1 : 0;
  }
  if (n1 < min_size_display) {
    info() << "T1=" << t1.to1DSpan();
  }
  expected_t2.resize(nb_filter);
  info() << "Expected NbFilter=" << nb_filter;

  switch (test_id) {
  case 3:
  case 4:
  case 5: {
    auto filter_lambda = [] ARCCORE_HOST_DEVICE(const DataType& x) -> bool {
      return (x > static_cast<DataType>(569));
    };
    NumArray<DataType, MDDim1> t1_bis(t1);
    NumArray<DataType, MDDim1> t2_bis(t2);
    Arcane::Accelerator::GenericFilterer generic_filterer(queue);
    Int32 nb_out = 0;
    if (test_id == 3) {
      {
        // Test appel vide
        generic_filterer.applyIf(0, t1.to1DSpan().begin(), t2.to1DSpan().begin(), filter_lambda);
        Int32 nb_out_empty = generic_filterer.nbOutputElement();
        vc.areEqual(0,nb_out_empty,"ApplyIf 0");
      }

      generic_filterer.applyIf(n1, t1.to1DSpan().begin(), t2.to1DSpan().begin(), filter_lambda);
      nb_out = generic_filterer.nbOutputElement();
    }
    if (test_id == 4) {
      generic_filterer.applyIf<DataType>(t1, t2, filter_lambda);
      nb_out = generic_filterer.nbOutputElement();
    }
    if (test_id == 5) {
      SmallSpan<const DataType> input_view = t1;
      SmallSpan<DataType> output_view = t2;
      auto filter_lambda_index = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
        return (input_view[index] > static_cast<DataType>(569));
      };
      auto setter_lambda_index = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
        output_view[output_index] = input_view[input_index];
      };
      {
        // Test appel vide
        generic_filterer.applyWithIndex(0, filter_lambda_index, setter_lambda_index);
        Int32 nb_out_empty = generic_filterer.nbOutputElement();
        vc.areEqual(0,nb_out_empty,"ApplyIf 1");
      }
      generic_filterer.applyWithIndex(input_view.size(), filter_lambda_index, setter_lambda_index);
      nb_out = generic_filterer.nbOutputElement();
    }
    info() << "NB_OUT_accelerator1=" << nb_out;
    vc.areEqual(nb_filter, nb_out, "Filter");
    t2.resize(nb_out);
    vc.areEqualArray(t2.to1DSpan(), expected_t2.to1DSpan(), "OutputArray1");
  } break;
  case 6: // Mode avec flag
  {
    NumArray<DataType, MDDim1> t1_bis(t1);
    NumArray<DataType, MDDim1> t2_bis(t2);

    Arcane::Accelerator::GenericFilterer filterer(queue);
    SmallSpan<const Int16> filter_flags_view = filter_flags;
    filterer.apply(t1.to1DConstSmallSpan(), t2.to1DSmallSpan(), filter_flags_view);
    Int32 nb_out = filterer.nbOutputElement();
    info() << "NB_OUT_accelerator1=" << nb_out;
    vc.areEqual(nb_filter, nb_out, "Filter");
    t2.resize(nb_out);
    vc.areEqualArray(t2.to1DSpan(), expected_t2.to1DSpan(), "OutputArray1 (test 6)");
    // Appelle une deuxième fois l'instance
    filterer.apply(t1_bis.to1DConstSmallSpan(), t2_bis.to1DSmallSpan(), filter_flags_view);
    Int32 nb_out2 = filterer.nbOutputElement();
    info() << "NB_OUT_accelerator2=" << nb_out2;
    t2_bis.resize(nb_out2);
    vc.areEqualArray(t2_bis.to1DSpan(), expected_t2.to1DSpan(), "OutputArray2 (test 6)");
  } break;
  }
}

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
