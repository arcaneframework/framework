// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorPartitionerUnitTest.cc                           (C) 2000-2025 */
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

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunQueueBuildInfo.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/GenericPartitioner.h"

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

  ax::RunQueue m_queue;

 public:

  void _executeTest1();
  template <typename DataType> void _executeTestDataType2(Int32 size, Int32 test_id);
  template <typename DataType> void _executeTestDataType3(Int32 size, Int32 test_id);

  void sample1();
  void sample2();
  void sample3();
  void sample4();

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
  m_queue = *(subDomain()->acceleratorMng()->defaultQueue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorPartitionerUnitTest::
executeTest()
{
  for (Int32 i = 0; i < 2; ++i) {
    executeTest2(400, i);
    executeTest2(1000000, i);
  }
  sample1();
  sample2();
  sample3();
  sample4();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorPartitionerUnitTest::
executeTest2(Int32 size, Int32 test_id)
{
  _executeTestDataType2<Int64>(size, test_id);
  _executeTestDataType2<Int32>(size, test_id);
  _executeTestDataType2<double>(size, test_id);

  _executeTestDataType3<Int64>(size, test_id);
  _executeTestDataType3<Int32>(size, test_id);
  _executeTestDataType3<double>(size, test_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Teste le partitionnement en 2 parties.
 */
template <typename DataType> void AcceleratorPartitionerUnitTest::
_executeTestDataType2(Int32 size, Int32 test_id)
{
  ValueChecker vc(A_FUNCINFO);

  RunQueue queue(makeQueue(subDomain()->acceleratorMng()->defaultRunner()));
  queue.setAsync(true);

  info() << "Execute Partitioner TestDataType2 size=" << size << " test_id=" << test_id;

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
    else {
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
    //![TestSampleListPartitionerTwoWayIf]
    auto filter_lambda = [] ARCCORE_HOST_DEVICE(DataType x) -> bool {
      return (x > static_cast<DataType>(569));
    };
    Arcane::Accelerator::GenericPartitioner generic_partitioner(m_queue);
    {
      // Teste liste vide
      generic_partitioner.applyIf(0, t1.to1DSpan().begin(), t2.to1DSpan().begin(), filter_lambda);
      Int32 nb_list1 = generic_partitioner.nbFirstPart();
      vc.areEqual(0,nb_list1,"applyIf 0");
    }
    Int32 nb_list1 = 0;
    generic_partitioner.applyIf(n1, t1.to1DSpan().begin(), t2.to1DSpan().begin(), filter_lambda);
    nb_list1 = generic_partitioner.nbFirstPart();
    info() << "NB_List1_accelerator1=" << nb_list1;
    //![TestSampleListPartitionerTwoWayIf]
    vc.areEqual(nb_list1, expected_nb_list1, "NbList1");
    if (n1 < min_size_display)
      info() << "Out T2=" << t2.to1DSpan();
    vc.areEqualArray(t2.to1DSpan(), expected_t2.to1DSpan(), "OutputList (1)");
  } break;
  case 1: // Mode avec lambda de filtrage et index
  {
    //![TestSampleListPartitionerTwoWayIndex]
    SmallSpan<const DataType> t1_view(t1.to1DSmallSpan());
    SmallSpan<DataType> t2_view(t2.to1DSmallSpan());
    auto filter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      return (t1_view[index] > static_cast<DataType>(569));
    };
    auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      t2_view[output_index] = t1_view[input_index];
    };
    Arcane::Accelerator::GenericPartitioner generic_partitioner(m_queue);
    {
      // Teste liste vide
      Int32 nb_list1 = 0;
      generic_partitioner.applyWithIndex(0, setter_lambda, filter_lambda);
      nb_list1 = generic_partitioner.nbFirstPart();
      vc.areEqual(0,nb_list1,"applyIf 1");
    }
    Int32 nb_list1 = 0;
    generic_partitioner.applyWithIndex(n1, setter_lambda, filter_lambda);
    nb_list1 = generic_partitioner.nbFirstPart();
    info() << "NB_List1_accelerator2=" << nb_list1;
    //![TestSampleListPartitionerTwoWayIndex]
    vc.areEqual(nb_list1, expected_nb_list1, "NbList1");
    if (n1 < min_size_display)
      info() << "Out T2=" << t2.to1DSpan();
    vc.areEqualArray(t2.to1DSpan(), expected_t2.to1DSpan(), "OutputList (2)");
  } break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Teste le partitionnement en 3 parties.
 */
template <typename DataType> void AcceleratorPartitionerUnitTest::
_executeTestDataType3(Int32 size, Int32 test_id)
{
  ValueChecker vc(A_FUNCINFO);

  RunQueue queue(makeQueue(subDomain()->acceleratorMng()->defaultRunner()));
  queue.setAsync(true);

  info() << "Execute Partitioner TestDataType3 size=" << size << " test_id=" << test_id;

  constexpr Int32 min_size_display = 100;
  const Int32 n1 = size;

  NumArray<DataType, MDDim1> t1(n1);
  NumArray<DataType, MDDim1> t2(n1);
  NumArray<DataType, MDDim1> t3(n1);
  NumArray<DataType, MDDim1> t4(n1);
  NumArray<DataType, MDDim1> expected_t2(n1);
  NumArray<DataType, MDDim1> expected_t3(n1);
  NumArray<DataType, MDDim1> expected_t4(n1);
  NumArray<Int16, MDDim1> filter_flags(n1);

  UniqueArray<DataType> result_part1;
  UniqueArray<DataType> result_part2;

  std::seed_seq rng_seed{ 37, 49, 23 };
  std::mt19937 randomizer(rng_seed);
  std::uniform_int_distribution<> rng_distrib(0, 32);
  Int32 list2_index = 0;
  Int32 list1_index = 0;
  Int32 unselected_index = 0;
  for (Int32 i = 0; i < n1; ++i) {
    int to_add = 2 + (rng_distrib(randomizer));
    DataType v = static_cast<DataType>(to_add + ((i * 2) % 2348));
    t1[i] = v;
    DataType unset_value = static_cast<DataType>(-1);
    t2[i] = unset_value;
    t3[i] = unset_value;
    t4[i] = unset_value;
    bool is_filter1 = (v > static_cast<DataType>(655));
    bool is_filter2 = (v < static_cast<DataType>(469));
    //info() << "Value I=" << i << " v=" << v << " is_1=" << is_filter1 << " is_2=" << is_filter2;
    if (is_filter1) {
      expected_t2[list1_index] = v;
      ++list1_index;
    }
    else {
      if (is_filter2) {
        expected_t3[list2_index] = v;
        ++list2_index;
      }
      else {
        expected_t4[unselected_index] = v;
        ++unselected_index;
      }
    }
  }
  Int32 expected_nb_list1 = list1_index;
  Int32 expected_nb_list2 = list2_index;
  Int32 expected_nb_unselected = unselected_index;
  expected_t2.resize(expected_nb_list1);
  expected_t3.resize(expected_nb_list2);
  expected_t4.resize(expected_nb_unselected);
  info() << "Expected NbList1=" << expected_nb_list1;
  info() << "Expected NbList2=" << expected_nb_list2;
  info() << "Expected NbUnselectedList1=" << expected_nb_unselected;
  if (n1 < min_size_display) {
    info() << "T1=" << t1.to1DSpan();
    info() << "Expected T2=" << expected_t2.to1DSpan();
    info() << "Expected T3=" << expected_t3.to1DSpan();
    info() << "Expected T4=" << expected_t4.to1DSpan();
  }

  Int32 nb_part1 = -1;
  Int32 nb_part2 = -2;

  switch (test_id) {
  case 0: // Mode avec lambda de filtrage
  {
    //![TestSampleListPartitionerThreeWayIf]
    auto filter1_lambda = [] ARCCORE_HOST_DEVICE(DataType x) -> bool {
      return (x > static_cast<DataType>(655));
    };
    auto filter2_lambda = [] ARCCORE_HOST_DEVICE(DataType x) -> bool {
      return (x < static_cast<DataType>(469));
    };
    Arcane::Accelerator::GenericPartitioner generic_partitioner(m_queue);
    {
      // Test liste vide
      generic_partitioner.applyIf(0, t1.to1DSpan().begin(), t2.to1DSpan().begin(),
                                  t3.to1DSpan().begin(), t4.to1DSpan().begin(),
                                  filter1_lambda, filter2_lambda, A_FUNCINFO);
      SmallSpan<const Int32> nb_parts = generic_partitioner.nbParts();
      vc.areEqual(nb_parts[0], 0, "Part1 0");
      vc.areEqual(nb_parts[1], 0, "Part2 0");
    }
    generic_partitioner.applyIf(n1, t1.to1DSpan().begin(), t2.to1DSpan().begin(),
                                t3.to1DSpan().begin(), t4.to1DSpan().begin(),
                                filter1_lambda, filter2_lambda, A_FUNCINFO);
    SmallSpan<const Int32> nb_parts = generic_partitioner.nbParts();
    nb_part1 = nb_parts[0];
    nb_part2 = nb_parts[1];
    //![TestSampleListPartitionerThreeWayIf]
  } break;
  case 1: // Mode avec lambda de filtrage et index
  {
    //![TestSampleListPartitionerThreeWayIndex]
    info() << "Doing test with index and setter/lambda";
    SmallSpan<const DataType> t1_view(t1.to1DSmallSpan());
    SmallSpan<DataType> t2_view(t2.to1DSmallSpan());
    SmallSpan<DataType> t3_view(t3.to1DSmallSpan());
    SmallSpan<DataType> t4_view(t4.to1DSmallSpan());
    auto filter1_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      return (t1_view[index] > static_cast<DataType>(655));
    };
    auto filter2_lambda = [=] ARCCORE_HOST_DEVICE(Int32 index) -> bool {
      return (t1_view[index] < static_cast<DataType>(469));
    };
    auto first_setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      t2_view[output_index] = t1_view[input_index];
    };
    auto second_setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      t3_view[output_index] = t1_view[input_index];
    };
    auto unselected_setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
      t4_view[output_index] = t1_view[input_index];
    };

    Arcane::Accelerator::GenericPartitioner generic_partitioner(m_queue);
    {
      // Test liste vide
      generic_partitioner.applyWithIndex(0, first_setter_lambda,
                                         second_setter_lambda, unselected_setter_lambda,
                                         filter1_lambda, filter2_lambda, A_FUNCINFO);
      SmallSpan<const Int32> nb_parts = generic_partitioner.nbParts();
      vc.areEqual(nb_parts[0], 0, "Part1 1");
      vc.areEqual(nb_parts[1], 0, "Part2 1");
    }
    generic_partitioner.applyWithIndex(n1, first_setter_lambda,
                                       second_setter_lambda, unselected_setter_lambda,
                                       filter1_lambda, filter2_lambda, A_FUNCINFO);
    SmallSpan<const Int32> nb_parts = generic_partitioner.nbParts();
    nb_part1 = nb_parts[0];
    nb_part2 = nb_parts[1];
    //![TestSampleListPartitionerThreeWayIndex]
  } break;
  }

  const Int32 nb_unselected = n1 - (nb_part1 + nb_part2);
  info() << "NB_List1_1=" << nb_part1;
  t2.resize(nb_part1);
  info() << "NB_List1_2=" << nb_part2;
  t3.resize(nb_part2);
  t4.resize(nb_unselected);
  if (n1 < min_size_display) {
    info() << "Out T2=" << t2.to1DSpan();
    info() << "Out T3=" << t3.to1DSpan();
    info() << "Out T4=" << t4.to1DSpan();
  }
  vc.areEqual(nb_part1, expected_nb_list1, "NbList1");
  vc.areEqual(nb_part2, expected_nb_list2, "NbList2");
  vc.areEqual(nb_unselected, expected_nb_unselected, "NbUnselected");
  vc.areEqualArray(t2.to1DSpan(), expected_t2.to1DSpan(), "OutputList1 (1)");
  vc.areEqualArray(t3.to1DSpan(), expected_t3.to1DSpan(), "OutputList2 (1)");
  vc.areEqualArray(t4.to1DSpan(), expected_t4.to1DSpan(), "OutputList3 (1)");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorPartitionerUnitTest::
sample1()
{
  info() << "Sample1";
  //![SampleListPartitionerTwoWayIndex]
  // Sample to do a partition of an array.
  // All elements greater than 2.0 will be put at the beginning of the list
  // The remaining elements will be at the end of the list in reverse order
  Arcane::Accelerator::RunQueue queue(*subDomain()->acceleratorMng()->defaultQueue());
  const Int32 nb_value = 6;
  Arcane::NumArray<Real, MDDim1> input(nb_value, { 1.3, 4.5, -1.2, 3.5, 7.0, 4.2 });
  Arcane::NumArray<Real, MDDim1> output(nb_value);
  auto input_values = viewIn(queue, input);
  auto output_values = viewOut(queue, output);
  auto select_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index) {
    return input_values[input_index] > 2.0;
  };
  auto setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
    output_values[output_index] = input_values[input_index];
  };
  Arcane::Accelerator::GenericPartitioner partitioner(queue);
  partitioner.applyWithIndex(nb_value, setter_lambda, select_lambda, A_FUNCINFO);
  Int32 nb_first_part = partitioner.nbFirstPart();
  info() << "NbFirstPart = " << nb_first_part;
  info() << "Output=" << output.to1DSmallSpan();
  // Expected nb_first_part = 4
  // Expected output : [0]="4.5" [1]="3.5" [2]="7" [3]="4.2" [4]="-1.2" [5]="1.3"
  //![SampleListPartitionerTwoWayIndex]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorPartitionerUnitTest::
sample2()
{
  info() << "Sample2";
  //![SampleListPartitionerTwoWayIf]
  // Sample to do a partition of an array.
  // All elements greater than 2.0 will be put at the beginning of the list
  // The remaining elements will be at the end of the list in reverse order
  Arcane::Accelerator::RunQueue queue(*subDomain()->acceleratorMng()->defaultQueue());
  const Int32 nb_value = 6;
  Arcane::NumArray<Real, MDDim1> input(nb_value, { 1.3, 4.5, -1.2, 3.5, 7.0, 4.2 });
  Arcane::NumArray<Real, MDDim1> output(nb_value);
  auto input_values = input.to1DSmallSpan();
  auto output_values = output.to1DSmallSpan();
  auto select_lambda = [=] ARCCORE_HOST_DEVICE(Real value) {
    return value > 2.0;
  };
  Arcane::Accelerator::GenericPartitioner partitioner(queue);
  partitioner.applyIf(nb_value, input_values.begin(), output_values.begin(),
                      select_lambda, A_FUNCINFO);
  Int32 nb_first_part = partitioner.nbFirstPart();
  info() << "NbFirstPart = " << nb_first_part;
  info() << "Output=" << output.to1DSmallSpan();
  // Expected nb_first_part = 4
  // Expected output : [0]="4.5" [1]="3.5" [2]="7" [3]="4.2" [4]="-1.2" [5]="1.3"
  //![SampleListPartitionerTwoWayIf]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorPartitionerUnitTest::
sample3()
{
  info() << "Sample3";
  //![SampleListPartitionerThreeWayIndex]
  // Sample to do a partition of an array.
  // All elements greater than 2.0 will be put at the beginning of the list
  // The remaining elements will be at the end of the list in reverse order
  Arcane::Accelerator::RunQueue queue(*subDomain()->acceleratorMng()->defaultQueue());
  const Int32 nb_value = 7;
  Arcane::NumArray<Real, MDDim1> input(nb_value, { 1.3, 4.5, -1.2, 3.5, 7.0, 4.2, 9.5 });
  Arcane::NumArray<Real, MDDim1> output1(nb_value);
  Arcane::NumArray<Real, MDDim1> output2(nb_value);
  Arcane::NumArray<Real, MDDim1> output_unselected(nb_value);
  auto input_values = viewIn(queue, input);
  auto output1_values = viewOut(queue, output1);
  auto output2_values = viewOut(queue, output2);
  auto output_unselected_values = viewOut(queue, output_unselected);
  auto select1_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index) {
    return input_values[input_index] > 6.0;
  };
  auto select2_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index) {
    return input_values[input_index] < 3.0;
  };
  auto setter1_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
    output1_values[output_index] = input_values[input_index];
  };
  auto setter2_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
    output2_values[output_index] = input_values[input_index];
  };
  auto unselected_setter_lambda = [=] ARCCORE_HOST_DEVICE(Int32 input_index, Int32 output_index) {
    output_unselected_values[output_index] = input_values[input_index];
  };
  Arcane::Accelerator::GenericPartitioner partitioner(queue);
  partitioner.applyWithIndex(nb_value, setter1_lambda, setter2_lambda, unselected_setter_lambda,
                             select1_lambda, select2_lambda, A_FUNCINFO);
  SmallSpan<const Int32> nb_parts = partitioner.nbParts();
  info() << "NbFirstPart = " << nb_parts[0] << " NbSecondPart=" << nb_parts[1];
  output1.resize(nb_parts[0]);
  output2.resize(nb_parts[1]);
  output_unselected.resize(nb_value - (nb_parts[0] + nb_parts[1]));
  info() << "Output1=" << output1.to1DSmallSpan();
  info() << "Output2=" << output2.to1DSmallSpan();
  info() << "UnselectedOutput=" << output_unselected.to1DSmallSpan();
  // Expected output :
  // NbFirstPart = 2 NbSecondPart=2
  // Output1=[0]="7" [1]="9.5"
  // Output2=[0]="1.3" [1]="-1.2"
  // UnselectedOutput=[0]="4.5" [1]="3.5" [2]="4.2"
  //![SampleListPartitionerThreeWayIndex]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorPartitionerUnitTest::
sample4()
{
  info() << "Sample4";
  //![SampleListPartitionerThreeWayIf]
  // Sample to do a partition of an array.
  // All elements greater than 2.0 will be put at the beginning of the list
  // The remaining elements will be at the end of the list in reverse order
  Arcane::Accelerator::RunQueue queue(*subDomain()->acceleratorMng()->defaultQueue());
  const Int32 nb_value = 7;
  Arcane::NumArray<Real, MDDim1> input(nb_value, { 1.3, 4.5, -1.2, 3.5, 7.0, 4.2, 9.5 });
  Arcane::NumArray<Real, MDDim1> output1(nb_value);
  Arcane::NumArray<Real, MDDim1> output2(nb_value);
  Arcane::NumArray<Real, MDDim1> output_unselected(nb_value);
  auto input_values = input.to1DSmallSpan();
  auto output1_values = output1.to1DSmallSpan();
  auto output2_values = output2.to1DSmallSpan();
  auto output_unselected_values = output_unselected.to1DSmallSpan();
  auto select1_lambda = [=] ARCCORE_HOST_DEVICE(Real value) {
    return value > 6.0;
  };
  auto select2_lambda = [=] ARCCORE_HOST_DEVICE(Real value) {
    return value < 3.0;
  };
  Arcane::Accelerator::GenericPartitioner partitioner(queue);
  partitioner.applyIf(nb_value, input_values.begin(),
                      output1_values.begin(), output2_values.begin(), output_unselected_values.begin(),
                      select1_lambda, select2_lambda, A_FUNCINFO);
  SmallSpan<const Int32> nb_parts = partitioner.nbParts();
  info() << "NbFirstPart = " << nb_parts[0] << " NbSecondPart=" << nb_parts[1];
  output1.resize(nb_parts[0]);
  output2.resize(nb_parts[1]);
  output_unselected.resize(nb_value - (nb_parts[0] + nb_parts[1]));
  info() << "Output1=" << output1.to1DSmallSpan();
  info() << "Output2=" << output2.to1DSmallSpan();
  info() << "UnselectedOutput=" << output_unselected.to1DSmallSpan();
  // Expected output :
  // NbFirstPart = 2 NbSecondPart=2
  // Output1=[0]="7" [1]="9.5"
  // Output2=[0]="1.3" [1]="-1.2"
  // UnselectedOutput=[0]="4.5" [1]="3.5" [2]="4.2"
  //![SampleListPartitionerThreeWayIf]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
