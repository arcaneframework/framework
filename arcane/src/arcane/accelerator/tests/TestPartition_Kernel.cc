// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"

#include "arcane/accelerator/GenericPartitioner.h"
#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/NumArrayViews.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;
using namespace Arcane::Accelerator;

extern "C++" void _doPartition1(RunQueue queue, SmallSpan<Real> values, SmallSpan<Real> final_values)
{
  std::cout << "DO_Partition_1\n";

  const Int32 nb_value = values.size();
  NumArray<Real, MDDim1> input(nb_value);
  //, { 1.3, 4.5, -1.2, 3.5, 7.0, 4.2, 2.3, 1.6 });
  input.copy(values, queue);
  NumArray<Real, MDDim1> output(nb_value);
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
  std::cout << "NbFirstPart = " << nb_first_part << "\n";
  std::cout << "Input=" << values << "\n";
  std::cout << "Output=" << output.to1DSmallSpan() << "\n";
  std::cout << "Expected=" << final_values << "\n";
  // Expected nb_first_part = 4
  // Expected output : [0]="4.5" [1]="3.5" [2]="7" [3]="4.2" [4]="2.3" [5]="1.6" [6]="-1.2" [7]="1.3"
  ASSERT_EQ(output.to1DSmallSpan(), final_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
