// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/ValueFiller.h"
#include "arccore/base/CoreArray.h"
#include "arccore/base/BuiltInDataTypeContainer.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;
using namespace Arcane;
using namespace Arcane::Impl;

template <typename DataType>
class TestInstance
{
 public:

  using InstanceType = TestInstance<DataType>;

 public:

  void doTest(Int64 nb_value)
  {
    const Int64 rng_seed{ 512515 };
    CoreArray<DataType> values;
    values.resize(nb_value);
    ValueFiller::fillRandom(rng_seed, values.span());
    if (nb_value < 5)
      std::cout << "Values=" << values.view() << "\n";
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(ValueFiller, Misc)
{
  BuiltInDataTypeContainer<TestInstance> test_container;

  std::array<Int64, 2> sizes = { 4, 5640 };
  for (Int64 n : sizes) {
    test_container.apply([&](auto& x) { x.doTest(n); });
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
