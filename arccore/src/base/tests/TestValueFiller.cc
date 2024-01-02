// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/ValueFiller.h"
#include "arccore/base/CoreArray.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{

template <typename DataType>
void _doTest(Int64 nb_value)
{
  const Int64 rng_seed{ 512515 };
  CoreArray<DataType> values;
  values.resize(nb_value);
  ValueFiller::fillRandom(rng_seed, values.view());
  if (nb_value < 5)
    std::cout << "Values=" << values.view() << "\n";
}
} // namespace

TEST(ValueFiller, Misc)
{
  std::array<Int64, 2> sizes = { 4, 5640 };
  for (Int64 n : sizes) {
    _doTest<char>(n);
    _doTest<signed char>(n);
    _doTest<unsigned char>(n);
    _doTest<short>(n);
    _doTest<unsigned short>(n);
    _doTest<int>(n);
    _doTest<unsigned int>(n);
    _doTest<long>(n);
    _doTest<unsigned long>(n);
    _doTest<long long>(n);
    _doTest<unsigned long long>(n);
    _doTest<float>(n);
    _doTest<double>(n);
    _doTest<long double>(n);
    _doTest<Float16>(n);
    _doTest<BFloat16>(n);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
