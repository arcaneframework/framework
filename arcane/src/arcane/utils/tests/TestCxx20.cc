// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/ArcaneCxx20.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

template <class T>
concept integral = std::is_integral_v<T>;

TEST(TestCxx20,Atomic)
{
  Int32 x = 25;
  std::atomic_ref<Int32> ax(x);
  ax.fetch_add(32);
  ASSERT_EQ(x,57);
}

namespace
{
template<integral DataType> DataType _testAdd(DataType a,DataType b)
{
  return a+b;
}
}

TEST(TestCxx20,Concept)
{
  Int32 a = 12;
  Int32 b = -48;
  ASSERT_EQ(_testAdd(a,b),(a+b));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
