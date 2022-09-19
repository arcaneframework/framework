// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/ValueConvert.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

TEST(ValueConvert,Basic)
{
  std::cout << "TEST_ValueConvert Basic\n";

  {
    // TODO: tester les autres conversions
    String s = "25e3";
    Int32 x = 0;
    bool is_bad = builtInGetValue(x,s);
    std::cout << "S=" << s << " X=" << x << " is_bad?=" << is_bad << "\n";
    ASSERT_TRUE(is_bad);
  }

  {
    String s2;
    auto v = Convert::Type<Int32>::tryParse(s2);
    ASSERT_FALSE(v.has_value());
  }

  {
    String s2;
    auto v = Convert::Type<Int32>::tryParseIfNotEmpty(s2,4);
    ASSERT_TRUE(v.has_value());
    ASSERT_EQ(v,4);
  }

  {
    String s2("2.3");
    auto v = Convert::Type<Real>::tryParse(s2);
    ASSERT_EQ(v,2.3);
  }

  {
    String s2("2.3w");
    auto v = Convert::Type<Real>::tryParse(s2);
    ASSERT_FALSE(v.has_value());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
