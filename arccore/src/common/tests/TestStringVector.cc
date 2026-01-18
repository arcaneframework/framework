// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/common/StringVector.h"
#include "arccore/base/String.h"

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(StringVector, Misc1)
{
  StringVector v;
  ASSERT_EQ(v.size(), 0);

  StringVector v2(v);
  ASSERT_EQ(v2.size(), 0);

  StringVector v3;
  v3.add("Titi");
  ASSERT_EQ(v3.size(), 1);
  ASSERT_EQ(v3[0], "Titi");

  StringVector v4(v3);
  ASSERT_EQ(v4.size(), 1);
  ASSERT_EQ(v4[0], "Titi");

  v3.add("Toto");
  ASSERT_EQ(v3.size(), 2);
  ASSERT_EQ(v3[0], "Titi");
  ASSERT_EQ(v3[1], "Toto");

  ASSERT_EQ(v4.size(), 1);
  ASSERT_EQ(v4[0], "Titi");

  v4 = v3;
  ASSERT_EQ(v4.size(), 2);
  ASSERT_EQ(v4[0], "Titi");
  ASSERT_EQ(v4[1], "Toto");

  v3 = v2;
  ASSERT_EQ(v3.size(), 0);
}
