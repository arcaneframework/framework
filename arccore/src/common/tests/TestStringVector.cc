// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/common/StringVector.h"

#include "arccore/base/String.h"

#include "arccore/common/List.h"

using namespace Arcane;

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

  // Test conversion vers/depuis StringList.
  {
    StringList sl3 = v3.toStringList();
    ASSERT_EQ(sl3.count(), 0);

    StringList sl4 = v4.toStringList();
    ASSERT_EQ(sl4.count(), 2);
    ASSERT_EQ(sl4.count(), v4.size());
    ASSERT_EQ(sl4[0], v4[0]);
    ASSERT_EQ(sl4[1], v4[1]);

    StringVector v5(sl3);
    ASSERT_EQ(v5.size(), 0);
    StringList sl5 = v5.toStringList();
    ASSERT_EQ(sl5.count(), 0);

    StringVector v6(sl4);
    ASSERT_EQ(v6.size(), 2);
    ASSERT_EQ(sl4.count(), v6.size());
    ASSERT_EQ(sl4[0], v6[0]);
    ASSERT_EQ(sl4[1], v6[1]);
  }
}
