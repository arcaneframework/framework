// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*
  ThatClass& operator=(const ThatClass& from)
  Data* lookup(KeyTypeConstRef id)
  ValueType& lookupValue(KeyTypeConstRef id)
  Data* lookupAdd(KeyTypeConstRef id, const ValueType& value, bool& is_add)
  Data* lookupAdd(KeyTypeConstRef id)
  void nocheckAdd(KeyTypeConstRef id, const ValueType& value)

 */
TEST(TestHashTable, Misc)
{
  {
    HashTableMapT<Int64, String> hash1(50, true);

    ASSERT_EQ(hash1.count(), 0);
    hash1.add(25, "Test1");

    ASSERT_EQ(hash1.count(), 1);

    hash1.add(32, "Test2");
    ASSERT_EQ(hash1.count(), 2);

    hash1.add(32, "Test3");
    ASSERT_EQ(hash1.count(), 2);

    ASSERT_TRUE(hash1.hasKey(32));
    ASSERT_FALSE(hash1.hasKey(47));

    ASSERT_EQ(hash1[32], "Test3");
    ASSERT_EQ(hash1[25], "Test1");

    hash1.remove(32);
    ASSERT_FALSE(hash1.hasKey(32));
    ASSERT_EQ(hash1.count(), 1);

    hash1.add(32, "Test4");
    ASSERT_EQ(hash1.count(), 2);
    ASSERT_EQ(hash1[32], "Test4");

    hash1.clear();
    ASSERT_EQ(hash1.count(), 0);
  }
  {
    HashTableMapT<Int64, Int32> hash2(1050, true);
    int n = 1000;
    for (int i = 0; i < n; ++i)
      hash2.add((i + 1), (i + 1) * 10);
    ASSERT_EQ(hash2.count(), n);

    hash2.clear();
    ASSERT_EQ(hash2.count(), 0);
    int n2 = 2000;
    for (int i = 0; i < n2; ++i)
      hash2.add((i + 1), (i + 1) * 10);
    ASSERT_EQ(hash2.count(), n2);

    hash2.resize(3000, true);
    ASSERT_EQ(hash2.count(), n2);
    for (int i = 0; i < n2; ++i)
      ASSERT_EQ(hash2[i + 1], (i + 1) * 10);

    hash2.rehash();
    ASSERT_EQ(hash2.count(), n2);
    for (int i = 0; i < n2; ++i)
      ASSERT_EQ(hash2[i + 1], (i + 1) * 10);

    HashTableMapT<Int64, Int32> hash3(50, true);
    hash3 = hash2;
    ASSERT_EQ(hash3.count(), n2);
    for (int i = 0; i < n2; ++i)
      ASSERT_EQ(hash3[i + 1], (i + 1) * 10);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
